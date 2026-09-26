import mimetypes
import os
import re
import time
from playwright.sync_api import sync_playwright

DOWNLOAD_DIR = os.path.abspath("pdf_store")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
TOTAL_PAGES = 36


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def detect_file_extension(content_bytes, content_type_header):
    """Detect actual file extension using magic bytes and Content-Type headers."""
    content_type = (content_type_header or "").lower()

    # 1. HTML check (skip or handle separately)
    if (
        content_bytes.startswith(b"<!DOCTYPE")
        or content_bytes.startswith(b"<html")
        or "text/html" in content_type
    ):
        return None  # Indicates HTML page, not a downloadable file

    # 2. PDF Magic Bytes: %PDF
    if content_bytes.startswith(b"%PDF"):
        return ".pdf"

    # 3. Office OpenXML / ZIP (xlsx, docx)
    if content_bytes.startswith(b"PK\x03\x04"):
        if "spreadsheet" in content_type or "excel" in content_type:
            return ".xlsx"
        elif "word" in content_type:
            return ".docx"
        return ".xlsx"  # ARAI tables using OpenXML are almost always Excel

    # 4. Legacy Office Formats (xls, doc)
    if content_bytes.startswith(b"\xd0\xcf\x11\xe0"):
        if "excel" in content_type or "ms-excel" in content_type:
            return ".xls"
        elif "msword" in content_type:
            return ".doc"
        return ".xls"

    # Fallback using Mappings
    if "pdf" in content_type:
        return ".pdf"
    elif "excel" in content_type or "spreadsheet" in content_type:
        return ".xlsx"

    return ".pdf"  # Default fallback


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()

        print("Navigating to initial ARAI page...")
        page.goto("https://www.araiindia.com/downloads/ais-downloads", wait_until="networkidle")

        for page_num in range(1, TOTAL_PAGES + 1):
            print(f"\n--- Processing Page {page_num}/{TOTAL_PAGES} ---")

            if page_num > 1:
                page_changed = page.evaluate(f"""
                    (targetPage) => {{
                        const links = Array.from(document.querySelectorAll('a, button, li'));
                        const exactLink = links.find(el => el.textContent.trim() === String(targetPage));
                        if (exactLink) {{
                            exactLink.click();
                            return true;
                        }}
                        const nextBtn = links.find(el => el.textContent.includes('Next') || el.textContent.includes('›'));
                        if (nextBtn) {{
                            nextBtn.click();
                            return true;
                        }}
                        return false;
                    }}
                """, page_num)

                if not page_changed:
                    print(f"[WARN] Could not trigger pagination to Page {page_num}.")
                    break

                page.wait_for_load_state("networkidle")
                time.sleep(2)

            try:
                page.wait_for_selector("table tr", timeout=15000)
            except Exception:
                print(f"[WARN] Timeout waiting for table on Page {page_num}")
                continue

            rows = page.query_selector_all("table tbody tr")
            if not rows:
                rows = page.query_selector_all("table tr")

            data_rows = [r for r in rows if r.query_selector_all("td")]
            print(f"Found {len(data_rows)} items on page {page_num}")

            if len(data_rows) == 0:
                print(f"No data rows found on page {page_num}. Ending.")
                break

            for index, row in enumerate(data_rows):
                btn = row.query_selector("a, button, .download-icon, svg")
                if not btn:
                    continue

                cells = row.query_selector_all("td")
                raw_title = f"AIS_Doc_P{page_num}_Item{index+1}"
                if len(cells) >= 2:
                    code_text = cells[1].inner_text().strip()
                    if code_text:
                        raw_title = code_text

                clean_name = sanitize_filename(raw_title)

                try:
                    pdf_url = None
                    href = btn.get_attribute("href")

                    if href and href != "#" and not href.startswith("javascript"):
                        pdf_url = page.evaluate("(url) => new URL(url, document.baseURI).href", href)
                    else:
                        with context.expect_page(timeout=10000) as new_page_info:
                            btn.click(force=True)

                        new_tab = new_page_info.value
                        new_tab.wait_for_load_state("domcontentloaded")
                        pdf_url = new_tab.url
                        new_tab.close()

                    if pdf_url:
                        response = context.request.get(pdf_url)
                        if response.ok:
                            body_bytes = response.body()
                            content_type = response.headers.get("content-type", "")

                            # Determine extension dynamically
                            ext = detect_file_extension(body_bytes, content_type)

                            if ext is None:
                                print(f"[SKIP/HTML] Link returned web content instead of document: {clean_name}")
                                continue

                            final_filename = f"{clean_name}{ext}"
                            target_path = os.path.join(DOWNLOAD_DIR, final_filename)

                            if os.path.exists(target_path):
                                print(f"[SKIP] Already exists: {final_filename}")
                                continue

                            with open(target_path, "wb") as f:
                                f.write(body_bytes)
                            print(f"[SUCCESS] Saved: {final_filename}")
                        else:
                            print(f"[ERROR] HTTP {response.status}: {clean_name}")

                    time.sleep(0.5)

                except Exception as e:
                    print(f"[WARN] Error on item {index+1}: {e}")

        browser.close()
        print("\nAll 36 pages processed!")


if __name__ == "__main__":
    run()
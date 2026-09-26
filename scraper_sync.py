import os
import re
import json
import requests
import urllib3
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from src.ingestion import IngestionPipeline

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PDF_DIR = "pdf_store"
MANIFEST_FILE = os.path.join(PDF_DIR, "standards_manifest.json")
os.makedirs(PDF_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

JUNK_KEYWORDS = [
    "annual", "report", "spandan", "tender", "newsletter", "hindi", 
    "5yrplan", "form", "brochure", "balance", "financial", "meeting", "notice"
]

FAMILY_PATTERNS = {
    "AIS": r'ais[-_\s]?\d{1,3}',
    "UNECE": r'(ece|unece|un[-_\s]?r\d{1,3})',
    "ASPICE": r'(aspice|pam|prm)'
}

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

def is_valid_standard(url: str, filename: str, family: str) -> bool:
    target_str = f"{url} {filename}".lower()
    
    # 1. Reject if blacklisted
    for junk in JUNK_KEYWORDS:
        if junk in target_str:
            return False

    # 2. Reject if no valid standard regex pattern matches
    pattern = FAMILY_PATTERNS.get(family)
    if pattern and not re.search(pattern, target_str):
        return False

    return True

def crawl_catalog_pages(start_url: str, max_pages: int = 5) -> set:
    """Discovers paginated/sub-category links across the target domain."""
    visited = set()
    to_visit = [start_url]
    discovered_pdfs = set()

    base_domain = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        visited.add(current_url)
        print(f"[Crawler] Fetching page ({len(visited)}/{max_pages}): {current_url}")

        try:
            res = requests.get(current_url, headers=HEADERS, verify=False, timeout=15)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                full_url = urljoin(current_url, href)

                # Extract PDF URLs directly
                if full_url.lower().endswith(".pdf") or ".pdf?" in full_url.lower():
                    discovered_pdfs.add(full_url)
                
                # Discover secondary catalog/pagination pages
                elif urlparse(full_url).netloc == base_domain:
                    href_lower = href.lower()
                    if any(p in href_lower for p in ["page=", "p=", "downloads", "standard", "ais"]):
                        if full_url not in visited and full_url not in to_visit:
                            to_visit.append(full_url)

        except Exception as e:
            print(f"[Crawler Warning] Failed scraping {current_url}: {e}")

    return discovered_pdfs

def scrape_and_download():
    print("[Sync Engine] Starting targeted automotive standards crawl...")
    manifest = load_manifest()
    pipeline = IngestionPipeline()

    sources = [
        {"name": "AIS", "start_url": "https://araiindia.com/downloads"},
    ]

    total_downloaded = 0

    for source in sources:
        family = source["name"]
        print(f"\n--- Scraping {family} Standards Catalog ---")
        pdf_urls = crawl_catalog_pages(source["start_url"], max_pages=8)
        print(f"[Sync Engine] Found {len(pdf_urls)} candidate PDF link(s) across catalog pages.")

        for pdf_url in pdf_urls:
            filename = os.path.basename(urlparse(pdf_url).path)
            if not filename.endswith(".pdf"):
                filename += ".pdf"

            if not is_valid_standard(pdf_url, filename, family):
                continue

            file_path = os.path.join(PDF_DIR, filename)
            doc_id = filename.replace(".pdf", "").upper()

            if os.path.exists(file_path):
                print(f"[Sync Engine] Skip {filename} (Already exists).")
                continue

            print(f"[Sync Engine] Downloading targeted standard: {filename}")
            try:
                pdf_res = requests.get(pdf_url, headers=HEADERS, verify=False, timeout=30)
                if pdf_res.status_code == 200 and len(pdf_res.content) > 1000:
                    with open(file_path, "wb") as f:
                        f.write(pdf_res.content)

                    manifest[doc_id] = {"url": pdf_url, "file_path": file_path, "family": family}
                    save_manifest(manifest)
                    total_downloaded += 1

                    # Vectorize and index into Qdrant
                    pipeline.process_and_index(file_path, {
                        "doc_id": doc_id,
                        "standard_family": family,
                        "url": pdf_url
                    })
            except Exception as e:
                print(f"[Sync Engine Error] Failed downloading {pdf_url}: {e}")

    print(f"\n[Sync Complete] Successfully downloaded and indexed {total_downloaded} new standard PDF(s).")

if __name__ == "__main__":
    scrape_and_download()
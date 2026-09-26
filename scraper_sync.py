import os
import re
import json
import requests
import urllib3
from bs4 import BeautifulSoup
from src.ingestion import IngestionPipeline

# Suppress insecure HTTPS request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PDF_DIR = "pdf_store"
MANIFEST_FILE = os.path.join(PDF_DIR, "standards_manifest.json")
os.makedirs(PDF_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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

def scrape_and_download():
    print("[Sync Engine] Starting automotive standards scraper...")
    manifest = load_manifest()
    pipeline = IngestionPipeline()
    
    # Target catalog sources
    sources = [
        {
            "name": "AIS",
            "url": "https://araiindia.com/downloads",
            "pattern": r".*AIS.*\.pdf$"
        }
    ]

    downloaded_count = 0

    for source in sources:
        print(f"[Sync Engine] Fetching catalog from {source['name']}: {source['url']}")
        try:
            res = requests.get(source['url'], headers=HEADERS, verify=False, timeout=15)
            if res.status_code != 200:
                print(f"[Sync Engine Warning] Received status code {res.status_code} from {source['url']}")
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            links = soup.find_all("a", href=True)
            print(f"[Sync Engine] Found {len(links)} total links on page.")

            for link in links:
                href = link["href"]
                # Match PDF URLs
                if href.lower().endswith(".pdf") or "AIS" in href.upper():
                    pdf_url = href if href.startswith("http") else f"https://araiindia.com{href}"
                    filename = os.path.basename(pdf_url.split("?")[0])
                    if not filename.endswith(".pdf"):
                        filename += ".pdf"

                    file_path = os.path.join(PDF_DIR, filename)

                    if os.path.exists(file_path):
                        print(f"[Sync Engine] Skipping {filename} (Already exists locally).")
                        continue

                    print(f"[Sync Engine] Downloading: {pdf_url}")
                    pdf_res = requests.get(pdf_url, headers=HEADERS, verify=False, timeout=30)
                    if pdf_res.status_code == 200 and len(pdf_res.content) > 1000:
                        with open(file_path, "wb") as f:
                            f.write(pdf_res.content)
                        
                        doc_id = filename.replace(".pdf", "").upper()
                        manifest[doc_id] = {"url": pdf_url, "file_path": file_path}
                        save_manifest(manifest)
                        downloaded_count += 1

                        # Process & Index into Qdrant
                        pipeline.process_and_index(file_path, {
                            "doc_id": doc_id,
                            "standard_family": source["name"],
                            "url": pdf_url
                        })
                    else:
                        print(f"[Sync Engine Warning] Failed to download valid PDF from {pdf_url}")

        except Exception as e:
            print(f"[Sync Engine Error] Failed scraping {source['name']}: {e}")

    print(f"[Sync Engine Complete] Finished sync. Downloaded {downloaded_count} new PDF(s).")

if __name__ == "__main__":
    scrape_and_download()
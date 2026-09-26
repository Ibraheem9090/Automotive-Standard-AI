import os
import re
import requests
import urllib3
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from src.ingestion import IngestionPipeline

# Suppress SSL warnings for ARAI / govt portals with self-signed or expired certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

ARAI_AIS_CATALOG_URL = "https://www.araiindia.com/downloads/ais-downloads"


def discover_all_arai_standards() -> List[Dict[str, Any]]:
    """
    Dynamically crawls ARAI's AIS Downloads portal and extracts all published AIS standards.
    """
    print(f"[Crawler] Scraping ARAI standards catalog from {ARAI_AIS_CATALOG_URL}...")
    discovered_standards = []
    page_num = 1
    
    while True:
        url = f"{ARAI_AIS_CATALOG_URL}?page={page_num}" if page_num > 1 else ARAI_AIS_CATALOG_URL
        try:
            res = requests.get(url, headers=HEADERS, verify=False, timeout=30)
            if res.status_code != 200:
                print(f"[Crawler] Stopped at page {page_num} (HTTP {res.status_code})")
                break
                
            soup = BeautifulSoup(res.text, "html.parser")
            rows = soup.find_all("tr")
            page_items = 0

            for row in rows:
                cols = row.find_all("td")
                pdf_link = row.find("a", href=re.compile(r'\.pdf$', re.IGNORECASE))
                
                if pdf_link:
                    href = pdf_link['href']
                    full_url = urllib.parse.urljoin(ARAI_AIS_CATALOG_URL, href)
                    
                    # Extract document title/code (e.g., AIS-156, AIS-038)
                    code_text = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                    doc_id = re.sub(r'[^A-Za-z0-9_-]', '_', code_text).strip('_')
                    
                    if not doc_id:
                        doc_id = href.split('/')[-1].replace('.pdf', '')

                    discovered_standards.append({
                        "doc_id": doc_id,
                        "standard_family": "AIS",
                        "domain": "ARAI Technical Regulation",
                        "url": full_url,
                        "source_site": "ARAI"
                    })
                    page_items += 1

            # Break loop if no PDF links found on page or pagination ends
            next_page = soup.find("a", string=re.compile(r'next|अगला|>', re.IGNORECASE))
            if not next_page or page_items == 0:
                break
                
            page_num += 1

        except Exception as e:
            print(f"[Crawler] Error scanning page {page_num}: {e}")
            break

    print(f"[Crawler] Total AIS standards discovered on ARAI: {len(discovered_standards)}")
    return discovered_standards


def download_file(url: str, output_path: str) -> bool:
    """Downloads PDF with SSL verification bypassed and custom user-agent."""
    try:
        response = requests.get(url, headers=HEADERS, verify=False, timeout=30, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"[HTTP {response.status_code}] Download failed: {url}")
            return False
    except Exception as e:
        print(f"[Error] Failed downloading {url}: {e}")
        return False


def run_full_extraction(pipeline: IngestionPipeline, static_config_path: str = "config/standards_sources.json"):
    all_targets = []

    # 1. Dynamically discover ALL AIS standards directly from ARAI website
    arai_standards = discover_all_arai_standards()
    all_targets.extend(arai_standards)

    # 2. Add static non-scrapable standards (ASPICE PAM, ISO specs requiring manual paths)
    if os.path.exists(static_config_path):
        import json
        with open(static_config_path, "r") as f:
            static_sources = json.load(f)
            # Add sources not covered by the ARAI web crawler
            for item in static_sources:
                if item.get("standard_family") != "AIS":
                    all_targets.append(item)

    print(f"\n[Pipeline] Ready to process {len(all_targets)} total standards into Qdrant...\n")

    # 3. Process every discovered standard
    for item in all_targets:
        doc_id = item["doc_id"]
        pdf_path = f"pdf_store/{doc_id}.pdf"
        
        print(f"--> Processing [{doc_id}] from {item['source_site']}...")

        if not os.path.exists(pdf_path):
            if "url" in item and item["url"]:
                success = download_file(item["url"], pdf_path)
                if not success:
                    continue
            else:
                print(f"[Skipped] Local file missing and no URL for {doc_id}")
                continue

        # Pass PDF to parser, chunker, NVIDIA Nemotron embedding, and Qdrant DB
        pipeline.process_and_index(pdf_path, item)
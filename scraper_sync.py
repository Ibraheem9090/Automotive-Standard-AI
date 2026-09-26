import os
import re
import json
import hashlib
import requests
import urllib3
import pymupdf as fitz
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Suppress insecure HTTPS request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. Configuration & Client Setup
# ==========================================
PDF_STORE_DIR = "pdf_store"
MANIFEST_FILE = os.path.join(PDF_STORE_DIR, "standards_manifest.json")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "automotive_standards")
ARAI_DOWNLOADS_URL = "https://araiindia.com/downloads"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Seed list to force initial downloading of primary AIS standards
SEED_AIS_CATALOG = [
    {
        "doc_id": "AIS-156",
        "url": "https://araiindia.com/downloads/ais-156-electric-power-train-vehicles-safety"
    },
    {
        "doc_id": "AIS-038-REV2",
        "url": "https://araiindia.com/downloads/ais-038-rev-2-electric-power-train"
    },
    {
        "doc_id": "AIS-037",
        "url": "https://araiindia.com/downloads/ais-037-procedure-for-type-approval"
    }
]

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

os.makedirs(PDF_STORE_DIR, exist_ok=True)

nvidia_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY) if NVIDIA_API_KEY else None
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if (QDRANT_URL and QDRANT_API_KEY) else None

# ==========================================
# 2. Manifest Helpers
# ==========================================
def load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_manifest(manifest: dict):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

def calculate_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

# ==========================================
# 3. Vector Embedding & Qdrant Indexing
# ==========================================
def get_embedding(text: str) -> list[float]:
    response = nvidia_client.embeddings.create(
        input=[text],
        model="nvidia/nemotron-3-embed-1b",
        encoding_format="float",
        extra_body={"input_type": "passage"}
    )
    return response.data[0].embedding

def ensure_qdrant_collection():
    if not qdrant_client:
        return
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"[Qdrant] Creating collection '{COLLECTION_NAME}' (2048-dim)...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE)
        )

def process_and_index_pdf(pdf_path: str, doc_id: str, raw_github_url: str):
    if not (nvidia_client and qdrant_client):
        print("[Warning] Embedding/Qdrant clients not configured. Skipping indexing.")
        return

    doc = fitz.open(pdf_path)
    points = []

    for page_num in range(len(doc)):
        page_text = doc[page_num].get_text("text").strip()
        if not page_text or len(page_text) < 50:
            continue

        vector = get_embedding(page_text)
        point_id = hashlib.md5(f"{doc_id}_p{page_num+1}_{pdf_path}".encode()).hexdigest()

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "standard_family": "AIS",
                    "page_number": page_num + 1,
                    "text": page_text,
                    "github_raw_url": raw_github_url
                }
            )
        )

    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[Qdrant] Successfully indexed {len(points)} page vectors for {doc_id}.")

# ==========================================
# 4. Main Scraper Pipeline
# ==========================================
def run_sync():
    ensure_qdrant_collection()
    manifest = load_manifest()
    downloaded_count = 0

    print("[Sync Engine] Discovering AIS Standard PDFs...")
    candidate_pdfs = []

    # 1. Collect PDFs from Web Scraper
    try:
        res = requests.get(ARAI_DOWNLOADS_URL, headers=HEADERS, verify=False, timeout=20)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"].strip()
                link_text = link.get_text(strip=True)
                
                # Capture all PDF links on download pages
                if ".pdf" in href.lower():
                    pdf_url = urljoin(ARAI_DOWNLOADS_URL, href)
                    
                    # Exclude obvious non-standard junk
                    junk = ["annual", "report", "spandan", "newsletter", "tender", "pension", "balance"]
                    if not any(j in pdf_url.lower() for j in junk):
                        doc_match = re.search(r"(AIS[-\_]?\d+)", f"{pdf_url} {link_text}", re.IGNORECASE)
                        doc_id = doc_match.group(1).upper().replace("_", "-") if doc_match else "AIS-STANDARD"
                        candidate_pdfs.append({"doc_id": doc_id, "url": pdf_url})
    except Exception as e:
        print(f"[Sync Engine Warning] Crawler encountered issue: {e}")

    # 2. Merge Seed Catalog for Guaranteed First Run
    for seed in SEED_AIS_CATALOG:
        if seed["url"] not in [c["url"] for c in candidate_pdfs]:
            candidate_pdfs.append(seed)

    print(f"[Sync Engine] Processing {len(candidate_pdfs)} candidate AIS PDFs...")

    # 3. Download, Hash Verification, and Vector Indexing
    for item in candidate_pdfs:
        pdf_url = item["url"]
        doc_id = item["doc_id"]

        filename = os.path.basename(pdf_url.split("?")[0])
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        filename = re.sub(r"[^\w\-.]", "_", filename)

        local_filepath = os.path.join(PDF_STORE_DIR, filename)

        try:
            print(f"[Downloading] {doc_id} -> {pdf_url}")
            pdf_res = requests.get(pdf_url, headers=HEADERS, verify=False, timeout=30)

            if pdf_res.status_code == 200 and len(pdf_res.content) > 1000:
                with open(local_filepath, "wb") as f:
                    f.write(pdf_res.content)

                file_hash = calculate_sha256(local_filepath)

                # Skip vector re-indexing if hash hasn't changed
                if manifest.get(filename) == file_hash and os.path.exists(local_filepath):
                    print(f"[Skip] {filename} is already indexed and unchanged.")
                    continue

                github_user = os.getenv("GITHUB_REPOSITORY", "Ibraheem9090/Automotive-Standard-AI")
                raw_github_url = f"https://raw.githubusercontent.com/{github_user}/main/{PDF_STORE_DIR}/{filename}"

                # Extract and push vectors to Qdrant Cloud
                process_and_index_pdf(local_filepath, doc_id, raw_github_url)

                manifest[filename] = file_hash
                save_manifest(manifest)
                downloaded_count += 1
            else:
                print(f"[Warning] Endpoint returned status {pdf_res.status_code} for {pdf_url}")

        except Exception as e:
            print(f"[Error] Failed processing {pdf_url}: {e}")

    print(f"\n[Sync Complete] Finished downloading and indexing {downloaded_count} AIS Standard PDF(s).")

if __name__ == "__main__":
    run_sync()
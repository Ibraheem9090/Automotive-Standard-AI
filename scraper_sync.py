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
# 1. Configuration & Setup
# ==========================================
PDF_STORE_DIR = "pdf_store"
MANIFEST_FILE = os.path.join(PDF_STORE_DIR, "standards_manifest.json")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "automotive_standards")
ARAI_DOWNLOADS_URL = "https://araiindia.com/downloads"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

os.makedirs(PDF_STORE_DIR, exist_ok=True)

nvidia_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY) if NVIDIA_API_KEY else None
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if (QDRANT_URL and QDRANT_API_KEY) else None

# ==========================================
# 2. Helpers
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

def get_embedding(text: str) -> list[float]:
    response = nvidia_client.embeddings.create(
        input=[text],
        model="nvidia/nemotron-3-embed-1b",
        encoding_format="float",
        extra_body={"input_type": "passage"}
    )
    return response.data[0].embedding

def process_and_index_pdf(pdf_path: str, doc_id: str, raw_github_url: str):
    if not (nvidia_client and qdrant_client):
        print("[Warning] API clients not fully initialized. Skipping indexing.")
        return

    doc = fitz.open(pdf_path)
    points = []

    for page_num in range(len(doc)):
        page_text = doc[page_num].get_text("text").strip()
        if not page_text or len(page_text) < 50:
            continue

        vector = get_embedding(page_text)
        point_id = hashlib.md5(f"{doc_id}_p{page_num+1}_{os.path.basename(pdf_path)}".encode()).hexdigest()

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
        print(f"[Qdrant] Indexed {len(points)} page vectors for {doc_id}.")

# ==========================================
# 3. Main Pipeline
# ==========================================
def run_sync():
    ensure_qdrant_collection()
    manifest = load_manifest()
    processed_count = 0

    github_user = os.getenv("GITHUB_REPOSITORY", "Ibraheem9090/Automotive-Standard-AI")

    # PHASE 1: Index existing local PDFs in pdf_store/
    print("[Sync Engine] Phase 1: Checking local pdf_store/ for files...")
    local_files = [f for f in os.listdir(PDF_STORE_DIR) if f.lower().endswith(".pdf")]
    
    for filename in local_files:
        local_filepath = os.path.join(PDF_STORE_DIR, filename)
        file_hash = calculate_sha256(local_filepath)

        if manifest.get(filename) == file_hash:
            continue

        doc_match = re.search(r"(AIS[-\_]?\d+)", filename, re.IGNORECASE)
        doc_id = doc_match.group(1).upper().replace("_", "-") if doc_match else filename.replace(".pdf", "")

        raw_github_url = f"https://raw.githubusercontent.com/{github_user}/main/{PDF_STORE_DIR}/{filename}"
        
        print(f"[Local PDF] Processing & Indexing: {filename} ({doc_id})")
        process_and_index_pdf(local_filepath, doc_id, raw_github_url)

        manifest[filename] = file_hash
        processed_count += 1

    # PHASE 2: Scrape web catalog for new standards
    print(f"[Sync Engine] Phase 2: Fetching web catalog from {ARAI_DOWNLOADS_URL}...")
    try:
        res = requests.get(ARAI_DOWNLOADS_URL, headers=HEADERS, verify=False, timeout=20)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            rows = soup.find_all("tr")

            for row in rows:
                row_text = row.get_text(" ", strip=True)
                
                # Exclude administrative files
                if any(j in row_text.lower() for j in ["annual report", "poster", "defaulter", "5yrplan"]):
                    continue

                if "ais" in row_text.lower():
                    link = row.find("a", href=True)
                    if not link:
                        continue
                    
                    href = link["href"].strip()
                    pdf_url = urljoin(ARAI_DOWNLOADS_URL, href)

                    doc_match = re.search(r"(AIS[-\_]?\d+)", row_text, re.IGNORECASE)
                    doc_id = doc_match.group(1).upper().replace("_", "-") if doc_match else "AIS-STANDARD"

                    filename = f"{doc_id}.pdf"
                    local_filepath = os.path.join(PDF_STORE_DIR, filename)

                    print(f"[Downloading Web PDF] {doc_id} -> {pdf_url}")
                    pdf_res = requests.get(pdf_url, headers=HEADERS, verify=False, timeout=30)

                    if pdf_res.status_code == 200 and len(pdf_res.content) > 1000:
                        with open(local_filepath, "wb") as f:
                            f.write(pdf_res.content)

                        file_hash = calculate_sha256(local_filepath)
                        raw_github_url = f"https://raw.githubusercontent.com/{github_user}/main/{PDF_STORE_DIR}/{filename}"

                        if manifest.get(filename) != file_hash:
                            process_and_index_pdf(local_filepath, doc_id, raw_github_url)
                            manifest[filename] = file_hash
                            processed_count += 1
    except Exception as e:
        print(f"[Sync Engine Warning] Crawler run encountered error: {e}")

    save_manifest(manifest)
    print(f"\n[Sync Complete] Successfully processed and indexed {processed_count} AIS Standard PDF(s).")

if __name__ == "__main__":
    run_sync()
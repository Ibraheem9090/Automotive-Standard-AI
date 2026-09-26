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

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

os.makedirs(PDF_STORE_DIR, exist_ok=True)

# Initialize API clients if available
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
# 3. AIS Standard Document Filter
# ==========================================
def is_valid_ais_pdf(url: str, link_text: str) -> bool:
    """Strictly evaluates if a link belongs to an AIS automotive standard document."""
    combined = f"{url.lower()} {link_text.lower()}"

    # Must be a PDF link
    if ".pdf" not in combined:
        return False

    # Block non-standard junk documents
    junk_keywords = [
        "annual", "report", "spandan", "newsletter", "tender", 
        "career", "privacy", "policy", "form", "balance", 
        "financial", "audited", "pension", "roster"
    ]
    if any(junk in combined for junk in junk_keywords):
        return False

    # Accept if explicit AIS standard patterns exist
    ais_patterns = [
        r"ais[-\_\s]?\d+",       # AIS-156, AIS_038, AIS 037, etc.
        r"draft[-\_\s]?ais",      # Draft AIS
        r"amendment",            # Standard amendments
        r"tac"                   # Type Approval
    ]
    
    return any(re.search(pattern, combined) for pattern in ais_patterns)

# ==========================================
# 4. Vector Embedding & Qdrant Indexing
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
        print(f"[Qdrant] Indexed {len(points)} page vectors for {doc_id}.")

# ==========================================
# 5. Main Scraper Pipeline
# ==========================================
def run_sync():
    print(f"[Sync Engine] Fetching catalog from: {ARAI_DOWNLOADS_URL}")
    ensure_qdrant_collection()
    manifest = load_manifest()

    try:
        res = requests.get(ARAI_DOWNLOADS_URL, headers=HEADERS, verify=False, timeout=20)
        if res.status_code != 200:
            print(f"[Sync Engine Error] Failed to reach {ARAI_DOWNLOADS_URL} (Status: {res.status_code})")
            return

        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.find_all("a", href=True)
        print(f"[Sync Engine] Scanning {len(links)} total links for AIS standard PDFs...")

        downloaded_count = 0

        for link in links:
            href = link["href"].strip()
            link_text = link.get_text(strip=True)

            if is_valid_ais_pdf(href, link_text):
                pdf_url = urljoin(ARAI_DOWNLOADS_URL, href)
                
                # Extract clean AIS Doc ID (e.g., AIS-156)
                doc_match = re.search(r"(AIS[-\_]?\d+(?:\s?\(Part\s?\d+\))?)", f"{pdf_url} {link_text}", re.IGNORECASE)
                doc_id = doc_match.group(1).upper().replace("_", "-") if doc_match else "AIS-STANDARD"

                # Standardize filename
                filename = os.path.basename(pdf_url.split("?")[0])
                if not filename.endswith(".pdf"):
                    filename += ".pdf"
                filename = re.sub(r"[^\w\-.]", "_", filename)

                local_filepath = os.path.join(PDF_STORE_DIR, filename)

                if os.path.exists(local_filepath):
                    print(f"[Skip] {filename} already exists locally.")
                    continue

                print(f"[Downloading AIS Standard] {doc_id} -> {pdf_url}")
                pdf_res = requests.get(pdf_url, headers=HEADERS, verify=False, timeout=30)
                
                if pdf_res.status_code == 200 and len(pdf_res.content) > 1000:
                    with open(local_filepath, "wb") as f:
                        f.write(pdf_res.content)

                    file_hash = calculate_sha256(local_filepath)
                    
                    github_user = os.getenv("GITHUB_REPOSITORY", "Ibraheem9090/Automotive-Standard-AI")
                    raw_github_url = f"https://raw.githubusercontent.com/{github_user}/main/{PDF_STORE_DIR}/{filename}"

                    process_and_index_pdf(local_filepath, doc_id, raw_github_url)

                    manifest[filename] = file_hash
                    save_manifest(manifest)
                    downloaded_count += 1
                else:
                    print(f"[Warning] Failed to download valid content from {pdf_url}")

        print(f"[Sync Complete] Successfully downloaded and indexed {downloaded_count} new AIS Standard PDF(s).")

    except Exception as e:
        print(f"[Sync Engine Error] {e}")

if __name__ == "__main__":
    run_sync()
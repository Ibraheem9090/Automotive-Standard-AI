import os
import re
import json
import hashlib
import requests
import fitz  # PyMuPDF
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ==========================================
# 1. Configuration & Client Setup
# ==========================================
PDF_STORE_DIR = "pdf_store"
MANIFEST_FILE = os.path.join(PDF_STORE_DIR, "standards_manifest.json")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "automotive_standards")

# Target ARAI AIS Downloads Endpoint
AIS_BASE_URL = "https://www.araiindia.com/downloads/ais-downloads"

# Initialize Secrets / Environment Variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not NVIDIA_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    print("[Error] Missing required environment variables (NVIDIA_API_KEY, QDRANT_URL, QDRANT_API_KEY).")
    exit(1)

nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Ensure pdf_store directory exists
os.makedirs(PDF_STORE_DIR, exist_ok=True)

# ==========================================
# 2. Manifest Management (SHA-256 Tracking)
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
# 3. Targeted ARAI AIS Web Scraper
# ==========================================
def fetch_ais_pdf_links() -> list[dict]:
    """Crawls ARAI AIS Download pages specifically for valid AIS standard PDFs."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    pdf_entries = []
    visited_urls = set()

    # Crawl main AIS downloads page and up to 10 paginated pages if present
    pages_to_crawl = [AIS_BASE_URL]
    for page_idx in range(1, 10):
        pages_to_crawl.append(f"{AIS_BASE_URL}?page={page_idx}")

    print(f"[AIS Scraper] Initiating crawl on target: {AIS_BASE_URL}")

    for url in pages_to_crawl:
        if url in visited_urls:
            continue
        visited_urls.add(url)

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.content, "html.parser")
            links = soup.find_all("a", href=True)

            for link in links:
                href = link["href"].strip()
                link_text = link.get_text(strip=True)

                if href.lower().endswith(".pdf"):
                    full_pdf_url = urljoin(url, href)

                    # Strict AIS Validation Rule
                    combined_str = f"{full_pdf_url.lower()} {link_text.lower()}"
                    
                    # Exclude non-standard documents
                    if any(junk in combined_str for junk in ["annual", "report", "spandan", "newsletter", "tender", "career"]):
                        continue

                    # Require explicit AIS tag or ARAI downloads origin
                    if re.search(r"ais[-\_]?\d+", combined_str) or "/downloads/" in full_pdf_url.lower():
                        
                        # Extract clean Document ID (e.g. AIS-156)
                        doc_match = re.search(r"(AIS[-\_]?\d+(?:\s?\(Part\s?\d+\))?)", combined_str, re.IGNORECASE)
                        doc_id = doc_match.group(1).upper().replace("_", "-") if doc_match else "AIS-STANDARD"

                        pdf_entries.append({
                            "url": full_pdf_url,
                            "doc_id": doc_id,
                            "title": link_text or doc_id
                        })

        except Exception as e:
            print(f"[AIS Scraper] Warning reading {url}: {e}")

    # Deduplicate by URL
    unique_pdfs = {item["url"]: item for item in pdf_entries}.values()
    print(f"[AIS Scraper] Found {len(unique_pdfs)} unique AIS PDF standard links.")
    return list(unique_pdfs)

# ==========================================
# 4. Vector Embedding & Qdrant Ingestion
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
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"[Qdrant] Creating collection '{COLLECTION_NAME}' (2048-dim)...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE)
        )

def process_and_index_pdf(pdf_path: str, doc_id: str, raw_github_url: str):
    """Extracts text page-by-page and indexes into Qdrant Cloud."""
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
# 5. Pipeline Execution
# ==========================================
def run_sync():
    ensure_qdrant_collection()
    manifest = load_manifest()
    ais_links = fetch_ais_pdf_links()

    new_download_count = 0

    for item in ais_links:
        pdf_url = item["url"]
        doc_id = item["doc_id"]
        
        # Clean filename formatting
        filename = re.sub(r"[^\w\-.]", "_", os.path.basename(pdf_url))
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        local_filepath = os.path.join(PDF_STORE_DIR, filename)

        try:
            # Download PDF
            print(f"[Downloading] {doc_id} from {pdf_url}...")
            resp = requests.get(pdf_url, timeout=30)
            if resp.status_code != 200:
                continue

            with open(local_filepath, "wb") as f:
                f.write(resp.content)

            # Compute SHA-256 Delta Hash
            file_hash = calculate_sha256(local_filepath)

            if manifest.get(filename) == file_hash:
                print(f"[Skip] {filename} is unchanged (SHA-256 match).")
                continue

            # Compute raw GitHub URL for visual rendering in app.py
            github_user = os.getenv("GITHUB_REPOSITORY", "Ibraheem9090/Automotive-Standard-AI")
            raw_github_url = f"https://raw.githubusercontent.com/{github_user}/main/{PDF_STORE_DIR}/{filename}"

            # Extract & Index into Qdrant Cloud
            process_and_index_pdf(local_filepath, doc_id, raw_github_url)

            # Update manifest entry
            manifest[filename] = file_hash
            new_download_count += 1

        except Exception as e:
            print(f"[Error] Failed processing {pdf_url}: {e}")

    save_manifest(manifest)
    print(f"\n[Sync Complete] Finished indexing {new_download_count} new/updated ARAI AIS Standard(s).")

if __name__ == "__main__":
    run_sync()
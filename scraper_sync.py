import os
import re
import json
import hashlib
import docx  # python-docx for .docx files
import pymupdf as fitz  # PyMuPDF for .pdf files
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

import subprocess
import os

def convert_doc_to_pdf(doc_path: str):
    """Converts .doc / .docx files to .pdf using headless LibreOffice."""
    output_dir = os.path.dirname(doc_path)
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", doc_path, "--outdir", output_dir],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"[Conversion Success] Converted {os.path.basename(doc_path)} to PDF.")
    except Exception as e:
        print(f"[Conversion Error] Could not convert {doc_path} to PDF: {e}")

PDF_STORE_DIR = "pdf_store"
MANIFEST_FILE = os.path.join(PDF_STORE_DIR, "standards_manifest.json")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "automotive_standards")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

os.makedirs(PDF_STORE_DIR, exist_ok=True)

nvidia_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY) if NVIDIA_API_KEY else None
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if (QDRANT_URL and QDRANT_API_KEY) else None

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
    
    for field in ["standard_family", "doc_id"]:
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

def get_embedding(text: str) -> list[float]:
    response = nvidia_client.embeddings.create(
        input=[text],
        model="nvidia/nemotron-3-embed-1b",
        encoding_format="float",
        extra_body={"input_type": "passage"}
    )
    return response.data[0].embedding

def extract_chunks_from_file(file_path: str) -> list[tuple[int, str]]:
    """Extracts text chunks from PDF, DOCX, and legacy DOC files."""
    ext = os.path.splitext(file_path)[1].lower()
    chunks = []

    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text").strip()
            if page_text and len(page_text) >= 50:
                chunks.append((page_num + 1, page_text))

    elif ext == ".docx":
        doc = docx.Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 30]
        # Group paragraphs into 5-paragraph chunks to mimic pages
        chunk_size = 5
        for i in range(0, len(paragraphs), chunk_size):
            chunk_text = "\n".join(paragraphs[i:i + chunk_size])
            chunks.append((i // chunk_size + 1, chunk_text))

    elif ext == ".doc":
        # Attempt docx parsing first (some .doc files are OpenXML)
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 30]
            chunk_size = 5
            for i in range(0, len(paragraphs), chunk_size):
                chunk_text = "\n".join(paragraphs[i:i + chunk_size])
                chunks.append((i // chunk_size + 1, chunk_text))
        except Exception:
            # Fallback binary string extraction for legacy Word streams
            with open(file_path, "rb") as f:
                content = f.read()
            raw_strings = re.findall(rb'[a-zA-Z0-9\s\.,;:()\-\n]{30,}', content)
            clean_text = "\n".join([s.decode('utf-8', errors='ignore') for s in raw_strings])
            if clean_text:
                chunks.append((1, clean_text))

    return chunks

def process_and_index_file(filepath: str, doc_id: str, raw_github_url: str):
    if not (nvidia_client and qdrant_client):
        return

    chunks = extract_chunks_from_file(filepath)
    points = []

    for chunk_num, text_content in chunks:
        vector = get_embedding(text_content)
        point_id = hashlib.md5(f"{doc_id}_c{chunk_num}_{os.path.basename(filepath)}".encode()).hexdigest()

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "standard_family": "AIS",
                    "page_number": chunk_num,
                    "text": text_content,
                    "github_raw_url": raw_github_url
                }
            )
        )

    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[Qdrant] Indexed {len(points)} text vectors for {doc_id}.")

def run_sync():
    ensure_qdrant_collection()
    manifest = load_manifest()
    processed_count = 0

    github_user = os.getenv("GITHUB_REPOSITORY", "Ibraheem9090/Automotive-Standard-AI")

    print("[Sync Engine] Scanning local pdf_store/ for files (.pdf, .docx, .doc)...")
    valid_extensions = (".pdf", ".docx", ".doc")
    local_files = [f for f in os.listdir(PDF_STORE_DIR) if f.lower().endswith(valid_extensions)]

    for filename in local_files:
        local_filepath = os.path.join(PDF_STORE_DIR, filename)
        file_hash = calculate_sha256(local_filepath)

        if manifest.get(filename) == file_hash:
            continue

        doc_match = re.search(r"(AIS[-\_]?\d+)", filename, re.IGNORECASE)
        doc_id = doc_match.group(1).upper().replace("_", "-") if doc_match else os.path.splitext(filename)[0]

        raw_github_url = f"https://raw.githubusercontent.com/{github_user}/main/{PDF_STORE_DIR}/{filename}"

        print(f"[Processing File] {filename} ({doc_id})")
        process_and_index_file(local_filepath, doc_id, raw_github_url)

        manifest[filename] = file_hash
        processed_count += 1

    save_manifest(manifest)
    print(f"\n[Sync Complete] Successfully processed and indexed {processed_count} Standard file(s).")

if __name__ == "__main__":
    run_sync()
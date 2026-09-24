import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from src.ingestion import PDFIngestionPipeline

MANIFEST_FILE = "pdf_store/standards_manifest.json"
PDF_DIR = "pdf_store"

# Ensure storage directory exists
os.makedirs(PDF_DIR, exist_ok=True)

def compute_sha256(content: bytes) -> str:
    """Calculates SHA-256 hash of raw file bytes."""
    return hashlib.sha256(content).hexdigest()

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest: dict):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

def run_sync_pipeline():
    manifest = load_manifest()
    pipeline = PDFIngestionPipeline()
    github_user_repo = os.getenv("GITHUB_REPO", "Ibraheem/automotive-standards-rag")

    # Example: Monitored target standards (expand with specific site scrapers)
    target_pdfs = [
        {
            "doc_id": "AIS-156-Rev2",
            "url": "https://araiindia.com/downloads/ais-156.pdf",
            "source_site": "ARAI"
        },
        {
            "doc_id": "UN-R155",
            "url": "https://unece.org/transport/documents/un-r155.pdf",
            "source_site": "UNECE"
        }
    ]

    print("🚀 Starting daily differential sync...")

    for item in target_pdfs:
        doc_id = item["doc_id"]
        url = item["url"]

        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                print(f"⚠️ Failed to fetch {doc_id} from {url}")
                continue

            pdf_bytes = response.content
            current_hash = compute_sha256(pdf_bytes)

            # Check if file has changed
            previous_hash = manifest.get(doc_id, {}).get("sha256")

            if previous_hash == current_hash:
                print(f"✅ [{doc_id}] No revision changes detected (Hash: {current_hash[:8]}...). Skipping.")
                continue

            print(f"🔄 [{doc_id}] New revision or document detected! Indexing...")

            # Save locally for GitHub commit
            pdf_path = os.path.join(PDF_DIR, f"{doc_id}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            github_raw_url = f"https://raw.githubusercontent.com/{github_user_repo}/main/pdf_store/{doc_id}.pdf"

            # Re-index into Qdrant Cloud
            pipeline.process_and_index_pdf(
                pdf_path=pdf_path,
                doc_id=doc_id,
                sha256_hash=current_hash,
                github_raw_url=github_raw_url
            )

            # Update Manifest
            manifest[doc_id] = {
                "sha256": current_hash,
                "url": url,
                "source_site": item["source_site"],
                "last_updated": requests.utils.default_user_agent()
            }

        except Exception as e:
            print(f"❌ Error processing {doc_id}: {e}")

    save_manifest(manifest)
    print("✨ Sync cycle completed successfully.")

if __name__ == "__main__":
    run_sync_pipeline()
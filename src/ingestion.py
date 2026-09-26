import os
import uuid
import fitz  # PyMuPDF
from typing import Dict, Any, List
from src.nvidia_client import NVIDIAClient
from src.qdrant_manager import QdrantManager

class IngestionPipeline:
    def __init__(
        self, 
        nvidia_client: NVIDIAClient | None = None, 
        qdrant_mgr: QdrantManager | None = None
    ):
        self.nvidia_client = nvidia_client or NVIDIAClient()
        self.qdrant_mgr = qdrant_mgr or QdrantManager()

    def extract_text_by_page(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from PDF using PyMuPDF."""
        pages_data = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text
                })
        doc.close()
        return pages_data

    def process_and_index(self, pdf_path: str, metadata: Dict[str, Any]):
        """Parses PDF, generates 2048-dim embeddings, and upserts points to Qdrant."""
        if not os.path.exists(pdf_path):
            print(f"[IngestionPipeline] File not found: {pdf_path}")
            return

        doc_id = metadata.get("doc_id", os.path.basename(pdf_path).replace(".pdf", ""))
        print(f"[IngestionPipeline] Processing '{doc_id}'...")

        pages = self.extract_text_by_page(pdf_path)
        if not pages:
            print(f"[IngestionPipeline] No extractable text in {pdf_path}")
            return

        points_data = []
        for p in pages:
            chunk_text = p["text"]
            embedding = self.nvidia_client.get_embedding(chunk_text)
            
            if not embedding:
                continue

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_p{p['page_number']}"))
            
            payload = {
                "doc_id": doc_id,
                "page_number": p["page_number"],
                "text": chunk_text,
                "standard_family": metadata.get("standard_family", "AIS"),
                "domain": metadata.get("domain", "Automotive Technical Regulation"),
                "source_site": metadata.get("source_site", "ARAI"),
                "url": metadata.get("url", "")
            }

            points_data.append({
                "id": point_id,
                "vector": embedding,
                "payload": payload
            })

        if points_data:
            self.qdrant_mgr.upsert_points(points_data)
            print(f"[IngestionPipeline] Indexed {len(points_data)} pages for '{doc_id}'.")
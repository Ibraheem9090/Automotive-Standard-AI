import os
import fitz  # PyMuPDF
from src.nvidia_client import NVIDIAClient
from src.qdrant_manager import QdrantManager

class PDFIngestionPipeline:
    def __init__(self):
        self.nvidia_client = NVIDIAClient()
        self.qdrant_manager = QdrantManager()

    def process_and_index_pdf(self, pdf_path: str, doc_id: str, source_site: str, url: str) -> bool:
        """
        Parses a PDF file, extracts page text and rendering metadata,
        generates embeddings via NVIDIA NIM API, and indexes vector points into Qdrant Cloud.
        """
        if not os.path.exists(pdf_path):
            print(f"[Ingestion Error] File not found: {pdf_path}")
            return False

        doc = fitz.open(pdf_path)
        print(f"[{doc_id}] Indexing {len(doc)} pages into Qdrant...")

        points = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if not text:
                continue

            # Generate vector embedding using NVIDIA NIM API
            embedding = self.nvidia_client.get_embedding(text)
            if not embedding:
                continue

            payload = {
                "doc_id": doc_id,
                "page_number": page_num + 1,
                "text": text[:1000],  # Stored text chunk preview
                "source_site": source_site,
                "url": url,
                "pdf_path": pdf_path
            }

            points.append({
                "page_number": page_num + 1,
                "vector": embedding,
                "payload": payload
            })

        if points:
            self.qdrant_manager.upsert_document_points(doc_id=doc_id, points=points)
            print(f"[{doc_id}] Successfully indexed {len(points)} vector chunks.")
            return True

        return False
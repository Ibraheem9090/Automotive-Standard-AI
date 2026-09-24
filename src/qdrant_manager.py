import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantManager:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, collection_name: Optional[str] = None):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "automotive_standards")

        if not self.url or not self.api_key:
            raise ValueError("[QdrantManager Error] QDRANT_URL or QDRANT_API_KEY environment variable missing.")

        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self, vector_size: int = 4096):
        """
        Ensures the target Qdrant collection exists (4096 dimensions for nvidia/nv-embed-v1).
        """
        try:
            collections = [col.name for col in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print(f"[QdrantManager] Created collection '{self.collection_name}'.")
        except Exception as e:
            print(f"[QdrantManager Error] Collection check/creation failed: {e}")

    def upsert_document_points(self, doc_id: str, points: List[Dict[str, Any]]):
        """
        Converts chunk payload points to PointStruct objects and upserts into Qdrant Cloud.
        """
        qdrant_points = []
        for item in points:
            # Create a deterministic UUID for each page chunk point
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_page_{item['page_number']}"))
            qdrant_points.append(
                PointStruct(
                    id=point_id,
                    vector=item["vector"],
                    payload=item["payload"]
                )
            )

        if qdrant_points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points
            )
            print(f"[QdrantManager] Upserted {len(qdrant_points)} vector points for doc '{doc_id}'.")
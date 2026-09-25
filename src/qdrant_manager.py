import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantManager:
    def __init__(
        self, 
        url: Optional[str] = None, 
        api_key: Optional[str] = None, 
        collection_name: Optional[str] = None,
        vector_size: int = 2048  # Updated default to 2048 for nvidia/nemotron-3-embed-1b
    ):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "automotive_standards")
        self.vector_size = vector_size

        if not self.url or not self.api_key:
            raise ValueError("[QdrantManager Error] QDRANT_URL or QDRANT_API_KEY environment variable missing.")

        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self._ensure_collection_exists(vector_size=self.vector_size)

    def _ensure_collection_exists(self, vector_size: int = 2048):
        """
        Ensures the target Qdrant collection exists (2048 dimensions for active NVIDIA NIM embedding models).
        """
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print(f"[QdrantManager] Created collection '{self.collection_name}' with size {vector_size}.")
        except Exception as e:
            print(f"[QdrantManager Error] Collection check/creation failed: {e}")

    def recreate_collection(self, vector_size: int = 2048):
        """
        Deletes and recreates the collection with updated vector dimensions (e.g. switching 4096 -> 2048).
        """
        try:
            if self.client.collection_exists(collection_name=self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)
                print(f"[QdrantManager] Deleted existing collection '{self.collection_name}'.")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"[QdrantManager] Recreated collection '{self.collection_name}' with size {vector_size}.")
        except Exception as e:
            print(f"[QdrantManager Error] Collection recreation failed: {e}")

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

    def search(self, query_vector: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches Qdrant Cloud using modern query_points syntax (qdrant-client >= 1.11.0).
        """
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,  # Uses 'query=' parameter required by current qdrant-client
                limit=top_k
            )
            return [point.payload for point in response.points if point.payload is not None]
        except Exception as e:
            print(f"[QdrantManager Error] Search failed: {e}")
            return []
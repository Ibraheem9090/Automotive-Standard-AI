import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)

class QdrantManager:
    def __init__(
        self, 
        url: Optional[str] = None, 
        api_key: Optional[str] = None, 
        collection_name: str = "automotive_standards"
    ):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        
        # Initialize Qdrant Client
        if self.url and self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            print("[QdrantManager] Warning: QDRANT_URL or QDRANT_API_KEY missing. Using in-memory client.")
            self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self):
        """Creates collection for 2048-dim vectors and registers payload indexes for fast filtering."""
        collections = [col.name for col in self.client.get_collections().collections]
        
        if self.collection_name not in collections:
            print(f"[QdrantManager] Creating collection '{self.collection_name}' (2048-dim, Cosine)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=2048, distance=Distance.COSINE)
            )

        # Ensure payload indexes exist for metadata filtering
        indexed_fields = ["standard_family", "process_id", "doc_id", "domain"]
        for field in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception:
                # Index likely already exists
                pass

    def upsert_points(self, points_data: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upserts vector points in batches to handle large multi-page PDF indexing.
        Accepts dicts formatted with 'id', 'vector', and 'payload'.
        """
        points = []
        for item in points_data:
            points.append(
                PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload=item["payload"]
                )
            )

        # Batch upload to avoid request size limits in Qdrant Cloud
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
        print(f"[QdrantManager] Successfully upserted {len(points)} points into '{self.collection_name}'.")

    def upsert_document_points(self, doc_id: str, points: List[Dict[str, Any]]):
        """Alias method to maintain backward compatibility with ingestion pipeline calls."""
        self.upsert_points(points_data=points)

    def query_standards(
        self, 
        query_vector: List[float], 
        standard_family: Optional[str] = None, 
        process_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes vector similarity search with optional payload filters.
        Supports standard family (AIS, UN_ECE, ASPICE), process ID (SYS.2), or specific doc ID.
        """
        must_conditions = []
        
        if standard_family:
            must_conditions.append(
                FieldCondition(key="standard_family", match=MatchValue(value=standard_family))
            )
        if process_id:
            must_conditions.append(
                FieldCondition(key="process_id", match=MatchValue(value=process_id))
            )
        if doc_id:
            must_conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Execute query points request
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )

        results = []
        for point in search_result.points:
            results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload
            })

        return results
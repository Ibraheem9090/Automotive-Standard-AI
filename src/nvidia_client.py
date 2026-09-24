import os
import requests
from typing import List, Optional

class NVIDIAClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.embedding_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-embed-v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Fetches text vector embeddings using NVIDIA NIM API (nvidia/nv-embed-v1).
        """
        if not self.api_key:
            print("[NVIDIAClient Error] NVIDIA_API_KEY environment variable is missing.")
            return None

        # Truncate text input safely to fit model context
        payload = {
            "input": [text[:2000]],
            "model": "nvidia/nv-embed-v1",
            "input_type": "passage"
        }

        try:
            response = requests.post(self.embedding_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            print(f"[NVIDIAClient Error] Failed to generate embedding: {e}")
            return None
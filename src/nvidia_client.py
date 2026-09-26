import os
import requests
from typing import List, Optional

class NvidiaClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.embed_url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.llm_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    def get_embedding(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": [text],
            "model": "nvidia/nemotron-3-embed-1b",  # 2048 dimensions
            "input_type": "passage"
        }
        try:
            res = requests.post(self.embed_url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                return res.json()["data"][0]["embedding"]
            else:
                print(f"[NvidiaClient Error] {res.status_code}: {res.text}")
                return []
        except Exception as e:
            print(f"[NvidiaClient Exception] {e}")
            return []
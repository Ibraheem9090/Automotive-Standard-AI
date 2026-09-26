import os
from openai import OpenAI

class NVIDIAClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable or argument is missing.")
        
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )

    def get_embedding(self, text: str) -> list[float]:
        """Generates embeddings using NVIDIA NIM API."""
        response = self.client.embeddings.create(
            input=[text],
            model="nvidia/nemotron-3-embed-1b",
            encoding_format="float",
            extra_body={"input_type": "passage"}
        )
        return response.data[0].embedding
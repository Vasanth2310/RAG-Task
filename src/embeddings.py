from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings)

# src/retrievers/dense_retriever.py
import numpy as np
from sentence_transformers import SentenceTransformer
from src.retrievers.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    def __init__(self, chunks: list[str], model_name="all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(chunks, convert_to_numpy=True)

    def retrieve(self, query: str, top_n: int = 5):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        scores = np.dot(self.embeddings, query_emb.T).squeeze()
        top_idx = scores.argsort()[-top_n:][::-1]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

# src/retrievers/combined.py
import numpy as np
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .base import BaseRetriever


class CombinedRetriever(BaseRetriever):
    def __init__(self, chunks: list[str], alpha=0.5, beta=0.5):
        self.chunks = chunks
        self.alpha = alpha
        self.beta = beta

        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks)

    def retrieve(self, query: str, top_n: int = 5):
        # 1. BM25 scores для ВСІХ чанків
        bm25_results = self.bm25.retrieve(query, top_n=len(self.chunks))
        bm25_scores = np.zeros(len(self.chunks))

        for text, score in bm25_results:
            idx = self.chunks.index(text)
            bm25_scores[idx] = score

        # 2. Dense scores для ВСІХ чанків
        dense_results = self.dense.retrieve(query, top_n=len(self.chunks))
        dense_scores = np.zeros(len(self.chunks))

        for text, score in dense_results:
            idx = self.chunks.index(text)
            dense_scores[idx] = score

        # 3. Комбінуємо
        combined_scores = self.alpha * bm25_scores + self.beta * dense_scores

        # 4. Top-n
        top_idx = combined_scores.argsort()[-top_n:][::-1]

        return [(self.chunks[i], float(combined_scores[i])) for i in top_idx]

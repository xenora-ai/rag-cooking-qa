# src/retrievers/bm25_retriever.py
from rank_bm25 import BM25Okapi
from src.retrievers.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[str]):
        tokenized = [c.split() for c in chunks]
        self.chunks = chunks
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_n: int = 5):
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[-top_n:][::-1]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

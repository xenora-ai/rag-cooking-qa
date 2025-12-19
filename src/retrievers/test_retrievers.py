import os

from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.combined import CombinedRetriever


def load_chunks(chunks_dir: str):
    chunks = []
    filenames = []

    for fname in sorted(os.listdir(chunks_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(chunks_dir, fname), "r", encoding="utf-8") as f:
                chunks.append(f.read())
                filenames.append(fname)

    return chunks, filenames


if __name__ == "__main__":
    chunks_dir = "../../data/chunks"  # перевір шлях
    chunks, filenames = load_chunks(chunks_dir)

    query = "Смачний та червоний борщ"

    retrievers = {
        "BM25": BM25Retriever(chunks),
        "Dense": DenseRetriever(chunks),
        "Combined": CombinedRetriever(chunks),
    }

    for name, retriever in retrievers.items():
        print(f"\n===== {name} Retriever =====")
        results = retriever.retrieve(query, top_n=3)

        for i, (text, score) in enumerate(results, 1):
            chunk_idx = chunks.index(text)
            print(f"[{i}] {filenames[chunk_idx]} | score={score:.4f}")
            print(text[:200].replace("\n", " "), "...\n")

# src/pipeline/rag_pipeline.py
import os
import csv

from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.combined import CombinedRetriever

from src.llm.llm_client import LLMClient

from src.reranker.reranker import SBERReranker


class RAGPipeline:
    def __init__(self, chunks_dir: str, doc_metadata_dir: str, groq_api_key=None, use_reranker=True):
        self.chunks_dir = chunks_dir
        self.chunks = self._load_chunks()

        self.doc_metadata = self._load_metadata(doc_metadata_dir)

        self.llm = LLMClient(api_key=groq_api_key)

        self.use_reranker = use_reranker
        if self.use_reranker:
            self.reranker = SBERReranker()

    def _load_chunks(self):
        texts = []
        files = []

        for filename in sorted(os.listdir(self.chunks_dir)):  # sorted для стабільності
            if filename.endswith(".txt"):
                with open(os.path.join(self.chunks_dir, filename), "r", encoding="utf-8") as f:
                    files.append(filename)
                    texts.append(f.read())
        self.chunk_files = files
        return texts

    @staticmethod
    def _load_metadata(csv_path: str):
        metadata = {}

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                doc_id = row["id"].strip()
                metadata[doc_id] = {
                    "url": row.get("url", "").strip(),
                    "category": row.get("category", "").strip()
                }

        return metadata

    def _get_retriever(self, retriever_name: str):
        retriever_name = retriever_name.lower()
        if retriever_name == "bm25":
            return BM25Retriever(self.chunks)
        elif retriever_name == "dense":
            return DenseRetriever(self.chunks)
        elif retriever_name == "combined":
            return CombinedRetriever(self.chunks)
        else:
            raise ValueError("Unknown retriever. Use: bm25, dense, combined")

    @staticmethod
    def _extract_doc_id(filename: str) -> str:
        # recipe_001_chunk_02.txt → recipe_001
        return filename.split("_chunk")[0]

    def run(
            self,
            query: str,
            retriever_name: str = "combined",
            top_n: int = 15,
            top_k: int = 5
    ):
        retriever = self._get_retriever(retriever_name)
        retrieved = retriever.retrieve(query, top_n)

        if self.use_reranker:
            retrieved = self.reranker.rerank(query, retrieved, top_k=top_k)

        context_blocks = []
        doc_citation_map = {}
        citation_counter = 1

        for text, score in retrieved:
            chunk_index = self.chunks.index(text)
            filename = self.chunk_files[chunk_index]
            doc_id = self._extract_doc_id(filename)

            if doc_id not in doc_citation_map:
                doc_citation_map[doc_id] = citation_counter
                citation_counter += 1

            citation_id = doc_citation_map[doc_id]
            url = self.doc_metadata.get(doc_id, {}).get("url", "N/A")

            context_blocks.append(
                f"[{citation_id}] Документ: {doc_id}\n"
                f"Джерело: {url}\n"
                f"Текст:\n{text}"
            )

        context = "\n\n".join(context_blocks)

        prompt = f"""
    Ти аналітичний асистент.

    Використовуй ВИКЛЮЧНО інформацію з контексту нижче.
    Не узагальнюй, якщо немає підтвердження в джерелах.

    ПРАВИЛА:
    1. КОЖЕН абзац відповіді ОБОВʼЯЗКОВО має inline-цитату.
    2. Формат цитати: [1], [1][2].
    3. Номер цитати відповідає номеру джерела в контексті.
    4. Якщо факт є лише в одному джерелі — використовуй лише його.
    5. Не вигадуй джерел.

    ФОРМАТ:

    ВІДПОВІДЬ:
    <абзац> [1]
    <абзац> [1][2]

    ДЖЕРЕЛА:
    [1] recipe_xxx – URL
    [2] recipe_yyy – URL

    Питання:
    {query}

    Контекст:
    {context}
    """.strip()

        answer = self.llm.generate(prompt)

        used_docs_sorted = sorted(
            doc_citation_map.items(),
            key=lambda x: x[1]
        )

        sources = [
            f"[{cid}] {doc_id} – {self.doc_metadata.get(doc_id, {}).get('url', '')}"
            for doc_id, cid in used_docs_sorted
        ]

        final_answer = answer + "\n\nДЖЕРЕЛА:\n" + "\n".join(sources)
        used_docs = [doc_id for doc_id, _ in used_docs_sorted]

        return final_answer, used_docs

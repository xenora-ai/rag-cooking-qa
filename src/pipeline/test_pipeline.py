# src/pipeline/test_pipeline.py
from src.pipeline.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    pipeline = RAGPipeline(chunks_dir="../../data/chunks", doc_metadata_dir="../../data/data.csv", groq_api_key="your_key;)")

    query = "Що можна приготувати на сніданок швидко?"
    answer, used_docs = pipeline.run(
        query=query,
        retriever_name="combined",
        top_k=7
    )

    print("Відповідь:\n", answer)
    print("\nВикористані документи:", used_docs)

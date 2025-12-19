from sentence_transformers import SentenceTransformer, util


class SBERReranker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def rerank(self, query, retrieved, top_k=5):
        """
        retrieved: List[(text, score)]
        """
        texts = [t for t, s in retrieved]

        # Ембедінги
        query_emb = self.model.encode(query, convert_to_tensor=True)
        docs_emb = self.model.encode(texts, convert_to_tensor=True)

        # Косинусна схожість
        scores = util.cos_sim(query_emb, docs_emb)[0]  # shape [num_docs]

        # Сортуємо по зменшенню схожості
        top_results = sorted(
            zip(texts, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return top_results

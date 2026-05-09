from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        print(f"[INFO] Loaded reranker model: {model_name}")

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not retrieved_chunks:
            return []

        pairs = []
        valid_items = []
        for item in retrieved_chunks:
            text = (item.get("metadata") or {}).get("text", "")
            if not text:
                continue
            pairs.append((query, text))
            valid_items.append(item)

        if not pairs:
            return []

        scores = self.model.predict(pairs)
        reranked = []
        for item, score in zip(valid_items, scores):
            updated = dict(item)
            updated["rerank_score"] = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is None:
            return reranked
        return reranked[:top_k]

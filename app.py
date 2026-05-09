from src.search import RAGSearch


def main():
    query = "What is attention mechanism?"
    rag = RAGSearch(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model="gemma2-9b-it",
    )

    reranked_chunks = rag.retrieve_and_rerank(query, retrieve_k=10, rerank_k=3)
    print("\n=== Top Re-ranked Chunks ===")
    for i, chunk in enumerate(reranked_chunks, start=1):
        text = (chunk.get("metadata") or {}).get("text", "")
        score = chunk.get("rerank_score", 0.0)
        preview = text[:240].replace("\n", " ")
        print(f"{i}. score={score:.4f} | {preview}...")

    answer = rag.search_and_summarize(query, retrieve_k=10, rerank_k=3)
    print("\n=== Final Answer ===")
    print(answer)


if __name__ == "__main__":
    main()

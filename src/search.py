import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.vector_store import FaissVectorStore
from src.reranker import CrossEncoderReranker

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "gemma2-9b-it",
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents

            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def retrieve_and_rerank(
        self, query: str, retrieve_k: int = 10, rerank_k: int = 5
    ):
        retrieved = self.vectorstore.query(query, top_k=retrieve_k)
        reranked = self.reranker.rerank(query, retrieved, top_k=rerank_k)
        return reranked

    def search_and_summarize(
        self, query: str, retrieve_k: int = 10, rerank_k: int = 5
    ) -> str:
        reranked_results = self.retrieve_and_rerank(
            query, retrieve_k=retrieve_k, rerank_k=rerank_k
        )
        texts = [
            r["metadata"].get("text", "")
            for r in reranked_results
            if r.get("metadata")
        ]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, retrieve_k=10, rerank_k=3)
    print("Summary:", summary)

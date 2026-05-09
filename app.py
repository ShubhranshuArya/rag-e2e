from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline


if __name__ == "__main__":
    docs = load_all_documents(data_dir="data")
    doc_chunks = EmbeddingPipeline().chunk_documents(docs)
    embed_chunks = EmbeddingPipeline().embed_chunks(doc_chunks)

    print(doc_chunks)

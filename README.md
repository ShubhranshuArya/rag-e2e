# RAG Pipeline

## Aim of Project
Build a RAG pipeline.

## Tech Stack

<p>
  <img src="https://cdn.simpleicons.org/python" alt="Python" width="26" />
  Python
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/langchain" alt="LangChain" width="26" />
  LangChain
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/pytorch" alt="Sentence Transformers" width="26" />
  Sentence Transformers
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/meta" alt="FAISS" width="26" />
  FAISS
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/groq" alt="Groq" width="26" />
  Groq
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/meta" alt="Llama 3.2" width="26" />
  Llama 3.2
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/openai" alt="OpenAI" width="26" />
  OpenAI
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/chromadb" alt="ChromaDB" width="26" />
  ChromaDB
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/typesense" alt="Typesense" width="26" />
  Typesense
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/langgraph" alt="LangGraph" width="26" />
  LangGraph
</p>

## Learnings

![RAG Pipeline Architecture](assets/rag-architecture.png)

This architecture shows a clean RAG flow:

1. **Ingestion and chunking**: Source documents (including PDFs) are split into small chunks.
2. **Embedding creation**: Each chunk is converted into a dense vector by the embedding model.
3. **Vector indexing**: Chunk vectors are stored in the vector database with metadata.
4. **Query encoding**: The user question is converted into a query vector in the same embedding space.
5. **ANN retrieval**: Nearest chunk vectors are fetched quickly using approximate nearest-neighbor search.
6. **Re-ranking**: A re-ranker scores retrieved chunks again to keep the most relevant context.
7. **Prompt construction**: The top context is inserted into a prompt template with the original query.
8. **LLM generation**: The LLM uses that grounded context to generate the final answer.

Key takeaway: retrieval narrows the search space, re-ranking sharpens relevance, and the LLM answers with context instead of guessing.

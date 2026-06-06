# RAG Service

A simple Retrieval-Augmented Generation (RAG) API service built with FastAPI that runs entirely locally without external LLM APIs.

## Features

- Document indexing using TF-IDF vectorization
- Semantic search with cosine similarity ranking
- Extractive answer generation from retrieved document chunks
- In-memory storage for fast retrieval
- CORS support for browser-based access

## Project Structure

```
rag-service/
├── app.py                  # FastAPI application with /index and /ask endpoints
├── requirements.txt        # Python dependencies
├── README.md
├── docs/                   # Document files for indexing
│   ├── sample_product.txt
│   ├── sample_refund.txt
│   └── sample_pricing.txt
└── services/
    ├── __init__.py
    ├── chunking.py         # Document loading and text chunking
    ├── indexing.py         # TF-IDF index building and search
    └── retrieval.py        # Similarity-based retrieval and extractive answering
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Build the Index

```bash
curl -X POST http://localhost:8000/index
```

Response:
```json
{
  "documents": 3,
  "chunks": 15
}
```

### 4. Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Response:
```json
{
  "answer": "Our refund policy allows customers to return most products within 30 days...",
  "sources": [
    {
      "document": "sample_refund.txt",
      "chunk_id": 0,
      "score": 0.61
    }
  ],
  "confidence": "answered_from_docs"
}
```

### 5. Health Check

```bash
curl http://localhost:8000/health
```

## API Endpoints

### POST /index

Reads all `.txt` files from the `docs/` directory, splits them into chunks of approximately 300 characters, and builds a TF-IDF index in memory.

### POST /ask

Queries the index with a question and returns a concise answer extracted from the retrieved documents, along with source information including document name, chunk ID, and relevance score.

Request body:
```json
{
  "question": "Your question here"
}
```

### GET /health

Returns the health status of the service.

## How It Works

1. **Chunking**: Documents are split by paragraphs, then by sentences into ~300 character chunks.
2. **Indexing**: TF-IDF vectors are computed for each chunk using scikit-learn.
3. **Retrieval**: The query is vectorized and cosine similarity is computed against all chunks. Top 3 most similar chunks are retrieved.
4. **Answering**: Sentences from the retrieved chunks are ranked by similarity to the question, and the top sentences are combined into a concise answer.

## Tradeoffs and Future Improvements

### Current Tradeoffs

- **TF-IDF vs. Embeddings**: TF-IDF is lightweight and runs locally without any model downloads, but it cannot capture semantic meaning beyond keyword overlap. For example, "laptop warranty" and "computer coverage" would not match well despite meaning similar things. Dense embeddings (e.g., sentence-transformers) would improve retrieval quality at the cost of additional dependencies and memory.

- **Extractive vs. Generative Answers**: The current approach selects the most relevant sentences from retrieved chunks. This guarantees factual accuracy (no hallucination) but produces answers that are verbatim excerpts rather than natural paraphrases. A small local model (e.g., GPT-2 or FLAN-T5) could generate more fluent answers while still avoiding external API calls.

- **In-Memory Index**: The TF-IDF matrix is stored entirely in RAM, which is fast but does not persist across restarts. For production use, a persistent vector database (e.g., Chroma, FAISS) would be preferable.

- **Chunking Strategy**: Documents are split by paragraphs and sentences into ~300 character chunks. This works well for the sample documents but may not be optimal for all content types. A single very long sentence (e.g., 2500 characters with no punctuation) will not be split further, which is a known limitation.

### What to Improve Next

1. **Add persistent storage** so the index survives server restarts (e.g., save TF-IDF vectors to disk).
2. **Improve chunking** to handle edge cases like very long sentences by adding a hard character limit with sentence-aware splitting.
3. **Add query expansion** to handle synonyms and improve recall (e.g., using WordNet or simple stemming).
4. **Implement re-ranking** of retrieved chunks using a more sophisticated scoring mechanism.
5. **Add unit tests** with pytest for automated regression testing.
6. **Add request rate limiting** and input sanitization for production deployment.

## Requirements

- Python 3.11+
- No external API keys or internet access needed

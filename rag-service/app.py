import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from services.chunking import load_and_chunk_docs
from services.indexing import Indexer
from services.retrieval import retrieve, generate_answer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join(os.path.dirname(__file__), 'docs')

app = FastAPI(title="RAG API Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

indexer: Indexer | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class SourceInfo(BaseModel):
    document: str
    chunk_id: int
    score: float


class IndexResponse(BaseModel):
    documents: int
    chunks: int


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    confidence: str


@app.get("/")
def root():
    return {
        "service": "RAG API Service",
        "endpoints": {
            "POST /index": "Build document index",
            "POST /ask": "Ask a question"
        }
    }


@app.post("/index", response_model=IndexResponse)
def build_index():
    global indexer

    try:
        chunks, doc_count = load_and_chunk_docs(DOCS_DIR)
        if doc_count == 0:
            raise ValueError("No .txt documents found in the docs/ directory.")
        indexer = Indexer()
        indexer.build(chunks, doc_count)
        logger.info(f"Index built: {doc_count} documents, {len(chunks)} chunks")
        return IndexResponse(documents=doc_count, chunks=len(chunks))

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during indexing.")


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    global indexer

    if indexer is None:
        raise HTTPException(status_code=400, detail="Index has not been built. Call POST /index first.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        results = retrieve(question, indexer)
        answer, sources, confidence = generate_answer(results, question, indexer.vectorizer)
        return AskResponse(answer=answer, sources=sources, confidence=confidence)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing question.")


@app.get("/health")
def health():
    return {"status": "healthy", "index_ready": indexer is not None}


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  RAG API Service")
    print("  http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

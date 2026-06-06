import threading

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.chunking import Chunk


class IndexNotBuiltError(Exception):
    pass


class Indexer:
    def __init__(self):
        self._lock = threading.Lock()
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self.doc_count: int = 0

    def build(self, chunks: list[Chunk], doc_count: int):
        with self._lock:
            self.chunks = chunks
            self.doc_count = doc_count
            texts = [chunk.text for chunk in chunks]
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            self.matrix = self.vectorizer.fit_transform(texts)

    def is_built(self) -> bool:
        return self.matrix is not None and self.vectorizer is not None

    def search(self, query: str, top_k: int = 3) -> tuple[list[Chunk], list[float]]:
        if not self.is_built():
            raise IndexNotBuiltError("Index has not been built. Call build() first.")

        with self._lock:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.matrix).flatten()

            top_indices = np.argsort(similarities)[::-1][:top_k]

            result_chunks = [self.chunks[i] for i in top_indices]
            result_scores = [float(similarities[i]) for i in top_indices]

            return result_chunks, result_scores

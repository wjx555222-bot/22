import re
from sklearn.metrics.pairwise import cosine_similarity

from services.chunking import Chunk
from services.indexing import Indexer


SIMILARITY_THRESHOLD = 0.05


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r'(Mr|Mrs|Dr|Ms|St|Jr|Sr)\.', r'\1<DOT>', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip().replace('<DOT>', '.') for s in sentences if s.strip()]
    return sentences


def _extractive_answer(question: str, chunks: list[Chunk], vectorizer) -> str:
    all_sentences: list[tuple[str, str]] = []
    for chunk in chunks:
        sentences = _split_sentences(chunk.text)
        for sent in sentences:
            all_sentences.append((sent, chunk.source))

    if not all_sentences:
        return ''

    if not question.strip():
        return ' '.join(s[0] for s in all_sentences)

    q_vec = vectorizer.transform([question])
    sent_vecs = vectorizer.transform([s[0] for s in all_sentences])
    sims = cosine_similarity(q_vec, sent_vecs).flatten()

    sentence_scores = list(zip(all_sentences, sims))
    sentence_scores.sort(key=lambda x: x[1], reverse=True)

    selected: list[str] = []
    total_len = 0
    max_len = 500

    for (sent, source), score in sentence_scores:
        if score < SIMILARITY_THRESHOLD:
            break
        if len(selected) >= 5:
            break
        if total_len + len(sent) > max_len:
            break
        if sent in selected:
            continue
        selected.append(sent)
        total_len += len(sent) + 1

    return ' '.join(selected)


def retrieve(question: str, indexer: Indexer, top_k: int = 3) -> list[dict]:
    result_chunks, result_scores = indexer.search(question, top_k=top_k)
    return [{"chunk": chunk, "score": score} for chunk, score in zip(result_chunks, result_scores)]


def generate_answer(results: list[dict], question: str, vectorizer) -> tuple[str, list[dict], str]:
    chunks = [r["chunk"] for r in results]
    scores = [r["score"] for r in results]

    if not chunks or all(s < SIMILARITY_THRESHOLD for s in scores):
        return ("Insufficient evidence found in the indexed documents.", [], "insufficient_context")

    answer = _extractive_answer(question, chunks, vectorizer)

    if not answer:
        return ("Insufficient evidence found in the indexed documents.", [], "insufficient_context")

    sources = []
    for r in results:
        chunk = r["chunk"]
        score = r["score"]
        if score >= SIMILARITY_THRESHOLD:
            sources.append({"document": chunk.source, "chunk_id": chunk.index, "score": score})

    return (answer, sources, "answered_from_docs")

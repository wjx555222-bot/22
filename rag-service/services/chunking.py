import re
import os
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    index: int


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r'(Mr|Mrs|Dr|Ms|St|Jr|Sr)\.', r'\1<DOT>', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip().replace('<DOT>', '.') for s in sentences if s.strip()]
    return sentences


def _get_word_boundary_overlap(text: str, overlap: int) -> str:
    """Get the last `overlap` characters of text, breaking at a word boundary."""
    if len(text) <= overlap:
        return text
    overlap_text = text[-overlap:]
    # Find the first space to break at a word boundary
    first_space = overlap_text.find(' ')
    if first_space != -1 and first_space < overlap:
        return overlap_text[first_space + 1:]
    return overlap_text


def chunk_document(text: str, source: str, chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks: list[Chunk] = []
    chunk_index = 0

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(Chunk(text=para, source=source, index=chunk_index))
            chunk_index += 1
            continue

        sentences = split_into_sentences(para)
        current_chunk = ''
        for sentence in sentences:
            if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                chunks.append(Chunk(text=current_chunk.strip(), source=source, index=chunk_index))
                chunk_index += 1
                current_chunk = _get_word_boundary_overlap(current_chunk, overlap)
                current_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(Chunk(text=current_chunk.strip(), source=source, index=chunk_index))
            chunk_index += 1

    return chunks


def load_and_chunk_docs(docs_dir: str, chunk_size: int = 300) -> tuple[list[Chunk], int]:
    all_chunks: list[Chunk] = []
    doc_count = 0

    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith('.txt'):
            continue
        doc_count += 1
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_document(text, source=filename, chunk_size=chunk_size)
        all_chunks.extend(chunks)

    return all_chunks, doc_count

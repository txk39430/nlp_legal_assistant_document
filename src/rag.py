from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from .groq_qa import answer_question_groq, summarize_with_groq
# Ensuring punkt is available for sentence splitting
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


@lru_cache(maxsize=1)
def load_embedder() -> SentenceTransformer:
    """
    Loading a sentence-transformer model once and cache it.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def chunk_text(
    text: str,
    max_sentences_per_chunk: int = 5,
    overlap: int = 1,
) -> List[str]:
    """
    Spliting long text into overlapping sentence chunks.

    Example:
        [s1, s2, s3, s4, s5, s6] with max_sentences_per_chunk=3, overlap=1
        -> [s1 s2 s3], [s3 s4 s5], [s5 s6]
    """
    sentences = sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents = sentences[i : i + max_sentences_per_chunk]
        chunk_text = " ".join(chunk_sents).strip()
        if chunk_text:
            chunks.append(chunk_text)
        i += max_sentences_per_chunk - overlap
        if max_sentences_per_chunk == overlap:  # avoid infinite loop
            break
    return chunks


def build_index(chunks: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Building an in-memory 'index':
      - chunk_texts: the original text chunks
      - embeddings: 2D numpy array of shape (num_chunks, dim)
    """
    embedder = load_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return chunks, embeddings


def retrieve_top_k(
    question: str,
    chunk_texts: List[str],
    embeddings: np.ndarray,
    top_k: int = 3,
) -> List[str]:
    """
    Retrieving top-k most relevant chunks for a question using cosine similarity.
    """
    embedder = load_embedder()
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Cosine similarity since vectors are normalized: dot product
    scores = embeddings @ q_emb  # shape: (num_chunks,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunk_texts[i] for i in top_indices]


def answer_question_rag(question: str, full_context: str, top_k: int = 3) -> dict:
    """
    End-to-end RAG-style QA:
      1. Chunks the full context into sentence windows.
      2. Embeds all chunks and the question.
      3. Retrieves top-k chunks.
      4. Concatenates these chunks into a focused context.
      5. Asks Groq LLM (Llama3) to answer using only that focused context.
    """
    # 1–2. Building index
    chunks = chunk_text(full_context)
    if not chunks:
        return {
            "answer": "No usable text found in the context.",
            "retrieved_chunks": [],
        }

    chunk_texts, embeddings = build_index(chunks)

    # 3. Retrievng top-k relevant chunks
    top_chunks = retrieve_top_k(question, chunk_texts, embeddings, top_k=top_k)

    # 4. Concatenating into a smaller context for Groq
    focused_context = "\n\n".join(top_chunks)

    # 5. Asking Groq using the focused context
    answer = answer_question_groq(question=question, context=focused_context)

    return {
        "answer": answer,
        "retrieved_chunks": top_chunks,
    }

def summarize_rag(full_text: str, top_k: int = 5) -> dict:
    """
    RAG-style summarization:
      1. Chunk the full text.
      2. Use a generic 'summary' query to retrieve top-k chunks.
      3. Ask Groq to summarize only those chunks.
    """
    chunks = chunk_text(full_text)
    if not chunks:
        return {
            "summary": "No usable text found to summarize.",
            "retrieved_chunks": [],
        }

    chunk_texts, embeddings = build_index(chunks)

    # Generic query representing "summary of the document"
    summary_query = (
        "What are the main obligations, rights, and key points described "
        "in this legal or policy text?"
    )

    top_chunks = retrieve_top_k(summary_query, chunk_texts, embeddings, top_k=top_k)
    focused_context = "\n\n".join(top_chunks)

    summary = summarize_with_groq(focused_context)

    return {
        "summary": summary,
        "retrieved_chunks": top_chunks,
    }

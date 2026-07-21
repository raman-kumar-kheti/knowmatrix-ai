"""Chunk reranking using lexical overlap + optional CrossEncoder."""

from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import settings
from app.rag.vector_store import RetrievedChunk


def _query_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.RERANKER_MODEL, device="cpu")


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 4,
) -> list[RetrievedChunk]:
    """Rerank retrieved chunks by term coverage + hybrid score (or CrossEncoder if enabled)."""
    if not chunks:
        return []

    if settings.ENABLE_CROSS_ENCODER_RERANK:
        model = _get_cross_encoder()
        pairs = [(query, c.document.page_content) for c in chunks]
        scores = model.predict(pairs)
        reranked = []
        for chunk, score in zip(chunks, scores):
            reranked.append(
                RetrievedChunk(
                    document=chunk.document,
                    semantic_score=chunk.semantic_score,
                    bm25_score=chunk.bm25_score,
                    hybrid_score=round(float(score), 4),
                )
            )
        reranked.sort(key=lambda c: c.hybrid_score, reverse=True)
        return reranked[:top_k]

    q_terms = _query_terms(query)
    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        c_terms = _query_terms(chunk.document.page_content)
        overlap = len(q_terms & c_terms) / max(len(q_terms), 1)
        combined = 0.75 * chunk.hybrid_score + 0.25 * overlap
        scored.append(
            RetrievedChunk(
                document=chunk.document,
                semantic_score=chunk.semantic_score,
                bm25_score=chunk.bm25_score,
                hybrid_score=round(combined, 4),
            )
        )
    scored.sort(key=lambda c: c.hybrid_score, reverse=True)
    return scored[:top_k]

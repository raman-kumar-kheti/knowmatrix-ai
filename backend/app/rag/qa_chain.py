"""End-to-end RAG orchestration with citations, evaluation, and telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.rag.evaluator import RagEvaluationResult, evaluate_rag_response
from app.rag.llm_provider import estimate_tokens, generate_answer
from app.rag.reranker import rerank_chunks
from app.rag.vector_store import HybridVectorStore, RetrievedChunk

_store_instance: HybridVectorStore | None = None


def get_vector_store() -> HybridVectorStore:
    """Return the singleton HybridVectorStore instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = HybridVectorStore()
    return _store_instance


@dataclass
class RagAnswerOutput:
    """Complete RAG response with citations, evaluation, and telemetry."""

    answer: str
    citations: list[dict[str, Any]]
    evaluation: RagEvaluationResult
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def answer_question(
    question: str,
    top_k: int | None = None,
    user_role: str = "public",
    chat_history: list[tuple[str, str]] | None = None,
    provider: str | None = None,
    store: HybridVectorStore | None = None,
) -> RagAnswerOutput:
    """Retrieve, rerank, generate, and evaluate an answer for a user question."""
    t0 = time.perf_counter()
    k = top_k or settings.TOP_K
    vs = store or get_vector_store()

    candidates: list[RetrievedChunk] = vs.search(query=question, top_k=max(k * 2, 6), user_role=user_role)
    reranked = rerank_chunks(query=question, chunks=candidates, top_k=k)

    if not reranked or max(c.hybrid_score for c in reranked) < settings.MIN_RELEVANCE_SCORE:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        eval_res = evaluate_rag_response(
            question=question,
            answer="Insufficient evidence in the indexed documents to answer this question reliably.",
            chunks=reranked,
            min_relevance_score=settings.MIN_RELEVANCE_SCORE,
        )
        return RagAnswerOutput(
            answer="Insufficient evidence in the indexed documents to answer this question reliably.",
            citations=[],
            evaluation=eval_res,
            latency_ms=latency_ms,
            prompt_tokens=estimate_tokens(question),
            completion_tokens=12,
        )

    context = "\n\n".join(
        f"[{c.document.metadata.get('source', 'doc')} p.{c.document.metadata.get('page', '?')}] {c.document.page_content}"
        for c in reranked
    )
    answer = generate_answer(
        question=question,
        context=context,
        chat_history=chat_history,
        provider=provider,
    )
    eval_res = evaluate_rag_response(
        question=question,
        answer=answer,
        chunks=reranked,
        min_relevance_score=settings.MIN_RELEVANCE_SCORE,
    )

    citations = [
        {
            "doc_id": c.document.metadata.get("doc_id", ""),
            "source": c.document.metadata.get("source", "unknown"),
            "page": c.document.metadata.get("page", 1),
            "score": c.hybrid_score,
            "snippet": c.document.page_content[:220],
        }
        for c in reranked
    ]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return RagAnswerOutput(
        answer=answer,
        citations=citations,
        evaluation=eval_res,
        latency_ms=latency_ms,
        prompt_tokens=estimate_tokens(context + " " + question),
        completion_tokens=estimate_tokens(answer),
    )

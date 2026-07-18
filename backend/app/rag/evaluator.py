"""RAGAS-inspired retrieval quality, faithfulness, and hallucination detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.vector_store import RetrievedChunk

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "with", "by", "as", "that", "this", "it", "be", "from",
}


def _keywords(text: str) -> set[str]:
    return {
        w for w in re.findall(r"\w+", text.lower())
        if len(w) > 2 and w not in STOPWORDS
    }


@dataclass
class RagEvaluationResult:
    """Evaluation metrics for a single RAG response."""

    context_precision: float
    faithfulness_score: float
    answer_relevance: float
    hallucination_risk: str
    insufficient_evidence: bool


def evaluate_rag_response(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    min_relevance_score: float = 0.25,
) -> RagEvaluationResult:
    """Compute retrieval precision, answer faithfulness, relevance, and hallucination risk."""
    if not chunks:
        return RagEvaluationResult(
            context_precision=0.0,
            faithfulness_score=0.0,
            answer_relevance=0.0,
            hallucination_risk="high",
            insufficient_evidence=True,
        )

    context_precision = round(
        sum(c.hybrid_score for c in chunks) / len(chunks),
        4,
    )
    top_score = max(c.hybrid_score for c in chunks)
    insufficient = top_score < min_relevance_score

    context_text = " ".join(c.document.page_content for c in chunks)
    context_kw = _keywords(context_text)
    answer_kw = _keywords(answer)
    question_kw = _keywords(question)

    if answer_kw:
        faithfulness = len(answer_kw & context_kw) / len(answer_kw)
    else:
        faithfulness = 0.0

    if question_kw:
        relevance = len(question_kw & answer_kw) / len(question_kw)
    else:
        relevance = 0.0

    faithfulness = round(min(1.0, faithfulness), 4)
    relevance = round(min(1.0, relevance), 4)

    if insufficient or faithfulness < 0.35:
        risk = "high"
    elif faithfulness < 0.60:
        risk = "medium"
    else:
        risk = "low"

    return RagEvaluationResult(
        context_precision=context_precision,
        faithfulness_score=faithfulness,
        answer_relevance=relevance,
        hallucination_risk=risk,
        insufficient_evidence=insufficient,
    )

"""Chat history and RAG telemetry metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ChatHistory, DocumentRecord, QueryLog
from app.models.schema import ChatHistoryItem

router = APIRouter()


@router.get("/history", response_model=list[ChatHistoryItem], tags=["History"])
def get_chat_history(
    session_id: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[ChatHistoryItem]:
    """Retrieve persisted chat turns, optionally filtered by session_id."""
    query = db.query(ChatHistory)
    if session_id:
        query = query.filter(ChatHistory.session_id == session_id)
    rows = query.order_by(ChatHistory.created_at.desc()).limit(limit).all()
    return [
        ChatHistoryItem(
            id=r.id,
            session_id=r.session_id,
            question=r.question,
            answer=r.answer,
            insufficient_evidence=bool(r.insufficient_evidence),
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/metrics", tags=["Evaluation"])
def get_rag_metrics(db: Session = Depends(get_db)) -> dict[str, float | int]:
    """Return aggregate RAG latency, token usage, and evaluation quality metrics."""
    doc_count = db.query(func.count(DocumentRecord.id)).scalar() or 0
    query_count = db.query(func.count(QueryLog.id)).scalar() or 0
    avg_latency = db.query(func.avg(QueryLog.latency_ms)).scalar() or 0.0
    avg_faithfulness = db.query(func.avg(QueryLog.faithfulness_score)).scalar() or 0.0
    avg_relevance = db.query(func.avg(QueryLog.relevance_score)).scalar() or 0.0
    total_tokens = (
        db.query(func.sum(QueryLog.prompt_tokens + QueryLog.completion_tokens)).scalar() or 0
    )
    return {
        "indexed_documents": int(doc_count),
        "total_queries": int(query_count),
        "avg_latency_ms": round(float(avg_latency), 2),
        "avg_faithfulness_score": round(float(avg_faithfulness), 4),
        "avg_relevance_score": round(float(avg_relevance), 4),
        "total_tokens": int(total_tokens),
    }

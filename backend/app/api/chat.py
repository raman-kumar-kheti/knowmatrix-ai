"""Chat and SSE streaming endpoints with conversational memory and evaluation logging."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ChatHistory, QueryLog
from app.models.schema import ChatRequest, ChatResponse, Citation, EvaluationMetrics
from app.rag.llm_provider import stream_answer_tokens
from app.rag.qa_chain import answer_question, get_vector_store

router = APIRouter()


def _load_session_history(db: Session, session_id: str) -> list[tuple[str, str]]:
    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(5)
        .all()
    )
    return [(r.question, r.answer) for r in reversed(rows)]


def _persist_turn(db: Session, req: ChatRequest, out) -> None:
    db.add(
        ChatHistory(
            session_id=req.session_id,
            user_role=req.user_role,
            question=req.question,
            answer=out.answer,
            citations_json=json.dumps(out.citations),
            insufficient_evidence=1 if out.evaluation.insufficient_evidence else 0,
        )
    )
    db.add(
        QueryLog(
            session_id=req.session_id,
            question=req.question,
            latency_ms=out.latency_ms,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            faithfulness_score=out.evaluation.faithfulness_score,
            relevance_score=out.evaluation.answer_relevance,
            hallucination_risk=out.evaluation.hallucination_risk,
        )
    )
    db.commit()


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Answer a question using hybrid RAG, citations, and evaluation metrics."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if get_vector_store().total_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a document first.")

    history = _load_session_history(db, request.session_id)
    out = answer_question(
        question=request.question,
        top_k=request.top_k,
        user_role=request.user_role,
        chat_history=history,
        provider=request.provider,
    )
    _persist_turn(db, request, out)

    return ChatResponse(
        answer=out.answer,
        session_id=request.session_id,
        citations=[Citation(**c) for c in out.citations],
        evaluation=EvaluationMetrics(
            context_precision=out.evaluation.context_precision,
            faithfulness_score=out.evaluation.faithfulness_score,
            answer_relevance=out.evaluation.answer_relevance,
            hallucination_risk=out.evaluation.hallucination_risk,
            insufficient_evidence=out.evaluation.insufficient_evidence,
        ),
        latency_ms=out.latency_ms,
        prompt_tokens=out.prompt_tokens,
        completion_tokens=out.completion_tokens,
    )


@router.post("/chat/stream", tags=["Chat"])
def chat_stream(request: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """Stream the RAG answer over Server-Sent Events (SSE) followed by citations & metrics."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if get_vector_store().total_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a document first.")

    history = _load_session_history(db, request.session_id)
    out = answer_question(
        question=request.question,
        top_k=request.top_k,
        user_role=request.user_role,
        chat_history=history,
        provider=request.provider,
    )
    _persist_turn(db, request, out)

    def event_generator():
        for token in stream_answer_tokens(out.answer):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        meta = {
            "type": "metadata",
            "citations": out.citations,
            "evaluation": {
                "context_precision": out.evaluation.context_precision,
                "faithfulness_score": out.evaluation.faithfulness_score,
                "answer_relevance": out.evaluation.answer_relevance,
                "hallucination_risk": out.evaluation.hallucination_risk,
                "insufficient_evidence": out.evaluation.insufficient_evidence,
            },
            "latency_ms": out.latency_ms,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

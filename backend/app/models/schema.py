"""Pydantic schemas for document management, RAG chat, citations, and evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    source: str
    page: int
    score: float
    snippet: str


class EvaluationMetrics(BaseModel):
    context_precision: float
    faithfulness_score: float
    answer_relevance: float
    hallucination_risk: Literal["low", "medium", "high"]
    insufficient_evidence: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    user_role: Literal["public", "internal", "admin"] = "public"
    top_k: int = Field(default=4, ge=1, le=12)
    provider: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: list[Citation]
    evaluation: EvaluationMetrics
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    access_role: str
    page_count: int
    chunk_count: int
    status: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    access_role: str
    page_count: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: datetime


class ChatHistoryItem(BaseModel):
    id: int
    session_id: str
    question: str
    answer: str
    insufficient_evidence: bool
    created_at: datetime

"""End-to-end tests for KnowMatrix AI (Enterprise Knowledge Assistant)."""

from __future__ import annotations

import pymupdf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 520, 720), text)
    return doc.tobytes()


def test_health_endpoint() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_pdf_upload_hybrid_search_citations_and_delete() -> None:
    pdf_bytes = _make_pdf_bytes(
        "KnowMatrix AI uses hybrid BM25 and FAISS dense retrieval. "
        "All employee expense reports must be submitted within 14 days of travel."
    )
    upload_res = client.post(
        "/api/upload",
        files={"file": ("policy.pdf", pdf_bytes, "application/pdf")},
        data={"access_role": "public", "chunk_size": "500", "chunk_overlap": "50"},
    )
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["doc_id"]
    assert upload_res.json()["page_count"] == 1

    # Query with extractive provider for fast deterministic CPU verification
    chat_res = client.post(
        "/api/chat",
        json={
            "question": "Within how many days must expense reports be submitted?",
            "session_id": "test-session",
            "user_role": "public",
            "top_k": 3,
            "provider": "extractive",
        },
    )
    assert chat_res.status_code == 200
    body = chat_res.json()
    assert "14 days" in body["answer"]
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source"] == "policy.pdf"
    assert body["citations"][0]["page"] == 1
    assert "faithfulness_score" in body["evaluation"]
    assert body["latency_ms"] >= 0

    # Streaming SSE endpoint
    stream_res = client.post(
        "/api/chat/stream",
        json={
            "question": "What retrieval method does KnowMatrix AI use?",
            "session_id": "test-session",
            "user_role": "public",
            "provider": "extractive",
        },
    )
    assert stream_res.status_code == 200
    assert "data:" in stream_res.text

    # Reindex
    reindex_res = client.post(f"/api/documents/{doc_id}/reindex?chunk_size=300&chunk_overlap=40")
    assert reindex_res.status_code == 200

    # Metrics & History
    assert client.get("/api/metrics").status_code == 200
    assert client.get("/api/history?session_id=test-session").status_code == 200

    # Delete
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"


def test_access_control_blocks_internal_doc_for_public_user() -> None:
    secret_pdf = _make_pdf_bytes("Project Apollo launch budget is 4.2 million dollars.")
    up = client.post(
        "/api/upload",
        files={"file": ("internal_memo.pdf", secret_pdf, "application/pdf")},
        data={"access_role": "internal"},
    )
    assert up.status_code == 200
    doc_id = up.json()["doc_id"]

    try:
        public_res = client.post(
            "/api/chat",
            json={
                "question": "What is the Project Apollo launch budget?",
                "user_role": "public",
                "provider": "extractive",
            },
        )
        assert public_res.status_code == 200
        assert public_res.json()["evaluation"]["insufficient_evidence"] is True

        internal_res = client.post(
            "/api/chat",
            json={
                "question": "What is the Project Apollo launch budget?",
                "user_role": "internal",
                "provider": "extractive",
            },
        )
        assert internal_res.status_code == 200
        assert "4.2 million" in internal_res.json()["answer"]
    finally:
        client.delete(f"/api/documents/{doc_id}")

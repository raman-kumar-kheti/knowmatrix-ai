"""Document upload, listing, deletion, and re-indexing endpoints."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.db.models import DocumentRecord
from app.models.schema import DocumentInfo, UploadResponse
from app.rag.chunker import chunk_pages
from app.rag.loader import load_document_pages
from app.rag.qa_chain import get_vector_store
from app.utils.logger import logger

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    access_role: str = Form("public"),
    chunk_size: int = Form(settings.CHUNK_SIZE),
    chunk_overlap: int = Form(settings.CHUNK_OVERLAP),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Upload a PDF/TXT/MD document, extract pages, chunk, and index into FAISS + BM25."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    safe_name = os.path.basename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")

    if access_role not in {"public", "internal", "admin"}:
        raise HTTPException(status_code=400, detail="access_role must be public, internal, or admin.")

    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_MB} MB limit.")

    doc_id = uuid.uuid4().hex[:12]
    dest_path = settings.UPLOAD_DIR / f"{doc_id}_{safe_name}"
    dest_path.write_bytes(content)

    pages = load_document_pages(dest_path)
    if not pages:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No extractable text found in document.")

    chunks = chunk_pages(
        pages=pages,
        doc_id=doc_id,
        filename=safe_name,
        access_role=access_role,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vs = get_vector_store()
    vs.add_documents(chunks)

    record = DocumentRecord(
        doc_id=doc_id,
        filename=safe_name,
        file_path=str(dest_path),
        access_role=access_role,
        page_count=len(pages),
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    db.add(record)
    db.commit()
    logger.info("Indexed document %s (%s) with %d chunks", doc_id, safe_name, len(chunks))

    return UploadResponse(
        doc_id=doc_id,
        filename=safe_name,
        access_role=access_role,
        page_count=len(pages),
        chunk_count=len(chunks),
        status=f"Indexed {len(pages)} pages into {len(chunks)} chunks",
    )


@router.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentInfo]:
    """List all indexed documents."""
    rows = db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()
    return [
        DocumentInfo(
            doc_id=r.doc_id,
            filename=r.filename,
            access_role=r.access_role,
            page_count=r.page_count,
            chunk_count=r.chunk_count,
            chunk_size=r.chunk_size,
            chunk_overlap=r.chunk_overlap,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str, db: Session = Depends(get_db)) -> dict[str, str | int]:
    """Delete a document from DB, disk, and re-index the vector store."""
    record = db.query(DocumentRecord).filter(DocumentRecord.doc_id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")

    removed_chunks = get_vector_store().delete_document(doc_id)
    Path(record.file_path).unlink(missing_ok=True)
    db.delete(record)
    db.commit()
    return {"status": "deleted", "doc_id": doc_id, "removed_chunks": removed_chunks}


@router.post("/documents/{doc_id}/reindex", response_model=UploadResponse, tags=["Documents"])
def reindex_document(
    doc_id: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Re-chunk and re-index an existing document with updated chunk parameters."""
    record = db.query(DocumentRecord).filter(DocumentRecord.doc_id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")

    pages = load_document_pages(record.file_path)
    vs = get_vector_store()
    vs.delete_document(doc_id)
    chunks = chunk_pages(
        pages=pages,
        doc_id=doc_id,
        filename=record.filename,
        access_role=record.access_role,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vs.add_documents(chunks)

    record.chunk_count = len(chunks)
    record.chunk_size = chunk_size
    record.chunk_overlap = chunk_overlap
    db.commit()

    return UploadResponse(
        doc_id=doc_id,
        filename=record.filename,
        access_role=record.access_role,
        page_count=len(pages),
        chunk_count=len(chunks),
        status=f"Re-indexed into {len(chunks)} chunks",
    )

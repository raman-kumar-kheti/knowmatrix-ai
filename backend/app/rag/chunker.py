"""Configurable page-aware text chunking."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.rag.loader import ExtractedPage


def chunk_pages(
    pages: list[ExtractedPage],
    doc_id: str,
    filename: str,
    access_role: str = "public",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split extracted pages into overlapping chunks while preserving page citations."""
    resolved_size = chunk_size or settings.CHUNK_SIZE
    resolved_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    if resolved_overlap >= resolved_size:
        resolved_overlap = max(0, resolved_size // 4)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_size,
        chunk_overlap=resolved_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    chunk_idx = 0
    for page in pages:
        for chunk in splitter.split_text(page.text):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            documents.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        "doc_id": doc_id,
                        "source": filename,
                        "page": page.page_number,
                        "chunk_id": chunk_idx,
                        "access_role": access_role,
                    },
                )
            )
            chunk_idx += 1

    return documents

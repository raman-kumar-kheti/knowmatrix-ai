"""Page-aware document parsing using PyMuPDF (fitz) and plain text readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class ExtractedPage:
    """Extracted text for a single document page."""

    page_number: int
    text: str


def load_document_pages(file_path: str | Path) -> list[ExtractedPage]:
    """Extract text page-by-page from PDF, TXT, or Markdown files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages: list[ExtractedPage] = []
        with pymupdf.open(str(path)) as doc:
            for idx, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append(ExtractedPage(page_number=idx + 1, text=text))
        return pages

    if suffix in {".txt", ".md"}:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        return [ExtractedPage(page_number=1, text=content)] if content else []

    raise ValueError(f"Unsupported file format: {suffix}. Allowed: .pdf, .txt, .md")

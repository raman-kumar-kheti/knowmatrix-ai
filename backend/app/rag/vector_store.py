"""Multi-document FAISS + BM25 hybrid vector store with access control and re-indexing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.embeddings import get_embeddings_model

ROLE_HIERARCHY: dict[str, set[str]] = {
    "public": {"public"},
    "internal": {"public", "internal"},
    "admin": {"public", "internal", "admin"},
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class RetrievedChunk:
    """A retrieved chunk with dense, BM25, and fused hybrid scores."""

    document: Document
    semantic_score: float
    bm25_score: float
    hybrid_score: float


class HybridVectorStore:
    """Manages multi-document FAISS persistence and BM25 lexical search."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or settings.VECTOR_DB_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.store_dir / "chunks.json"
        self._faiss: FAISS | None = None
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._load_existing()

    def _load_existing(self) -> None:
        if self.chunks_path.exists():
            raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            self._documents = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in raw
            ]
            if self._documents:
                self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._documents])

        index_file = self.store_dir / "index.faiss"
        if index_file.exists():
            self._faiss = FAISS.load_local(
                str(self.store_dir),
                get_embeddings_model(),
                allow_dangerous_deserialization=True,
            )

    def _persist(self) -> None:
        serialized = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in self._documents
        ]
        self.chunks_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
        if self._documents:
            self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._documents])
            if self._faiss is not None:
                self._faiss.save_local(str(self.store_dir))
        else:
            self._bm25 = None
            self._faiss = None
            for fname in ("index.faiss", "index.pkl"):
                fpath = self.store_dir / fname
                if fpath.exists():
                    fpath.unlink()

    def add_documents(self, docs: list[Document]) -> int:
        """Append new document chunks to FAISS and BM25 without overwriting existing docs."""
        if not docs:
            return 0
        embeddings = get_embeddings_model()
        if self._faiss is None:
            self._faiss = FAISS.from_documents(docs, embeddings)
        else:
            self._faiss.add_documents(docs)
        self._documents.extend(docs)
        self._persist()
        return len(docs)

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks for doc_id and rebuild the FAISS + BM25 index."""
        before = len(self._documents)
        remaining = [d for d in self._documents if d.metadata.get("doc_id") != doc_id]
        removed = before - len(remaining)
        if removed == 0:
            return 0
        self._documents = remaining
        if remaining:
            self._faiss = FAISS.from_documents(remaining, get_embeddings_model())
        else:
            self._faiss = None
        self._persist()
        return removed

    def search(
        self,
        query: str,
        top_k: int = 4,
        user_role: str = "public",
        alpha: float | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid semantic + BM25 search filtered by document-level access role."""
        if not self._documents or self._faiss is None:
            return []

        allowed_roles = ROLE_HIERARCHY.get(user_role, {"public"})
        w_semantic = settings.HYBRID_ALPHA if alpha is None else alpha
        w_bm25 = 1.0 - w_semantic

        # Dense search with oversampling to allow role filtering
        fetch_k = min(len(self._documents), max(top_k * 4, 16))
        dense_results = self._faiss.similarity_search_with_relevance_scores(query, k=fetch_k)

        # BM25 scores over all documents
        bm25_raw = self._bm25.get_scores(_tokenize(query)) if self._bm25 is not None else [0.0] * len(self._documents)
        max_bm25 = max(float(max(bm25_raw)) if len(bm25_raw) > 0 else 0.0, 1e-6)

        # Map chunk key -> (Document, semantic_score)
        candidates: dict[tuple[str, int], dict[str, Any]] = {}
        for doc, sem_score in dense_results:
            if doc.metadata.get("access_role", "public") not in allowed_roles:
                continue
            key = (str(doc.metadata.get("doc_id", "")), int(doc.metadata.get("chunk_id", -1)))
            candidates[key] = {
                "doc": doc,
                "semantic": max(0.0, min(1.0, float(sem_score))),
                "bm25": 0.0,
            }

        # Add BM25 scores and top BM25 candidates
        bm25_indexed = sorted(enumerate(bm25_raw), key=lambda x: x[1], reverse=True)[:fetch_k]
        for idx, raw_score in bm25_indexed:
            doc = self._documents[idx]
            if doc.metadata.get("access_role", "public") not in allowed_roles:
                continue
            key = (str(doc.metadata.get("doc_id", "")), int(doc.metadata.get("chunk_id", -1)))
            norm_bm25 = max(0.0, float(raw_score) / max_bm25)
            if key not in candidates:
                candidates[key] = {"doc": doc, "semantic": 0.0, "bm25": norm_bm25}
            else:
                candidates[key]["bm25"] = norm_bm25

        scored: list[RetrievedChunk] = []
        for item in candidates.values():
            hybrid = w_semantic * item["semantic"] + w_bm25 * item["bm25"]
            scored.append(
                RetrievedChunk(
                    document=item["doc"],
                    semantic_score=round(item["semantic"], 4),
                    bm25_score=round(item["bm25"], 4),
                    hybrid_score=round(hybrid, 4),
                )
            )

        scored.sort(key=lambda c: c.hybrid_score, reverse=True)
        return scored[:top_k]

    @property
    def total_chunks(self) -> int:
        return len(self._documents)

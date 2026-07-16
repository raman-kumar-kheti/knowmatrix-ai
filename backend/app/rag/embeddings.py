"""Lazy-loaded CPU-compatible Hugging Face embedding model."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance running on CPU."""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

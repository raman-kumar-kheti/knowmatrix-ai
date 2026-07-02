"""Centralized configuration for Enterprise Knowledge Assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "Enterprise Knowledge Assistant"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api"
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Storage paths
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "app" / "data"
    UPLOAD_DIR: Path = DATA_DIR / "documents"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_store"
    DATABASE_URL: str = ""

    # Chunking & retrieval
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 4
    HYBRID_ALPHA: float = 0.6  # Weight for semantic score vs BM25 (1 - alpha)
    MIN_RELEVANCE_SCORE: float = 0.25
    MAX_UPLOAD_MB: int = 25

    # Embeddings & Reranker (CPU-compatible local models)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_CROSS_ENCODER_RERANK: bool = False  # Optional heavier rerank; lightweight lexical+cosine rerank always runs

    # Configurable LLM provider: "extractive" | "huggingface" | "ollama" | "openai_compatible"
    LLM_PROVIDER: str = "huggingface"
    HF_LLM_MODEL: str = "google/flan-t5-base"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OPENAI_API_BASE: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    def model_post_init(self, __context: Any) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        if not self.DATABASE_URL:
            db_path = (self.DATA_DIR / "enterprise_rag.db").resolve()
            self.DATABASE_URL = f"sqlite:///{db_path}"


settings = Settings()

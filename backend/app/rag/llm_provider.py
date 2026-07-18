"""Configurable, lazy-loaded LLM provider supporting local CPU and API backends."""

from __future__ import annotations

import re
from collections.abc import Iterator
from functools import lru_cache

import httpx

from app.core.config import settings


def estimate_tokens(text: str) -> int:
    """Approximate token count for telemetry without requiring a specific tokenizer."""
    return max(1, len(re.findall(r"\S+", text))) if text.strip() else 0


@lru_cache(maxsize=1)
def _get_hf_pipeline():
    from transformers import pipeline

    return pipeline(
        "text2text-generation",
        model=settings.HF_LLM_MODEL,
        device=-1,
        max_new_tokens=256,
    )


def _extractive_fallback(question: str, context: str) -> str:
    """Deterministic CPU-only grounded summary when no generative LLM is desired."""
    q_words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if len(s.strip()) > 15]
    if not sentences:
        return "I could not find enough supporting information in the indexed documents."

    ranked = sorted(
        sentences,
        key=lambda s: len(q_words & set(re.findall(r"\w+", s.lower()))),
        reverse=True,
    )
    selected = ranked[:3]
    return " ".join(selected)


def generate_answer(
    question: str,
    context: str,
    chat_history: list[tuple[str, str]] | None = None,
    provider: str | None = None,
) -> str:
    """Generate a grounded answer using the configured LLM provider."""
    selected_provider = (provider or settings.LLM_PROVIDER).lower()
    history_block = ""
    if chat_history:
        turns = [f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]]
        history_block = "Conversation History:\n" + "\n".join(turns) + "\n\n"

    # Keep prompt concise so local 512-token models (like flan-t5-base) do not truncate context.
    trimmed_context = context[:1600]
    prompt = (
        "Answer the question using ONLY the provided context. "
        "If the context does not contain the answer, say 'Insufficient evidence in the provided documents.'\n\n"
        f"{history_block}"
        f"Context:\n{trimmed_context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    if selected_provider == "extractive":
        return _extractive_fallback(question, trimmed_context)

    if selected_provider == "huggingface":
        llm = _get_hf_pipeline()
        out = llm(prompt, truncation=True)
        return str(out[0]["generated_text"]).strip()

    if selected_provider == "ollama":
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        return str(resp.json().get("response", "")).strip()

    if selected_provider == "openai_compatible":
        if not settings.OPENAI_API_BASE or not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_BASE and OPENAI_API_KEY must be set for openai_compatible provider.")
        resp = httpx.post(
            f"{settings.OPENAI_API_BASE.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={
                "model": settings.OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()

    raise ValueError(f"Unsupported LLM_PROVIDER: {selected_provider}")


def stream_answer_tokens(answer: str) -> Iterator[str]:
    """Yield word-level chunks for Server-Sent Events streaming."""
    words = answer.split(" ")
    for idx, word in enumerate(words):
        yield word + (" " if idx < len(words) - 1 else "")

# KnowMatrix AI (`enterprise-rag-assistant`)

> **Enterprise document intelligence and RAG** platform with hybrid semantic + BM25 search, page-level source citations, access control, streaming responses, and RAG evaluation.

## 1. Problem Statement
Organizations store critical knowledge across thousands of PDFs and internal documents. Standard keyword search misses semantic meaning, while naive LLM chatbots hallucinate without verifiable sources. KnowMatrix AI solves this with grounded, citation-backed retrieval, access control, and automated faithfulness/hallucination evaluation on CPU-friendly hardware.

## 2. Project Overview
This project transforms [upppy/rag-ai-assistant](https://github.com/upppy/rag-ai-assistant) into an enterprise-grade RAG platform.

### Upstream vs. Original Contributions
- **Inherited from `rag-ai-assistant`:** FastAPI + SQLite backend layout, React + Vite + Tailwind frontend structure, and `all-MiniLM-L6-v2` + FAISS foundation.
- **Original Additions:**
  - Page-aware PDF/TXT/MD extraction via **PyMuPDF (`fitz`)** with configurable chunk size and overlap.
  - **Multi-document FAISS + BM25 hybrid retrieval** with document deletion and re-indexing.
  - **Reranking** (lexical coverage + optional CrossEncoder) and **document-level access control** (`public`, `internal`, `admin`).
  - **Configurable LLM provider** (`extractive`, `huggingface`, `ollama`, `openai_compatible`) with lazy loading and **SSE streaming** (`/api/chat/stream`).
  - **RAGAS-inspired evaluation & hallucination detection** (context precision, faithfulness, answer relevance, insufficient-evidence refusal).
  - **Token usage & latency telemetry** (`/api/metrics`) and an enhanced React UI.

## 3. Features
- Multi-document PDF/TXT/MD upload, listing, deletion, and re-indexing
- Page-level source citations (`source`, `page`, `score`, `snippet`)
- Hybrid semantic (FAISS) + keyword (BM25) search and reranking
- Document-level role-based access control (`public`, `internal`, `admin`)
- Conversational memory per `session_id`
- Server-Sent Events (SSE) streaming responses
- RAG evaluation metrics, hallucination risk scoring, and insufficient-evidence detection
- Configurable LLM providers (`extractive`, `huggingface`, `ollama`, `openai_compatible`)

## 4. Technology Stack
- **Backend:** Python 3.12, FastAPI, Pydantic, SQLAlchemy (SQLite), PyMuPDF, LangChain, FAISS-CPU, `rank-bm25`, `sentence-transformers`, Transformers, PyTorch (CPU)
- **Frontend:** React 19, Vite, Tailwind CSS, Axios
- **Testing & DevOps:** Pytest, HTTPX, Docker, Docker Compose

## 5. Architecture & Folder Structure
```text
enterprise-rag-assistant/
├── .env.example
├── docker-compose.yml
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── api/          # health, upload/documents, chat/stream, history/metrics
│       ├── core/         # centralized Pydantic Settings
│       ├── db/           # SQLAlchemy engine, DocumentRecord, ChatHistory, QueryLog
│       ├── models/       # Pydantic request/response schemas
│       ├── rag/          # loader, chunker, embeddings, vector_store, reranker, llm_provider, evaluator, qa_chain
│       └── tests/        # pytest integration suite
└── frontend/
    └── src/              # React + Tailwind UI (Upload, Document Manager, ChatBox, Message)
```

## 6. Installation & Environment Configuration
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
cp .env.example .env
```

## 7. How to Run Locally
```bash
# Backend (from enterprise-rag-assistant/backend)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (from enterprise-rag-assistant/frontend)
npm install
npm run dev
```

## 8. How to Run Tests
```bash
cd backend
pytest app/tests/ -v
```

## 9. API Documentation
Interactive OpenAPI docs at `http://localhost:8000/docs`:
- `GET /api/health` — Service health
- `POST /api/upload` — Upload and index PDF/TXT/MD with `access_role`, `chunk_size`, `chunk_overlap`
- `GET /api/documents` — List indexed documents
- `DELETE /api/documents/{doc_id}` — Delete document and rebuild index
- `POST /api/documents/{doc_id}/reindex` — Re-chunk and re-index a document
- `POST /api/chat` — Grounded Q&A with citations, evaluation metrics, and latency/token telemetry
- `POST /api/chat/stream` — SSE streaming Q&A
- `GET /api/history` — Session chat history
- `GET /api/metrics` — Aggregate latency, token, and evaluation metrics

## 10. Docker Instructions
```bash
docker compose up --build
```

## 11. Limitations & Future Improvements
- **Limitations:** Uses local SQLite and in-memory BM25 reconstruction on startup; `huggingface` (`flan-t5-base`) is limited to short contexts compared to API LLMs.
- **Future Improvements:** Add Qdrant/pgvector for distributed vector storage, JWT authentication for role enforcement, and OCR for scanned PDFs.

## 12. Original Repository Attribution & License
- **Original Upstream Repository:** [upppy/rag-ai-assistant](https://github.com/upppy/rag-ai-assistant) (`© RAG AI Assistant Project` in upstream README; no `LICENSE` file was present in upstream).
- All original upstream code attribution is preserved.

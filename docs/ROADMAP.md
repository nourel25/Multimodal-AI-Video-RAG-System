# Roadmap

Tracks the v2 overhaul agreed on before implementation. Each phase lands on its own branch / PR.

## Phase 0 — Repo cleanup (`chore/repo-cleanup`)
- [x] Stop tracking generated transcripts, root `.gitignore`
- [x] `pyproject.toml` with ruff + pytest config

## Phase 1 — Foundation (`feat/foundation`)
- [x] Fix bugs: `system` role typo, `do_reset` per page, duplicate re-processing, missing-user crashes,
      hard-coded transcription language, size-based validation, directory creation, Qdrant/template bugs
- [x] Models and clients loaded once (service container + FastAPI dependencies)
- [x] Provider abstraction: LLM (Ollama / Groq / Gemini) with retry + fallback, transcription
      (local faster-whisper / Groq Whisper), local multilingual embeddings
- [x] Background processing with a job queue (ARQ + Redis, or in-process) and job status endpoint
- [x] Data model: per-video status, YouTube video id de-duplication, timestamps on every chunk
- [x] RESTful `/api/v1` routes, response models, error handlers, optional API key auth, logging
- [x] Docker compose for the full stack, isolated by project name and non-default ports
- [x] Tests + GitHub Actions CI

## Phase 2 — Multimodal retrieval (`feat/multimodal-retrieval`)
- [x] Timestamped citations with YouTube deep links
- [x] Visual pipeline: scene-based keyframes → vision-LLM captions + on-screen text
- [x] Hybrid retrieval (dense + BM25-style sparse, RRF fusion) + cross-encoder reranking
- [x] Streaming answers over SSE + conversation history
- [x] Retrieval evaluation harness (Recall@k, MRR) comparing retrieval configurations

## Phase 3 — Presentation (`feat/ui-docs`)
- [x] Streamlit UI: library, job progress, chat with timestamp jumps
- [x] README: architecture, setup, evaluation results

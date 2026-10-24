from fastapi import APIRouter, Depends

from core.container import ServiceContainer

from .dependencies import get_container

base_router = APIRouter(prefix="/api/v1", tags=["health"])


@base_router.get("/health")
async def health(container: ServiceContainer = Depends(get_container)):
    s = container.settings
    checks = {}
    try:
        await container.db_client.command("ping")
        checks["mongodb"] = "ok"
    except Exception as exc:
        checks["mongodb"] = f"error: {exc}"
    try:
        await container.vectordb.list_all_collections()
        checks["vectordb"] = "ok"
    except Exception as exc:
        checks["vectordb"] = f"error: {exc}"

    # Selected backends whose API key is missing: requests using them will return 503.
    needs_key = {"GROQ": s.GROQ_API_KEY, "GEMINI": s.GEMINI_API_KEY}
    selected = {
        "TRANSCRIPTION_BACKEND": s.TRANSCRIPTION_BACKEND,
        "GENERATION_BACKEND": s.GENERATION_BACKEND,
        "GENERATION_FALLBACK_BACKEND": s.GENERATION_FALLBACK_BACKEND,
        "VISION_BACKEND": s.VISION_BACKEND,
    }
    missing_keys = sorted({
        f"{backend.upper()}_API_KEY" for backend in selected.values()
        if backend.upper() in needs_key and not needs_key[backend.upper()]
    })

    return {
        "app_name": s.APP_NAME,
        "app_version": s.APP_VERSION,
        "status": "ok" if all(v == "ok" for v in checks.values()) else "degraded",
        "checks": checks,
        "missing_api_keys": missing_keys,
        "config": {
            "job_queue": s.JOB_QUEUE_BACKEND,
            "transcription": s.TRANSCRIPTION_BACKEND,
            "generation": s.GENERATION_BACKEND,
            "generation_fallback": s.GENERATION_FALLBACK_BACKEND or None,
            "vision": s.VISION_BACKEND or None,
            "embedding_model": s.EMBEDDING_MODEL,
            "retrieval_mode": s.RETRIEVAL_MODE,
            "reranker": s.RERANKER_MODEL if s.RERANK_ENABLED else None,
        },
    }

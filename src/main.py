import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.container import ServiceContainer
from helpers.config import get_settings
from helpers.logging_config import setup_logging
from routes import base, jobs, nlp, videos
from routes.errors import register_exception_handlers

logger = logging.getLogger(__name__)


def create_app(container: ServiceContainer | None = None) -> FastAPI:
    """Build the app. Tests pass a ready container; production builds one at startup."""
    settings = container.settings if container else get_settings()
    setup_logging(settings.LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = container is None
        app.state.container = container or await ServiceContainer.create(settings)
        if owned and settings.PRELOAD_MODELS:
            await asyncio.to_thread(app.state.container.preload)
        logger.info("%s v%s started", settings.APP_NAME, settings.APP_VERSION)
        yield
        if owned:
            await app.state.container.close()

    app = FastAPI(
        title="Multimodal Video RAG",
        version=settings.APP_VERSION,
        description="Ask questions about YouTube videos: transcripts + on-screen content, "
                    "hybrid retrieval and timestamped citations.",
        lifespan=lifespan,
    )
    if container is not None:
        app.state.container = container

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    register_exception_handlers(app)
    for router in (base.base_router, videos.videos_router, jobs.jobs_router, nlp.nlp_router):
        app.include_router(router)
    return app


app = create_app()

"""Composition root: builds every client/model once and shares it across requests and jobs.

Heavy components (embedding model, reranker, Whisper, LLM clients) are created lazily on
first use, so the API starts fast and the API process never loads Whisper when jobs run in
the separate ARQ worker. Tests replace any component by plain attribute assignment.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from controllers.NLPController import NLPController
from controllers.VideoController import VideoController
from helpers.config import Settings
from models.database import ensure_indexes
from stores.embedding.SparseEncoder import BM25SparseEncoder
from stores.jobs.JobQueue import ArqJobQueue, InlineJobQueue, JobQueueInterface
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.templates.template_parser import TemplateParser
from stores.transcription.TranscriptionProviderFactory import TranscriptionProviderFactory
from stores.vectordb.VectorDBInterface import VectorDBInterface
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

logger = logging.getLogger(__name__)

_UNSET = object()


class lazy_component:
    """Thread-safe `cached_property`: model loading may be triggered from worker threads."""

    def __init__(self, factory: Callable[[Any], Any]):
        self.factory = factory
        self.name = factory.__name__
        self.lock = threading.Lock()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = instance.__dict__.get(self.name, _UNSET)
        if value is _UNSET:
            with self.lock:
                value = instance.__dict__.get(self.name, _UNSET)
                if value is _UNSET:
                    value = self.factory(instance)
                    instance.__dict__[self.name] = value
        return value

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


class ServiceContainer:
    def __init__(self, settings: Settings, *, db_client, vectordb: VectorDBInterface,
                 mongo_client=None):
        self.settings = settings
        self.mongo_client = mongo_client
        self.db_client = db_client
        self.vectordb = vectordb
        self.job_queue: JobQueueInterface | None = None

    @classmethod
    async def create(cls, settings: Settings, *, with_job_queue: bool = True
                     ) -> "ServiceContainer":
        mongo_client = AsyncIOMotorClient(settings.MONGODB_URL)
        db_client = mongo_client[settings.MONGODB_DATABASE]
        await ensure_indexes(db_client)

        vectordb = VectorDBProviderFactory(settings).create(settings.VECTOR_DB_BACKEND)
        await vectordb.connect()

        container = cls(settings, db_client=db_client, vectordb=vectordb,
                        mongo_client=mongo_client)
        if with_job_queue:
            container.job_queue = container.build_job_queue()
        return container

    def build_job_queue(self) -> JobQueueInterface:
        if self.settings.JOB_QUEUE_BACKEND.lower() == "arq":
            if not self.settings.QDRANT_URL:
                logger.warning("JOB_QUEUE_BACKEND=arq without QDRANT_URL: the embedded Qdrant "
                               "database cannot be shared between the API and the worker.")
            return ArqJobQueue(self.settings.REDIS_URL)

        from controllers.ProcessController import ProcessController

        return InlineJobQueue(runner=lambda job_id: ProcessController(self).run_job(job_id))

    async def close(self) -> None:
        if self.job_queue is not None:
            await self.job_queue.close()
        await self.vectordb.disconnect()
        if self.mongo_client is not None:
            self.mongo_client.close()

    def preload(self) -> None:
        """Load query-time models eagerly so the first request isn't slow."""
        _ = self.embedder
        if self.settings.RERANK_ENABLED:
            _ = self.reranker

    # ---------- lazily built components ----------

    @lazy_component
    def template_parser(self) -> TemplateParser:
        return TemplateParser(language=self.settings.PRIMARY_LANG,
                              default_language=self.settings.DEFAULT_LANG)

    @lazy_component
    def sparse_encoder(self) -> BM25SparseEncoder:
        return BM25SparseEncoder()

    @lazy_component
    def embedder(self):
        from stores.embedding.SentenceTransformerEmbedding import SentenceTransformerEmbedding

        s = self.settings
        return SentenceTransformerEmbedding(
            s.EMBEDDING_MODEL, device=s.EMBEDDING_DEVICE, batch_size=s.EMBEDDING_BATCH_SIZE,
            query_prefix=s.EMBEDDING_QUERY_PREFIX, document_prefix=s.EMBEDDING_DOCUMENT_PREFIX,
        )

    @lazy_component
    def reranker(self):
        if not self.settings.RERANK_ENABLED or not self.settings.RERANKER_MODEL:
            return None
        from stores.embedding.SentenceTransformerEmbedding import CrossEncoderReranker

        return CrossEncoderReranker(self.settings.RERANKER_MODEL,
                                    device=self.settings.EMBEDDING_DEVICE)

    @lazy_component
    def generation_client(self):
        return LLMProviderFactory(self.settings).create_generation_client()

    @lazy_component
    def vision_client(self):
        return LLMProviderFactory(self.settings).create_vision_client()

    @lazy_component
    def transcriber(self):
        return TranscriptionProviderFactory(self.settings).create_with_fallback()

    @lazy_component
    def video_controller(self) -> VideoController:
        return VideoController(self.settings)

    # ---------- controller builders ----------

    def nlp_controller(self, *, with_generation: bool = False,
                       with_reranker: bool = True) -> NLPController:
        return NLPController(
            vectordb_client=self.vectordb,
            embedder=self.embedder,
            sparse_encoder=self.sparse_encoder,
            template_parser=self.template_parser,
            settings=self.settings,
            generation_client=self.generation_client if with_generation else None,
            reranker=self.reranker if with_reranker else None,
        )

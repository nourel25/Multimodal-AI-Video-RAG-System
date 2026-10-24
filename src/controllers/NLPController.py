import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from starlette.concurrency import iterate_in_threadpool

from helpers.citations import StreamingCitationNormalizer, normalize_citations
from helpers.config import Settings
from helpers.text import detect_language
from helpers.youtube import format_timestamp
from models.db_schemas import Chunk, RetrievedDocument, User, Video
from models.enums import Modality
from stores.embedding.EmbeddingInterface import EmbeddingInterface, RerankerInterface
from stores.embedding.SparseEncoder import BM25SparseEncoder
from stores.llm import ChatMessage, ChatRole, LLMInterface
from stores.templates.template_parser import TemplateParser
from stores.vectordb.VectorDBEnums import SearchMode
from stores.vectordb.VectorDBInterface import VectorDBInterface, VectorRecord

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 10


@dataclass
class AnswerResult:
    answer: str
    sources: list[RetrievedDocument]
    language: str


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id, so re-indexing a chunk overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{chunk_id}"))


class NLPController:
    def __init__(self, *, vectordb_client: VectorDBInterface, embedder: EmbeddingInterface,
                 sparse_encoder: BM25SparseEncoder, template_parser: TemplateParser,
                 settings: Settings, generation_client: LLMInterface | None = None,
                 reranker: RerankerInterface | None = None):
        self.vectordb_client = vectordb_client
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.template_parser = template_parser
        self.generation_client = generation_client
        self.reranker = reranker
        self.settings = settings

    @staticmethod
    def create_collection_name(user_id: str) -> str:
        return f"collection_{user_id}"

    # ---------- indexing ----------

    async def reset_collection(self, user: User) -> None:
        await self.vectordb_client.delete_collection(self.create_collection_name(user.user_id))

    async def build_records(self, video: Video, chunks: list[Chunk]) -> list[VectorRecord]:
        """Embed chunks (the slow, failure-prone step) without writing anything yet."""
        if not chunks:
            return []
        texts = [chunk.chunk_text for chunk in chunks]
        dense = await asyncio.to_thread(self.embedder.embed_documents, texts)
        sparse = [self.sparse_encoder.encode_document(text) for text in texts]
        return [
            VectorRecord(
                id=point_id(str(chunk.id)),
                dense=dense[i],
                sparse=sparse[i],
                payload={
                    "chunk_id": str(chunk.id),
                    "video_id": str(video.id),
                    "youtube_video_id": video.youtube_video_id,
                    "video_title": video.title,
                    "text": chunk.chunk_text,
                    "start": chunk.start,
                    "end": chunk.end,
                    "modality": chunk.modality.value if isinstance(chunk.modality, Modality)
                    else chunk.modality,
                    "order": chunk.chunk_order,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    async def upsert_records(self, user: User, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        collection_name = self.create_collection_name(user.user_id)
        await self.vectordb_client.create_collection(collection_name, self.embedder.size)
        return await self.vectordb_client.upsert_records(collection_name, records)

    async def index_chunks(self, user: User, video: Video, chunks: list[Chunk]) -> int:
        return await self.upsert_records(user, await self.build_records(video, chunks))

    async def delete_video_vectors(self, user: User, video_id: str) -> None:
        await self.vectordb_client.delete_by_video(
            self.create_collection_name(user.user_id), video_id)

    # ---------- retrieval ----------

    async def search(self, user: User, query: str, limit: int = 5,
                     mode: SearchMode | None = None, rerank: bool | None = None,
                     video_ids: list[str] | None = None) -> list[RetrievedDocument]:
        mode = SearchMode(mode or self.settings.RETRIEVAL_MODE)
        rerank = self.settings.RERANK_ENABLED if rerank is None else rerank
        rerank = rerank and self.reranker is not None
        candidates = max(limit, self.settings.RETRIEVAL_CANDIDATES) if rerank else limit

        dense_vector = None
        if mode != SearchMode.SPARSE:
            dense_vector = await asyncio.to_thread(self.embedder.embed_query, query)
        sparse_vector = self.sparse_encoder.encode_query(query)

        records = await self.vectordb_client.search(
            self.create_collection_name(user.user_id),
            dense_vector=dense_vector, sparse_vector=sparse_vector,
            limit=candidates, mode=mode, video_ids=video_ids,
        )
        documents = [
            RetrievedDocument(
                chunk_id=r.payload.get("chunk_id", r.id),
                video_id=r.payload.get("video_id", ""),
                youtube_video_id=r.payload.get("youtube_video_id", ""),
                video_title=r.payload.get("video_title", ""),
                text=r.payload.get("text", ""),
                start=r.payload.get("start", 0),
                end=r.payload.get("end", 0),
                modality=r.payload.get("modality", Modality.TRANSCRIPT.value),
                score=r.score,
            )
            for r in records
        ]

        if rerank and documents:
            scores = await asyncio.to_thread(self.reranker.score, query,
                                             [doc.text for doc in documents])
            for doc, score in zip(documents, scores, strict=True):
                doc.score = score
            documents.sort(key=lambda doc: doc.score, reverse=True)

        return documents[:limit]

    # ---------- generation ----------

    def build_messages(self, query: str, documents: list[RetrievedDocument], language: str,
                       history: list[ChatMessage] | None = None) -> list[ChatMessage]:
        tp = self.template_parser
        sources = "\n\n".join(
            tp.get("rag", "document_prompt", {
                "doc_num": i + 1,
                "modality": tp.get("rag", f"modality_{doc.modality.value}", language=language),
                "video_title": doc.video_title or doc.youtube_video_id,
                "timestamp": format_timestamp(doc.start),
                "chunk_text": doc.text,
            }, language=language)
            for i, doc in enumerate(documents)
        )
        footer = tp.get("rag", "footer_prompt", {"query": query}, language=language)

        messages: list[ChatMessage] = [
            {"role": ChatRole.SYSTEM.value,
             "content": tp.get("rag", "system_prompt", language=language)},
        ]
        for turn in (history or [])[-MAX_HISTORY_MESSAGES:]:
            if turn["role"] in (ChatRole.USER.value, ChatRole.ASSISTANT.value):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": ChatRole.USER.value, "content": f"{sources}\n\n{footer}"})
        return messages

    def _language(self, query: str, language: str | None) -> str:
        return self.template_parser.resolve_language(language or detect_language(query))

    async def answer(self, user: User, query: str, *, limit: int = 5,
                     language: str | None = None, history: list[ChatMessage] | None = None,
                     **search_kwargs) -> AnswerResult | None:
        documents = await self.search(user, query, limit=limit, **search_kwargs)
        if not documents:
            return None
        language = self._language(query, language)
        messages = self.build_messages(query, documents, language, history)
        answer = await asyncio.to_thread(self.generation_client.generate, messages)
        return AnswerResult(answer=normalize_citations(answer).strip(), sources=documents,
                            language=language)

    async def stream_answer(self, user: User, query: str, *, limit: int = 5,
                            language: str | None = None,
                            history: list[ChatMessage] | None = None,
                            **search_kwargs) -> AsyncIterator[tuple[str, object]]:
        """Yield ("sources", [...]) first, then ("token", str) events as the LLM generates."""
        documents = await self.search(user, query, limit=limit, **search_kwargs)
        yield "sources", documents
        if not documents:
            return
        language = self._language(query, language)
        messages = self.build_messages(query, documents, language, history)
        citations = StreamingCitationNormalizer()
        async for token in iterate_in_threadpool(self.generation_client.stream(messages)):
            if text := citations.feed(token):
                yield "token", text
        if text := citations.flush():
            yield "token", text

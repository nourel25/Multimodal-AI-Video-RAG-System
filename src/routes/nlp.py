import asyncio
import json
import logging

from fastapi import APIRouter, Depends, status
from sse_starlette.sse import EventSourceResponse

from core.container import ServiceContainer
from models.ChunkModel import ChunkModel
from models.db_schemas import User
from models.enums import ResponseSignal, VideoStatus
from models.VideoModel import VideoModel
from stores.llm import LLMError

from .dependencies import get_container, get_user_or_404, verify_api_key
from .errors import AppError
from .schemas.nlp import (
    AnswerRequest,
    AnswerResponse,
    RebuildIndexResponse,
    SearchRequest,
    SearchResponse,
    to_sources,
)

logger = logging.getLogger(__name__)

nlp_router = APIRouter(prefix="/api/v1/users/{user_id}", tags=["rag"],
                       dependencies=[Depends(verify_api_key)])


def _search_kwargs(body: SearchRequest) -> dict:
    return {"mode": body.mode, "rerank": body.rerank, "video_ids": body.video_ids}


@nlp_router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, user: User = Depends(get_user_or_404),
                 container: ServiceContainer = Depends(get_container)):
    nlp = await asyncio.to_thread(container.nlp_controller)
    documents = await nlp.search(
        user, body.query, limit=body.limit, **_search_kwargs(body))
    return SearchResponse(signal=ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
                          results=to_sources(documents))


@nlp_router.post("/answer", response_model=AnswerResponse)
async def answer(body: AnswerRequest, user: User = Depends(get_user_or_404),
                 container: ServiceContainer = Depends(get_container)):
    nlp = await asyncio.to_thread(container.nlp_controller, with_generation=True)
    result = await nlp.answer(
        user, body.query, limit=body.limit, language=body.language,
        history=[turn.model_dump() for turn in body.history], **_search_kwargs(body),
    )
    if result is None:
        raise AppError(status.HTTP_404_NOT_FOUND, ResponseSignal.NO_RELEVANT_CONTEXT,
                       "No indexed content matched the question")
    return AnswerResponse(signal=ResponseSignal.RAG_ANSWER_SUCCESS.value, answer=result.answer,
                          language=result.language, sources=to_sources(result.sources))


@nlp_router.post("/answer/stream")
async def answer_stream(body: AnswerRequest, user: User = Depends(get_user_or_404),
                        container: ServiceContainer = Depends(get_container)):
    """Server-Sent Events: `sources` (JSON list), many `token` events, then `done` or `error`."""
    nlp = await asyncio.to_thread(container.nlp_controller, with_generation=True)

    async def events():
        try:
            async for kind, payload in nlp.stream_answer(
                user, body.query, limit=body.limit, language=body.language,
                history=[turn.model_dump() for turn in body.history], **_search_kwargs(body),
            ):
                if kind == "sources":
                    sources = to_sources(payload)
                    yield {"event": "sources",
                           "data": json.dumps([s.model_dump(mode="json") for s in sources],
                                              ensure_ascii=False)}
                    if not sources:
                        yield {"event": "error",
                               "data": ResponseSignal.NO_RELEVANT_CONTEXT.value}
                        return
                else:
                    # JSON-encode tokens so newlines inside them survive SSE framing.
                    yield {"event": "token", "data": json.dumps(payload, ensure_ascii=False)}
            yield {"event": "done", "data": ResponseSignal.RAG_ANSWER_SUCCESS.value}
        except LLMError as exc:
            logger.error("Streaming answer failed: %s", exc)
            yield {"event": "error", "data": ResponseSignal.PROVIDER_ERROR.value}
        except Exception:
            # Headers are already sent, so report the failure in-band instead of truncating.
            logger.exception("Streaming answer failed")
            yield {"event": "error", "data": ResponseSignal.INTERNAL_ERROR.value}

    return EventSourceResponse(events())


@nlp_router.post("/index/rebuild", response_model=RebuildIndexResponse)
async def rebuild_index(user: User = Depends(get_user_or_404),
                        container: ServiceContainer = Depends(get_container)):
    """Re-embed every stored chunk, e.g. after changing EMBEDDING_MODEL. No re-transcription."""
    videos = {v.id: v for v in await VideoModel(container.db_client).list_user_videos(user.id)}
    if any(v.status == VideoStatus.PROCESSING for v in videos.values()):
        raise AppError(status.HTTP_409_CONFLICT, ResponseSignal.VIDEO_IS_PROCESSING,
                       "Wait for running processing jobs to finish before rebuilding the index")
    nlp = await asyncio.to_thread(container.nlp_controller, with_reranker=False)
    await nlp.reset_collection(user)

    indexed = 0
    async for page in ChunkModel(container.db_client).iter_user_chunks(user.id):
        by_video: dict = {}
        for chunk in page:
            by_video.setdefault(chunk.chunk_video_id, []).append(chunk)
        for video_id, chunks in by_video.items():
            if video_id in videos:
                indexed += await nlp.index_chunks(user, videos[video_id], chunks)
    return RebuildIndexResponse(signal=ResponseSignal.INDEX_REBUILT.value,
                                indexed_chunks=indexed)

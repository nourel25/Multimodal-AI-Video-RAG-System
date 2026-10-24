import asyncio
import logging

from fastapi import APIRouter, Depends, status
from pymongo.errors import DuplicateKeyError

from controllers.JobController import JobController
from controllers.NLPController import NLPController
from core.container import ServiceContainer
from helpers.youtube import extract_video_id, watch_url
from models.ChunkModel import ChunkModel
from models.db_schemas import ProcessOptions, User, Video
from models.enums import ResponseSignal, VideoStatus
from models.UserModel import UserModel
from models.VideoModel import VideoModel

from .dependencies import (
    UserIdPath,
    get_container,
    get_user_or_404,
    parse_object_id,
    verify_api_key,
)
from .errors import AppError
from .schemas.jobs import JobOut
from .schemas.videos import (
    AddVideosRequest,
    AddVideosResponse,
    DeleteVideoResponse,
    ListVideosResponse,
    SkippedUrl,
    VideoOut,
    VideoResponse,
)

logger = logging.getLogger(__name__)

videos_router = APIRouter(prefix="/api/v1/users/{user_id}/videos", tags=["videos"],
                          dependencies=[Depends(verify_api_key)])

# yt-dlp metadata calls are network bound; a few in parallel speeds up bulk adds.
METADATA_CONCURRENCY = 4


@videos_router.post("", status_code=status.HTTP_201_CREATED, response_model=AddVideosResponse)
async def add_videos(body: AddVideosRequest, user_id: str = UserIdPath,
                     container: ServiceContainer = Depends(get_container)):
    user = await UserModel(container.db_client).get_or_create_user(user_id)
    video_model = VideoModel(container.db_client)
    downloader = container.video_controller

    skipped: list[SkippedUrl] = []
    candidates: dict[str, str] = {}  # youtube_video_id -> original url
    for url in map(str, body.youtube_urls):
        youtube_id = extract_video_id(url)
        if youtube_id is None:
            skipped.append(SkippedUrl(url=url, signal=ResponseSignal.VIDEO_INVALID_URL.value,
                                      detail="Not a YouTube video URL"))
        elif youtube_id in candidates or await video_model.find_by_youtube_id(user.id,
                                                                              youtube_id):
            skipped.append(SkippedUrl(url=url, signal=ResponseSignal.VIDEO_ALREADY_EXISTS.value))
        else:
            candidates[youtube_id] = url

    semaphore = asyncio.Semaphore(METADATA_CONCURRENCY)

    async def fetch(youtube_id: str):
        async with semaphore:
            try:
                return await asyncio.to_thread(downloader.fetch_metadata, youtube_id)
            except Exception as exc:
                logger.warning("Metadata fetch for %s failed: %s", youtube_id, exc)
                return exc

    metadata_list = await asyncio.gather(*(fetch(yid) for yid in candidates))

    added: list[Video] = []
    for (youtube_id, url), metadata in zip(candidates.items(), metadata_list, strict=True):
        if isinstance(metadata, Exception):
            skipped.append(SkippedUrl(url=url, signal=ResponseSignal.VIDEO_METADATA_FAILED.value,
                                      detail=str(metadata)))
            continue
        valid, signal = downloader.validate_metadata(metadata)
        if not valid:
            skipped.append(SkippedUrl(url=url, signal=signal))
            continue
        try:
            video = await video_model.insert_video(Video(
                video_user_id=user.id,
                youtube_url=watch_url(youtube_id),
                youtube_video_id=youtube_id,
                title=metadata.title,
                channel=metadata.channel,
                duration_seconds=metadata.duration_seconds,
                thumbnail_url=metadata.thumbnail_url,
            ))
        except DuplicateKeyError:  # added concurrently by another request
            skipped.append(SkippedUrl(url=url, signal=ResponseSignal.VIDEO_ALREADY_EXISTS.value))
            continue
        added.append(video)

    job = None
    if body.auto_process and added:
        s = container.settings
        job = await JobController(container).queue_processing(user, added, ProcessOptions(
            chunk_max_chars=s.CHUNK_MAX_CHARS, chunk_overlap_segments=s.CHUNK_OVERLAP_SEGMENTS,
            include_visual=body.include_visual,
        ))

    return AddVideosResponse(
        signal=ResponseSignal.VIDEO_ADDED.value if added else
        ResponseSignal.NO_VIDEOS_TO_PROCESS.value,
        user_id=user.user_id,
        added=[VideoOut.from_video(v) for v in added],
        skipped=skipped,
        job=JobOut.from_job(job) if job else None,
    )


@videos_router.get("", response_model=ListVideosResponse)
async def list_videos(user: User = Depends(get_user_or_404),
                      container: ServiceContainer = Depends(get_container)):
    videos = await VideoModel(container.db_client).list_user_videos(user.id)
    return ListVideosResponse(signal=ResponseSignal.VIDEOS_LISTED.value,
                              videos=[VideoOut.from_video(v) for v in videos])


async def _get_video_or_404(container, user: User, video_id: str) -> Video:
    object_id = parse_object_id(video_id, ResponseSignal.VIDEO_NOT_FOUND)
    video = await VideoModel(container.db_client).get_user_video(user.id, object_id)
    if video is None:
        raise AppError(status.HTTP_404_NOT_FOUND, ResponseSignal.VIDEO_NOT_FOUND)
    return video


@videos_router.get("/{video_id}", response_model=VideoResponse)
async def get_video(video_id: str, user: User = Depends(get_user_or_404),
                    container: ServiceContainer = Depends(get_container)):
    video = await _get_video_or_404(container, user, video_id)
    return VideoResponse(signal=ResponseSignal.VIDEOS_LISTED.value,
                         video=VideoOut.from_video(video))


@videos_router.delete("/{video_id}", response_model=DeleteVideoResponse)
async def delete_video(video_id: str, user: User = Depends(get_user_or_404),
                       container: ServiceContainer = Depends(get_container)):
    video = await _get_video_or_404(container, user, video_id)
    if video.status == VideoStatus.PROCESSING:
        raise AppError(status.HTTP_409_CONFLICT, ResponseSignal.VIDEO_IS_PROCESSING,
                       "Wait for the processing job to finish before deleting this video")
    deleted = await ChunkModel(container.db_client).delete_chunks_by_video_id(video.id)
    await container.vectordb.delete_by_video(NLPController.create_collection_name(user.user_id),
                                             str(video.id))
    await VideoModel(container.db_client).delete_video(video.id)
    return DeleteVideoResponse(signal=ResponseSignal.VIDEO_DELETED.value, video_id=video_id,
                               deleted_chunks=deleted)

from fastapi import APIRouter, Depends, Query, status

from controllers.JobController import JobController
from core.container import ServiceContainer
from models.db_schemas import Job, ProcessOptions, User
from models.enums import ResponseSignal, VideoStatus
from models.JobModel import JobModel
from models.VideoModel import VideoModel

from .dependencies import get_container, get_user_or_404, parse_object_id, verify_api_key
from .errors import AppError
from .schemas.jobs import JobOut, JobResponse, ListJobsResponse, ProcessRequest

jobs_router = APIRouter(prefix="/api/v1/users/{user_id}", tags=["jobs"],
                        dependencies=[Depends(verify_api_key)])


@jobs_router.post("/process", status_code=status.HTTP_202_ACCEPTED, response_model=JobResponse)
async def process_videos(body: ProcessRequest, user: User = Depends(get_user_or_404),
                         container: ServiceContainer = Depends(get_container)):
    """Queue a background job; poll `GET /jobs/{job_id}` for progress."""
    video_model = VideoModel(container.db_client)

    if body.video_ids:
        videos = []
        for raw_id in body.video_ids:
            video = await video_model.get_user_video(
                user.id, parse_object_id(raw_id, ResponseSignal.VIDEO_NOT_FOUND))
            if video is None:
                raise AppError(status.HTTP_404_NOT_FOUND, ResponseSignal.VIDEO_NOT_FOUND,
                               f"Unknown video {raw_id!r}")
            videos.append(video)
    else:
        statuses = [VideoStatus.PENDING, VideoStatus.FAILED]
        if body.reprocess:
            statuses.append(VideoStatus.INDEXED)
        videos = await video_model.list_user_videos(user.id, statuses)

    if not body.reprocess:
        # Skip finished videos and ones already queued or running. `reprocess=true` overrides
        # this, which also recovers videos left "processing" by a crashed worker.
        videos = [v for v in videos
                  if v.status not in (VideoStatus.INDEXED, VideoStatus.PROCESSING)]
    if not videos:
        raise AppError(status.HTTP_409_CONFLICT, ResponseSignal.NO_VIDEOS_TO_PROCESS,
                       "Nothing to process: videos are already indexed or in progress "
                       "(use reprocess=true to rebuild indexed videos)")

    s = container.settings
    options = ProcessOptions(
        chunk_max_chars=body.chunk_max_chars or s.CHUNK_MAX_CHARS,
        chunk_overlap_segments=(s.CHUNK_OVERLAP_SEGMENTS if body.chunk_overlap_segments is None
                                else body.chunk_overlap_segments),
        include_visual=body.include_visual,
        language=body.language,
    )
    job = await JobController(container).queue_processing(user, videos, options)
    return JobResponse(signal=ResponseSignal.PROCESSING_QUEUED.value, job=JobOut.from_job(job))


async def _get_job_or_404(container, user: User, job_id: str) -> Job:
    parse_object_id(job_id, ResponseSignal.JOB_NOT_FOUND)
    job = await JobModel(container.db_client).get_job(job_id)
    if job is None or job.job_user_id != user.id:
        raise AppError(status.HTTP_404_NOT_FOUND, ResponseSignal.JOB_NOT_FOUND)
    return job


@jobs_router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, user: User = Depends(get_user_or_404),
                  container: ServiceContainer = Depends(get_container)):
    job = await _get_job_or_404(container, user, job_id)
    return JobResponse(signal=ResponseSignal.JOB_FOUND.value, job=JobOut.from_job(job))


@jobs_router.get("/jobs", response_model=ListJobsResponse)
async def list_jobs(limit: int = Query(10, ge=1, le=50), user: User = Depends(get_user_or_404),
                    container: ServiceContainer = Depends(get_container)):
    jobs = await JobModel(container.db_client).list_user_jobs(user.id, limit=limit)
    return ListJobsResponse(signal=ResponseSignal.JOB_FOUND.value,
                            jobs=[JobOut.from_job(job) for job in jobs])

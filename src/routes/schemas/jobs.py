from datetime import datetime

from pydantic import BaseModel, Field

from models.db_schemas import Job, JobVideoResult, ProcessOptions
from models.enums import JobStage, JobStatus


class ProcessRequest(BaseModel):
    video_ids: list[str] | None = Field(
        None, description="Videos to process; defaults to every pending or failed video")
    reprocess: bool = Field(False, description="Also re-process already indexed videos")
    include_visual: bool = True
    chunk_max_chars: int | None = Field(None, ge=100, le=4000)
    chunk_overlap_segments: int | None = Field(None, ge=0, le=5)
    language: str | None = Field(None, description="Force a transcription language, e.g. 'ar'")


class JobOut(BaseModel):
    job_id: str
    user_id: str
    status: JobStatus
    stage: JobStage
    progress: float
    current_video_id: str | None
    video_ids: list[str]
    options: ProcessOptions
    results: list[JobVideoResult]
    error: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None

    @classmethod
    def from_job(cls, job: Job) -> "JobOut":
        data = job.model_dump(exclude={"id", "job_user_id", "video_ids"})
        return cls(job_id=str(job.id), video_ids=[str(v) for v in job.video_ids], **data)


class JobResponse(BaseModel):
    signal: str
    job: JobOut


class ListJobsResponse(BaseModel):
    signal: str
    jobs: list[JobOut]

import logging

from models.db_schemas import Job, ProcessOptions, User, Video
from models.db_schemas.base import utc_now
from models.enums import JobStage, JobStatus
from models.JobModel import JobModel
from models.VideoModel import VideoModel

logger = logging.getLogger(__name__)

INTERRUPTED = "Processing was interrupted (the server or worker restarted); process it again"


class JobController:
    def __init__(self, container):
        self.container = container
        self.job_model = JobModel(container.db_client)
        self.video_model = VideoModel(container.db_client)

    async def queue_processing(self, user: User, videos: list[Video],
                               options: ProcessOptions) -> Job | None:
        """Claim the videos and queue one job for them; None if all were already claimed."""
        claimed = await self.video_model.claim_for_processing([video.id for video in videos])
        if not claimed:
            return None

        job = await self.job_model.insert_job(Job(
            job_user_id=user.id, user_id=user.user_id, video_ids=claimed, options=options,
        ))
        try:
            await self.container.job_queue.enqueue_process(str(job.id))
        except Exception as exc:
            # Never leave videos marked "processing" for a job nobody will run.
            message = f"Could not queue the processing job: {exc}"
            await self.video_model.fail_processing(claimed, message)
            await self.job_model.update_job(job.id, status=JobStatus.FAILED, error=message,
                                            finished_at=utc_now())
            raise
        return job

    async def recover_interrupted(self, statuses: list[JobStatus]) -> int:
        """Fail jobs in `statuses` left over from a previous process, and their videos.

        Called at startup: by the API for in-process jobs (lost on restart) and by the
        worker for jobs it was running when it stopped.
        """
        jobs = await self.job_model.list_jobs_by_status(statuses)
        for job in jobs:
            await self.video_model.fail_processing(job.video_ids, INTERRUPTED)
            await self.job_model.update_job(job.id, status=JobStatus.FAILED,
                                            stage=JobStage.DONE, error=INTERRUPTED,
                                            current_video_id=None, finished_at=utc_now())
        if jobs:
            logger.warning("Marked %d interrupted job(s) as failed", len(jobs))
        return len(jobs)

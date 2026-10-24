from models.db_schemas import Job, ProcessOptions, User, Video
from models.enums import VideoStatus
from models.JobModel import JobModel
from models.VideoModel import VideoModel


class JobController:
    def __init__(self, container):
        self.container = container
        self.job_model = JobModel(container.db_client)
        self.video_model = VideoModel(container.db_client)

    async def queue_processing(self, user: User, videos: list[Video],
                               options: ProcessOptions) -> Job:
        job = await self.job_model.insert_job(Job(
            job_user_id=user.id, user_id=user.user_id,
            video_ids=[video.id for video in videos], options=options,
        ))
        for video in videos:
            # Mark immediately so a second request doesn't queue the same video again.
            await self.video_model.update_video(video.id, status=VideoStatus.PROCESSING,
                                                error=None)
        await self.container.job_queue.enqueue_process(str(job.id))
        return job

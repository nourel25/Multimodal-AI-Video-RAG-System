"""Runs a processing job: download -> transcribe -> chunk -> (frames -> captions) -> index.

Used both by the ARQ worker and by the in-process job queue.
"""

import asyncio
import contextlib
import logging
import os

from bson import ObjectId

from models.ChunkModel import ChunkModel
from models.db_schemas import Chunk, Job, JobVideoResult, User, Video
from models.db_schemas.base import utc_now
from models.enums import JobStage, JobStatus, Modality, VideoStatus
from models.JobModel import JobModel
from models.UserModel import UserModel
from models.VideoModel import VideoModel

from .AudioController import AudioController
from .ChunkController import ChunkController, TimedChunk
from .VisionController import VisionController

logger = logging.getLogger(__name__)

# Share of a single video's progress reached when each stage starts.
_STAGE_PROGRESS = {
    JobStage.DOWNLOADING: 0.05,
    JobStage.TRANSCRIBING: 0.15,
    JobStage.CHUNKING: 0.55,
    JobStage.EXTRACTING_FRAMES: 0.6,
    JobStage.CAPTIONING: 0.7,
    JobStage.INDEXING: 0.9,
}


class VideoProcessingError(RuntimeError):
    pass


class ProcessController:
    def __init__(self, container):
        self.container = container
        self.settings = container.settings
        db = container.db_client
        self.job_model = JobModel(db)
        self.video_model = VideoModel(db)
        self.chunk_model = ChunkModel(db)
        self.user_model = UserModel(db)

    async def run_job(self, job_id: str) -> None:
        job = await self.job_model.get_job(job_id)
        if job is None:
            logger.error("Job %s not found", job_id)
            return
        try:
            await self._run(job)
        except BaseException as exc:
            # Cancellation (shutdown, timeout) or an unexpected error: never leave the job
            # "running" or its videos "processing".
            message = f"Processing stopped: {exc!r}"
            logger.error("Job %s aborted: %s", job.id, message)
            await self.video_model.fail_processing(job.video_ids, message)
            await self.job_model.update_job(job.id, status=JobStatus.FAILED, error=message,
                                            stage=JobStage.DONE, current_video_id=None,
                                            finished_at=utc_now())
            raise

    async def _run(self, job: Job) -> None:
        user = await self.user_model.get_user(job.user_id)
        await self.job_model.update_job(job.id, status=JobStatus.RUNNING)
        failures = 0
        total = len(job.video_ids)

        for index, video_id in enumerate(job.video_ids):
            video = await self.video_model.get_video(video_id)
            if video is None or user is None:
                failures += 1
                continue

            async def set_stage(stage: JobStage, _index=index, _video=video):
                progress = (_index + _STAGE_PROGRESS.get(stage, 0)) / total * 100
                await self.job_model.update_job(job.id, stage=stage, progress=round(progress, 1),
                                                current_video_id=str(_video.id))

            result = await self.process_video(job, user, video, set_stage)
            if result.status == VideoStatus.FAILED:
                failures += 1
            await self.job_model.add_result(job.id, result)

        if failures == 0:
            status = JobStatus.COMPLETED
        elif failures == total:
            status = JobStatus.FAILED
        else:
            status = JobStatus.PARTIAL
        await self.job_model.update_job(job.id, status=status, stage=JobStage.DONE,
                                        progress=100.0, current_video_id=None,
                                        finished_at=utc_now())

    async def process_video(self, job: Job, user: User, video: Video, set_stage
                            ) -> JobVideoResult:
        result = JobVideoResult(video_id=str(video.id), youtube_url=video.youtube_url,
                                status=VideoStatus.PROCESSING)
        await self.video_model.update_video(video.id, status=VideoStatus.PROCESSING, error=None)
        media_files: list[str] = []
        try:
            transcript_chunks, language, transcript_path = await self._transcript_chunks(
                job, user, video, set_stage, media_files)

            visual_chunks: list[TimedChunk] = []
            if job.options.include_visual and self.settings.VISION_BACKEND:
                visual_chunks = await self._visual_chunks(user, video, language, set_stage,
                                                          media_files)

            await set_stage(JobStage.INDEXING)
            nlp = await asyncio.to_thread(self.container.nlp_controller, with_reranker=False)
            chunks = self._to_chunk_documents(user, video, transcript_chunks, visual_chunks)
            # Embed before touching stored data: if embedding fails, a previously indexed
            # version of this video stays searchable.
            records = await nlp.build_records(video, chunks)
            if await self.video_model.get_video(video.id) is None:
                raise VideoProcessingError("Video was deleted while it was being processed")
            await self.chunk_model.delete_chunks_by_video_id(video.id)
            await nlp.delete_video_vectors(user, str(video.id))
            await self.chunk_model.insert_many_chunks(chunks)
            await nlp.upsert_records(user, records)

            await self.video_model.update_video(
                video.id, status=VideoStatus.INDEXED, language=language,
                transcript_path=transcript_path, transcript_chunks=len(transcript_chunks),
                visual_chunks=len(visual_chunks),
            )
            result.status = VideoStatus.INDEXED
            result.transcript_chunks = len(transcript_chunks)
            result.visual_chunks = len(visual_chunks)
        except Exception as exc:
            logger.exception("Processing video %s failed", video.youtube_video_id)
            message = str(exc) or exc.__class__.__name__
            await self.video_model.update_video(video.id, status=VideoStatus.FAILED,
                                                error=message)
            result.status = VideoStatus.FAILED
            result.error = message
        finally:
            if not self.settings.KEEP_MEDIA_FILES:
                for path in media_files:
                    with contextlib.suppress(OSError):
                        os.remove(path)
        return result

    async def _transcript_chunks(self, job, user, video, set_stage, media_files):
        downloader = self.container.video_controller

        await set_stage(JobStage.DOWNLOADING)
        audio_path = await asyncio.to_thread(downloader.download_audio,
                                             video.youtube_video_id, user.user_id)
        media_files.append(audio_path)

        await set_stage(JobStage.TRANSCRIBING)
        transcriber = await asyncio.to_thread(lambda: self.container.transcriber)
        audio = AudioController(transcriber, self.settings)
        transcription = await asyncio.to_thread(audio.transcribe, audio_path,
                                                job.options.language)
        if not transcription.segments:
            raise VideoProcessingError("Transcription produced no speech")
        transcript_path = audio.save_transcript(transcription, user.user_id,
                                                video.youtube_video_id)

        await set_stage(JobStage.CHUNKING)
        chunks = ChunkController().build_chunks(
            transcription.segments,
            max_chars=job.options.chunk_max_chars,
            overlap_segments=job.options.chunk_overlap_segments,
        )
        if not chunks:
            raise VideoProcessingError("Chunking produced no chunks")
        return chunks, transcription.language, transcript_path

    async def _visual_chunks(self, user, video, language, set_stage, media_files):
        # The visual pipeline is best-effort: a failure here must not lose the transcript.
        try:
            vision_client = await asyncio.to_thread(lambda: self.container.vision_client)
            if vision_client is None:
                return []
            await set_stage(JobStage.EXTRACTING_FRAMES)
            video_path = await asyncio.to_thread(self.container.video_controller.download_video,
                                                 video.youtube_video_id, user.user_id)
            media_files.append(video_path)
            vision = VisionController(vision_client, self.container.template_parser,
                                      self.settings)
            keyframes = await asyncio.to_thread(vision.extract_keyframes, video_path,
                                                video.duration_seconds)

            await set_stage(JobStage.CAPTIONING)
            return await asyncio.to_thread(vision.caption_keyframes, keyframes, video.title,
                                           language)
        except Exception:
            logger.exception("Visual pipeline failed for %s; continuing with transcript only",
                             video.youtube_video_id)
            return []

    @staticmethod
    def _to_chunk_documents(user: User, video: Video, transcript: list[TimedChunk],
                            visual: list[TimedChunk]) -> list[Chunk]:
        ordered = [(c, Modality.TRANSCRIPT) for c in transcript] + \
                  [(c, Modality.VISUAL) for c in visual]
        return [
            Chunk(
                _id=ObjectId(),
                chunk_user_id=user.id,
                chunk_video_id=video.id,
                chunk_text=chunk.text,
                chunk_order=i + 1,
                start=chunk.start,
                end=chunk.end,
                modality=modality,
            )
            for i, (chunk, modality) in enumerate(ordered)
        ]

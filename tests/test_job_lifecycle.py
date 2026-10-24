"""Regression tests for job/video state under failures, restarts and concurrent requests."""

import asyncio

import pytest

from controllers.JobController import JobController
from controllers.ProcessController import ProcessController
from models.db_schemas import Job
from models.enums import JobStatus, VideoStatus
from models.JobModel import JobModel
from models.VideoModel import VideoModel
from tests.fakes import FakeEmbedder, FakeTranscriber

VIDEO = "https://youtu.be/FastAPI0001"


async def _add(client, user="alice"):
    response = await client.post(f"/api/v1/users/{user}/videos", json={"youtube_urls": [VIDEO]})
    return response.json()["added"][0]["video_id"]


async def _video(client, user="alice"):
    return (await client.get(f"/api/v1/users/{user}/videos")).json()["videos"][0]


async def test_double_submit_queues_a_video_only_once(client, container):
    await _add(client)
    first = await client.post("/api/v1/users/alice/process", json={})
    second = await client.post("/api/v1/users/alice/process", json={})  # before the job runs
    assert first.status_code == 202
    assert second.status_code == 409
    await container.job_queue.wait_idle()
    assert (await _video(client))["status"] == "indexed"


async def test_enqueue_failure_does_not_strand_videos(client, container):
    await _add(client)

    class BrokenQueue:
        async def enqueue_process(self, job_id):
            raise ConnectionError("redis down")

        async def close(self):
            return None

    container.job_queue = BrokenQueue()
    # The test transport re-raises app errors; a real client gets a 500 internal_error.
    with pytest.raises(ConnectionError):
        await client.post("/api/v1/users/alice/process", json={})
    video = await _video(client)
    assert video["status"] == "failed"
    assert "redis down" in video["error"]


class CancellingTranscriber(FakeTranscriber):
    def transcribe(self, audio_path, language=None):
        raise asyncio.CancelledError()  # e.g. shutdown or the worker's job timeout


async def test_cancelled_job_marks_job_and_videos_failed(client, container):
    await _add(client)
    container.transcriber = CancellingTranscriber()
    user = await container.db_client["users"].find_one({"user_id": "alice"})
    video_ids = [v["_id"] async for v in container.db_client["videos"].find()]
    await VideoModel(container.db_client).claim_for_processing(video_ids)
    job = await JobModel(container.db_client).insert_job(
        Job(job_user_id=user["_id"], user_id="alice", video_ids=video_ids))

    with pytest.raises(asyncio.CancelledError):
        await ProcessController(container).run_job(str(job.id))

    job = await JobModel(container.db_client).get_job(str(job.id))
    assert job.status == JobStatus.FAILED
    assert (await _video(client))["status"] == "failed"


async def test_restart_recovers_interrupted_jobs(client, container):
    await _add(client)
    user = await container.db_client["users"].find_one({"user_id": "alice"})
    video_ids = [v["_id"] async for v in container.db_client["videos"].find()]
    await VideoModel(container.db_client).claim_for_processing(video_ids)
    job = await JobModel(container.db_client).insert_job(
        Job(job_user_id=user["_id"], user_id="alice", video_ids=video_ids,
            status=JobStatus.RUNNING))

    assert await JobController(container).recover_interrupted([JobStatus.RUNNING]) == 1
    assert (await JobModel(container.db_client).get_job(str(job.id))).status == JobStatus.FAILED
    video = await _video(client)
    assert video["status"] == "failed"
    assert "interrupted" in video["error"]
    # ...and the video can simply be processed again.
    assert (await client.post("/api/v1/users/alice/process", json={})).status_code == 202


class FailingEmbedder(FakeEmbedder):
    def embed_documents(self, texts):
        raise RuntimeError("CUDA out of memory")


async def test_failed_reprocess_keeps_the_previous_index(client, container, add_and_process):
    await add_and_process()
    vectors_before = await container.vectordb.count("collection_alice")
    chunks_before = await container.db_client["chunks"].count_documents({})

    container.embedder = FailingEmbedder()
    response = await client.post("/api/v1/users/alice/process", json={"reprocess": True})
    assert response.status_code == 202
    await container.job_queue.wait_idle()

    assert (await _video(client))["status"] == "failed"
    assert await container.vectordb.count("collection_alice") == vectors_before
    assert await container.db_client["chunks"].count_documents({}) == chunks_before
    container.embedder = FakeEmbedder()
    search = await client.post("/api/v1/users/alice/search", json={"query": "Pydantic"})
    assert search.json()["results"]


async def test_cannot_delete_a_video_while_processing(client, container):
    video_id = await _add(client)
    await client.post("/api/v1/users/alice/process", json={})  # queued, not run yet
    response = await client.delete(f"/api/v1/users/alice/videos/{video_id}")
    assert response.status_code == 409
    assert response.json()["signal"] == "video_is_processing"
    await container.job_queue.wait_idle()
    assert (await client.delete(f"/api/v1/users/alice/videos/{video_id}")).status_code == 200


async def test_stopword_only_sparse_query_returns_nothing(client, add_and_process):
    await add_and_process()
    response = await client.post("/api/v1/users/alice/search",
                                 json={"query": "what is the", "mode": "sparse"})
    assert response.json()["results"] == []


async def test_stream_reports_non_llm_failures_in_band(client, add_and_process, container):
    await add_and_process()

    class BrokenEmbedder(FakeEmbedder):
        def embed_query(self, text):
            raise RuntimeError("embedder crashed")

    container.embedder = BrokenEmbedder()
    async with client.stream("POST", "/api/v1/users/alice/answer/stream",
                             json={"query": "Starlette"}) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])
    assert "event: error" in body
    assert "internal_error" in body


def test_video_status_enum_round_trip():
    assert VideoStatus("processing") is VideoStatus.PROCESSING

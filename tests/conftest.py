import httpx
import pytest
from mongomock_motor import AsyncMongoMockClient

from controllers.ProcessController import ProcessController
from controllers.VideoController import VideoController
from core.container import ServiceContainer
from helpers.config import Settings
from main import create_app
from models.database import ensure_indexes
from stores.jobs.JobQueue import InlineJobQueue
from stores.vectordb.providers import QdrantDBProvider
from tests.fakes import FakeDownloader, FakeEmbedder, FakeLLM, FakeTranscriber, KeywordReranker


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        ASSETS_DIR=str(tmp_path / "assets"),
        MONGODB_DATABASE="test",
        JOB_QUEUE_BACKEND="inline",
        VISION_BACKEND="",
        RERANK_ENABLED=False,
        RETRIEVAL_MODE="hybrid",
        CHUNK_MAX_CHARS=120,
        CHUNK_OVERLAP_SEGMENTS=1,
        MAX_VIDEO_DURATION_MINUTES=10,
    )


@pytest.fixture
async def container(settings):
    db = AsyncMongoMockClient()[settings.MONGODB_DATABASE]
    await ensure_indexes(db)
    vectordb = QdrantDBProvider(path=":memory:")
    await vectordb.connect()

    c = ServiceContainer(settings, db_client=db, vectordb=vectordb)
    c.embedder = FakeEmbedder()
    c.reranker = KeywordReranker()
    c.generation_client = FakeLLM()
    c.vision_client = FakeLLM()
    c.transcriber = FakeTranscriber()
    c.video_controller = FakeDownloader(VideoController(settings), durations={"LongVideo01": 3600})
    c.job_queue = InlineJobQueue(runner=lambda job_id: ProcessController(c).run_job(job_id))
    yield c
    await c.close()


@pytest.fixture
async def client(container):
    app = create_app(container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture
def add_and_process(client, container):
    """Add videos, run the processing job to completion and return the finished job."""

    async def _run(user_id="alice", urls=("https://youtu.be/FastAPI0001",), **process_body):
        response = await client.post(f"/api/v1/users/{user_id}/videos",
                                     json={"youtube_urls": list(urls)})
        assert response.status_code == 201, response.text
        response = await client.post(f"/api/v1/users/{user_id}/process", json=process_body)
        assert response.status_code == 202, response.text
        await container.job_queue.wait_idle()
        job_id = response.json()["job"]["job_id"]
        response = await client.get(f"/api/v1/users/{user_id}/jobs/{job_id}")
        return response.json()["job"]

    return _run

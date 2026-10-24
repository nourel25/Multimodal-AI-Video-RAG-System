import json

from controllers.VisionController import Keyframe, VisionController
from tests.fakes import FakeLLM, FakeTranscriber

VIDEO = "https://youtu.be/FastAPI0001"


async def test_health(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["config"]["job_queue"] == "inline"


async def test_add_videos_reports_each_skipped_url(client):
    response = await client.post("/api/v1/users/alice/videos", json={"youtube_urls": [
        VIDEO,
        "https://www.youtube.com/watch?v=FastAPI0001",  # same video, other URL shape
        "https://vimeo.com/1234",
        "https://youtu.be/LongVideo01",
        "https://youtu.be/unavailable",
    ]})
    assert response.status_code == 201
    body = response.json()
    assert [v["youtube_video_id"] for v in body["added"]] == ["FastAPI0001"]
    assert body["added"][0]["status"] == "pending"
    assert {s["signal"] for s in body["skipped"]} == {
        "video_already_exists", "video_invalid_url", "video_duration_exceeded",
        "video_metadata_failed"}

    again = await client.post("/api/v1/users/alice/videos", json={"youtube_urls": [VIDEO]})
    assert again.json()["skipped"][0]["signal"] == "video_already_exists"


async def test_unknown_user_is_404_not_a_crash(client):
    for method, path, body in [
        ("get", "/api/v1/users/ghost/videos", None),
        ("post", "/api/v1/users/ghost/process", {}),
        ("post", "/api/v1/users/ghost/search", {"query": "x"}),
        ("post", "/api/v1/users/ghost/answer", {"query": "x"}),
    ]:
        response = await client.request(method, path, json=body)
        assert response.status_code == 404, path
        assert response.json()["signal"] == "user_not_found"


async def test_invalid_user_id_is_rejected(client):
    response = await client.post("/api/v1/users/a!/videos", json={"youtube_urls": [VIDEO]})
    assert response.status_code == 422
    assert response.json()["signal"] == "validation_error"


async def test_processing_indexes_video_with_timestamps(client, add_and_process):
    job = await add_and_process()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["results"][0]["transcript_chunks"] > 1

    videos = (await client.get("/api/v1/users/alice/videos")).json()["videos"]
    assert videos[0]["status"] == "indexed"
    assert videos[0]["language"] == "en"

    response = await client.post("/api/v1/users/alice/search",
                                 json={"query": "Pydantic data validation", "limit": 3})
    top = response.json()["results"][0]
    assert "Pydantic" in top["text"]
    assert top["start"] > 0
    assert top["url"] == f"https://www.youtube.com/watch?v=FastAPI0001&t={int(top['start'])}s"
    assert top["modality"] == "transcript"


async def test_processing_twice_does_not_duplicate(client, add_and_process, container):
    await add_and_process()
    response = await client.post("/api/v1/users/alice/process", json={})
    assert response.status_code == 409  # nothing new to process

    response = await client.post("/api/v1/users/alice/process", json={"reprocess": True})
    assert response.status_code == 202
    await container.job_queue.wait_idle()
    count = await container.vectordb.count("collection_alice")
    chunks = await container.db_client["chunks"].count_documents({})
    assert count == chunks  # vectors were replaced, not appended


async def test_failed_transcription_marks_video_failed(client, add_and_process, container):
    container.transcriber = FakeTranscriber(fail=True)
    job = await add_and_process()
    assert job["status"] == "failed"
    assert "whisper exploded" in job["results"][0]["error"]
    video = (await client.get("/api/v1/users/alice/videos")).json()["videos"][0]
    assert video["status"] == "failed"

    # Failed videos are picked up again by a plain retry.
    container.transcriber = FakeTranscriber()
    response = await client.post("/api/v1/users/alice/process", json={})
    assert response.status_code == 202


async def test_language_option_is_passed_to_transcriber(add_and_process, container):
    await add_and_process(language="ar")
    assert container.transcriber.languages == ["ar"]


async def test_answer_returns_citations_and_uses_system_role(client, add_and_process, container):
    await add_and_process()
    response = await client.post("/api/v1/users/alice/answer", json={
        "query": "What powers the web parts?",
        "history": [{"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "FastAPI is built on Starlette [1]."
    assert body["language"] == "en"
    assert body["sources"][0]["index"] == 1

    messages = container.generation_client.calls[-1]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "Source [1]" in messages[-1]["content"]


async def test_answer_in_arabic_uses_arabic_prompt(client, add_and_process, container):
    await add_and_process()
    await client.post("/api/v1/users/alice/answer", json={"query": "ما هو FastAPI وما فائدته؟"})
    assert "السؤال" in container.generation_client.calls[-1][-1]["content"]


async def test_provider_failure_returns_502(client, add_and_process, container):
    await add_and_process()
    container.generation_client = FakeLLM(fail=True)
    response = await client.post("/api/v1/users/alice/answer", json={"query": "Starlette"})
    assert response.status_code == 502
    assert response.json()["signal"] == "provider_error"


async def test_answer_stream_emits_sources_then_tokens(client, add_and_process):
    await add_and_process()
    async with client.stream("POST", "/api/v1/users/alice/answer/stream",
                             json={"query": "Starlette web parts"}) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    events = [line.removeprefix("event: ").strip() for line in body.splitlines()
              if line.startswith("event:")]
    assert events[0] == "sources"
    assert events[-1] == "done"
    data = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data:")]
    assert json.loads(data[0])[0]["url"].startswith("https://www.youtube.com/watch?v=")
    assert "".join(json.loads(d) for d in data[1:-1]).startswith("FastAPI")


async def test_delete_video_removes_chunks_and_vectors(client, add_and_process, container):
    await add_and_process()
    video_id = (await client.get("/api/v1/users/alice/videos")).json()["videos"][0]["video_id"]
    response = await client.delete(f"/api/v1/users/alice/videos/{video_id}")
    assert response.json()["deleted_chunks"] > 0
    assert await container.vectordb.count("collection_alice") == 0
    missing = await client.get(f"/api/v1/users/alice/videos/{video_id}")
    assert missing.status_code == 404


async def test_rebuild_index_restores_vectors(client, add_and_process, container):
    await add_and_process()
    before = await container.vectordb.count("collection_alice")
    await container.vectordb.delete_collection("collection_alice")
    response = await client.post("/api/v1/users/alice/index/rebuild")
    assert response.json()["indexed_chunks"] == before
    assert await container.vectordb.count("collection_alice") == before


async def test_visual_chunks_are_indexed(client, add_and_process, container, monkeypatch):
    container.settings.VISION_BACKEND = "FAKE"
    monkeypatch.setattr(VisionController, "extract_keyframes", lambda self, path, duration: [
        Keyframe(start=10.0, end=20.0, timestamp=15.0, image=b"jpeg")])

    job = await add_and_process()
    assert job["results"][0]["visual_chunks"] == 1

    response = await client.post("/api/v1/users/alice/search",
                                 json={"query": "slide Depends", "limit": 1, "mode": "sparse"})
    top = response.json()["results"][0]
    assert top["modality"] == "visual"
    assert top["start"] == 10.0


async def test_visual_failure_keeps_transcript(client, add_and_process, container, monkeypatch):
    container.settings.VISION_BACKEND = "FAKE"

    def broken(self, path, duration):
        raise RuntimeError("no codec")

    monkeypatch.setattr(VisionController, "extract_keyframes", broken)
    job = await add_and_process()
    assert job["status"] == "completed"
    assert job["results"][0]["visual_chunks"] == 0


async def test_rerank_reorders_results(client, add_and_process):
    await add_and_process()
    response = await client.post("/api/v1/users/alice/search", json={
        "query": "Swagger interactive documentation", "limit": 2, "rerank": True})
    assert "Swagger" in response.json()["results"][0]["text"]


async def test_api_key_is_enforced_when_configured(client, container):
    container.settings.API_KEY = "secret"
    url = "/api/v1/users/alice/videos"
    assert (await client.get(url)).status_code == 401
    assert (await client.get(url, headers={"X-API-Key": "wrong"})).status_code == 401
    ok = await client.post(url, json={"youtube_urls": [VIDEO]}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 201
    assert (await client.get("/api/v1/health")).status_code == 200  # health stays public

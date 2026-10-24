"""Thin client for the Multimodal Video RAG API (kept separate so the UI stays declarative)."""

import json
from collections.abc import Iterator

import httpx


class APIError(RuntimeError):
    def __init__(self, status_code: int, signal: str, detail=None):
        super().__init__(f"{status_code} {signal}: {detail}")
        self.status_code = status_code
        self.signal = signal
        self.detail = detail


class RagClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 60.0):
        headers = {"X-API-Key": api_key} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self.http.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = {"signal": "http_error", "detail": response.text}
            raise APIError(response.status_code, body.get("signal", "error"), body.get("detail"))
        return response.json()

    def health(self) -> dict:
        return self._request("GET", "/api/v1/health")

    def add_videos(self, user_id: str, urls: list[str], auto_process: bool,
                   include_visual: bool) -> dict:
        return self._request("POST", f"/api/v1/users/{user_id}/videos", json={
            "youtube_urls": urls, "auto_process": auto_process, "include_visual": include_visual})

    def list_videos(self, user_id: str) -> list[dict]:
        try:
            return self._request("GET", f"/api/v1/users/{user_id}/videos")["videos"]
        except APIError as exc:
            if exc.signal == "user_not_found":
                return []
            raise

    def delete_video(self, user_id: str, video_id: str) -> dict:
        return self._request("DELETE", f"/api/v1/users/{user_id}/videos/{video_id}")

    def process(self, user_id: str, video_ids: list[str] | None = None, reprocess: bool = False,
                include_visual: bool = True) -> dict:
        return self._request("POST", f"/api/v1/users/{user_id}/process", json={
            "video_ids": video_ids, "reprocess": reprocess, "include_visual": include_visual})

    def list_jobs(self, user_id: str, limit: int = 5) -> list[dict]:
        try:
            return self._request("GET", f"/api/v1/users/{user_id}/jobs",
                                 params={"limit": limit})["jobs"]
        except APIError as exc:
            if exc.signal == "user_not_found":
                return []
            raise

    def search(self, user_id: str, query: str, **options) -> list[dict]:
        return self._request("POST", f"/api/v1/users/{user_id}/search",
                             json={"query": query, **options})["results"]

    def stream_answer(self, user_id: str, query: str, history: list[dict], **options
                      ) -> Iterator[tuple[str, object]]:
        """Yield ("sources", list) / ("token", str) / ("error", signal) from the SSE endpoint."""
        payload = {"query": query, "history": history, **options}
        with self.http.stream("POST", f"/api/v1/users/{user_id}/answer/stream", json=payload,
                              timeout=None) as response:
            if response.status_code >= 400:
                response.read()
                body = response.json()
                raise APIError(response.status_code, body.get("signal"), body.get("detail"))
            event, data = "message", []
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line and data:
                    raw = "\n".join(data)
                    if event == "sources":
                        yield "sources", json.loads(raw)
                    elif event == "token":
                        yield "token", json.loads(raw)
                    elif event == "error":
                        yield "error", raw
                    event, data = "message", []

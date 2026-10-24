import re
from urllib.parse import parse_qs, urlparse

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")


def extract_video_id(url: str) -> str | None:
    """Return the 11-char YouTube video id for any common URL shape, or None."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()

    candidate = None
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        else:
            for prefix in _PATH_PREFIXES:
                if parsed.path.startswith(prefix):
                    candidate = parsed.path[len(prefix):].split("/")[0]
                    break

    if candidate and _VIDEO_ID.match(candidate):
        return candidate
    return None


def watch_url(youtube_video_id: str, start_seconds: float | None = None) -> str:
    url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    if start_seconds is not None and start_seconds > 0:
        url += f"&t={int(start_seconds)}s"
    return url


def format_timestamp(seconds: float) -> str:
    total = int(max(seconds, 0))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

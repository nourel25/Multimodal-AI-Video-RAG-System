import logging
from dataclasses import dataclass

import yt_dlp

from helpers.youtube import watch_url
from models import ResponseSignal

from .BaseController import BaseController

logger = logging.getLogger(__name__)

# Low-bitrate audio keeps files small (fast uploads to Groq, ~25 MB for an hour) and
# needs no ffmpeg post-processing: faster-whisper decodes webm/m4a directly.
AUDIO_FORMAT = "bestaudio[abr<=64]/worstaudio/bestaudio"
# Video-only H.264 at <=480p is enough for scene detection and captioning, and OpenCV
# can decode it without extra codecs.
FRAMES_FORMAT = (
    "bestvideo[height<=480][vcodec^=avc1]/bestvideo[height<=480]/worst[height>=240]/worst"
)


class VideoDownloadError(RuntimeError):
    pass


@dataclass
class VideoMetadata:
    youtube_video_id: str
    title: str
    channel: str
    duration_seconds: float
    thumbnail_url: str
    is_live: bool


class VideoController(BaseController):
    _BASE_OPTS = {"quiet": True, "no_warnings": True, "noplaylist": True}

    def fetch_metadata(self, youtube_video_id: str) -> VideoMetadata:
        with yt_dlp.YoutubeDL({**self._BASE_OPTS, "skip_download": True}) as ydl:
            info = ydl.extract_info(watch_url(youtube_video_id), download=False)
        return VideoMetadata(
            youtube_video_id=info.get("id") or youtube_video_id,
            title=info.get("title") or "",
            channel=info.get("channel") or info.get("uploader") or "",
            duration_seconds=float(info.get("duration") or 0),
            thumbnail_url=info.get("thumbnail") or "",
            is_live=bool(info.get("is_live")),
        )

    def validate_metadata(self, metadata: VideoMetadata) -> tuple[bool, str]:
        if metadata.is_live:
            return False, ResponseSignal.VIDEO_IS_LIVE.value
        max_seconds = self.app_settings.MAX_VIDEO_DURATION_MINUTES * 60
        if metadata.duration_seconds > max_seconds:
            return False, ResponseSignal.VIDEO_DURATION_EXCEEDED.value
        return True, ResponseSignal.VIDEO_ADDED.value

    def _download(self, youtube_video_id: str, directory, fmt: str) -> str:
        opts = {
            **self._BASE_OPTS,
            "format": fmt,
            "outtmpl": str(directory / f"{youtube_video_id}.%(ext)s"),
            "overwrites": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(watch_url(youtube_video_id), download=True)
                downloads = info.get("requested_downloads") or []
                path = downloads[0].get("filepath") if downloads else None
                return path or ydl.prepare_filename(info)
        except Exception as exc:
            raise VideoDownloadError(f"Download of {youtube_video_id} failed: {exc}") from exc

    def download_audio(self, youtube_video_id: str, user_id: str) -> str:
        return self._download(youtube_video_id, self.user_dir(self.audios_dir, user_id),
                              AUDIO_FORMAT)

    def download_video(self, youtube_video_id: str, user_id: str) -> str:
        return self._download(youtube_video_id, self.user_dir(self.videos_dir, user_id),
                              FRAMES_FORMAT)

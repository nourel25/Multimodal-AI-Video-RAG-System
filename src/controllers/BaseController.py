from pathlib import Path

from helpers.config import Settings, get_settings


class BaseController:
    def __init__(self, settings: Settings | None = None):
        self.app_settings = settings or get_settings()

        assets = self.app_settings.assets_path
        self.audios_dir = assets / "audios"
        self.videos_dir = assets / "videos"
        self.transcripts_dir = assets / "transcripts"
        self.database_dir = assets / "database"
        for directory in (self.audios_dir, self.videos_dir, self.transcripts_dir,
                          self.database_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def user_dir(base: Path, user_id: str) -> Path:
        path = base / user_id
        path.mkdir(parents=True, exist_ok=True)
        return path

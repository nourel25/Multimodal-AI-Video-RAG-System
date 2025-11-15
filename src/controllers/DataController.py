from .BaseController import BaseController
from pathlib import Path
import yt_dlp
import uuid
from models import ResponseSignal 

class DataController(BaseController):
    def __init__(self):
        super().__init__()
        
    def generate_audio_path(self, user_id: str):
        user_dir = Path(self.audios_dir) / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Use video_id if available, otherwise random UUID
        unique_name = f"{uuid.uuid4().hex}.mp3"
        return str(user_dir / unique_name)
    
    def download_youtube_audio(self, youtube_url: str, file_path: str):

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': file_path.replace(".mp3", ""), 
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'noplaylist': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            return True, ResponseSignal.FILE_UPLOAD_SUCCESS.value
        except Exception as e:
            print(f"Download failed: {e}")
            return False, ResponseSignal.FILE_UPLOAD_FAILED.value        
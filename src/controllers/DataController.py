from .BaseController import BaseController
from pathlib import Path
import yt_dlp
import uuid
from models import ResponseSignal 


class DataController(BaseController):
    def __init__(self):
        super().__init__()
    
    
    def get_video_info(self, youtube_url: str):
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'noplaylist': True,      
            'forcejson': True,      
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        return info

    def validate_uploaded_video(self, youtube_url: str):
        info = self.get_video_info(youtube_url)
        filesize_mb = (info.get('filesize') or info.get('filesize_approx') or 0) / (1024 * 1024)
        
        if filesize_mb > self.app_settings.MAX_VIDEO_SIZE_MB:
            return False, ResponseSignal.VIDEO_SIZE_EXCEEDED.value

        return True, ResponseSignal.VIDEO_VALIDATED_SUCCESS.value
    
    def generate_audio_path(self, user_id: str):
        user_dir = Path(self.audios_dir) / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Use video_id if available, otherwise random UUID
        unique_name = f"{uuid.uuid4().hex}.mp3"
        return str(user_dir / unique_name)
    
    def download_youtube_audio(self, youtube_url: str, audio_path: str):

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path.replace(".mp3", ""), 
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
            return True, ResponseSignal.VIDEO_UPLOAD_SUCCESS.value
        except Exception as e:
            print(f"Download failed: {e}")
            return False, ResponseSignal.VIDEO_UPLOAD_FAILED.value 
        
       
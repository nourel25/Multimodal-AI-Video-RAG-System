from .BaseController import BaseController
from faster_whisper import WhisperModel
from pathlib import Path
import uuid
from models.enums.ResponseEnum import ResponseSignal
import torch


class AudioController(BaseController):
    def __init__(self):
        super().__init__()
        # Check for GPU
        if torch.cuda.is_available():
            device = "cuda"
            print("✅ CUDA detected: Using GPU for transcription")
        else:
            device = "cpu"
            print("⚠️ CUDA not detected: Using CPU for transcription (slower)")

        self.model = WhisperModel('small', device=device)
            
    def generate_transcript_path(self, user_id: str):
        user_dir = Path(self.transcripts_dir) / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        unique_name = f"{uuid.uuid4().hex}.txt"
        return  str(user_dir / unique_name) 
    
    def transcribe_audio(self, audio_path: str, transcript_path: str):
        try:
            segments, _ = self.model.transcribe(audio_path, language='ar')
            transcript = " ".join([segment.text for segment in segments])
            
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            return True, ResponseSignal.AUDIO_TRANSCRIPT_SUCCESS.value
        except Exception as e:
            print(f"Transcription failed: {e}")
            return False, ResponseSignal.AUDIO_TRANSCRIPT_FAILED.value   

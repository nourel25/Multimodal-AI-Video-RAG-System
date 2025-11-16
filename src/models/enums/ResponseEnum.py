from enum import Enum

class ResponseSignal(Enum):
    VIDEO_UPLOAD_SUCCESS = "video_upload_success"
    VIDEO_UPLOAD_FAILED = "video_upload_failed"
    PROCESSING_SUCCESS = "processing_success"
    PROCESSING_FAILED = "processing_failed"
    VIDEO_SIZE_EXCEEDED = "video_size_exceeded"
    VIDEO_VALIDATED_SUCCESS = "video_validate_successfully"
    AUDIO_TRANSCRIPT_SUCCESS = "audio_transcript_success"
    AUDIO_TRANSCRIPT_FAILED = "audio_transcript_failed"
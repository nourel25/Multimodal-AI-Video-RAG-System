from enum import Enum


class ResponseSignal(Enum):
    # Videos
    VIDEO_ADDED = "video_added"
    VIDEO_ALREADY_EXISTS = "video_already_exists"
    VIDEO_INVALID_URL = "video_invalid_url"
    VIDEO_METADATA_FAILED = "video_metadata_failed"
    VIDEO_DURATION_EXCEEDED = "video_duration_exceeded"
    VIDEO_IS_LIVE = "video_is_live"
    VIDEO_NOT_FOUND = "video_not_found"
    VIDEO_DELETED = "video_deleted"
    VIDEOS_LISTED = "videos_listed"

    # Processing
    VIDEO_DOWNLOAD_FAILED = "video_download_failed"
    AUDIO_TRANSCRIPT_FAILED = "audio_transcript_failed"
    CHUNKING_FAILED = "chunking_failed"
    PROCESSING_QUEUED = "processing_queued"
    NO_VIDEOS_TO_PROCESS = "no_videos_to_process"
    JOB_FOUND = "job_found"
    JOB_NOT_FOUND = "job_not_found"

    # Users
    USER_NOT_FOUND = "user_not_found"

    # Vector DB / RAG
    INDEX_REBUILT = "index_rebuilt"
    VECTORDB_SEARCH_SUCCESS = "vectordb_search_success"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    RAG_ANSWER_SUCCESS = "rag_answer_success"
    RAG_ANSWER_ERROR = "rag_answer_error"

    # Generic
    UNAUTHORIZED = "unauthorized"
    VALIDATION_ERROR = "validation_error"
    INTERNAL_ERROR = "internal_error"
    PROVIDER_ERROR = "provider_error"

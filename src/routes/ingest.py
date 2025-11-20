from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

# Controllers
from controllers.VideoController import VideoController
from controllers.AudioController import AudioController
from controllers.ChunkController import ChunkController

# Models
from models.UserModel import UserModel
from models.VideoModel import VideoModel
from models.ChunkModel import ChunkModel

# DB Schemas
from models.db_schemas import Video, Chunk

# Request Schemas
from .schemas.ingest import IngestRequest
from .schemas.process import ProcessRequest

# Enums
from models.enums.ResponseEnum import ResponseSignal



ingest_router = APIRouter()


@ingest_router.post("/ingest/{user_id}")
async def ingest_urls(request: Request, user_id: str, ingest_request: IngestRequest):
    video_controller = VideoController()
    user_model = UserModel(request.app.db_client)

    youtube_urls = ingest_request.youtube_urls

    inserted_urls = []
    failed_urls = []

    for youtube_url in youtube_urls:
        youtube_url = str(youtube_url)
        
        valid, v_signal = video_controller.validate_uploaded_video(youtube_url)
    
        if not valid:
            failed_urls.append({"url": youtube_url, "signal": v_signal})
            continue
        
    
        user = await user_model.get_user_or_insert_one(
            user_id=user_id
        )
        user = await user_model.insert_youtube_url(user.id, youtube_url)
        
        inserted_urls.append(youtube_url)

    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "success",
            "inserted_urls": inserted_urls,
            "failed_urls": failed_urls,
            "videos_user_id": str(user.id),
        }
    )


@ingest_router.post("/process/{user_id}")
async def process_audio(request: Request, user_id: str, process_request: ProcessRequest):
        
    user_model = UserModel(request.app.db_client)
    video_model = VideoModel(request.app.db_client)
    chunk_model = ChunkModel(db_client=request.app.db_client)
    
    video_controller = VideoController()
    chunk_controller = ChunkController()
    audio_controller = AudioController()
    
    user = await user_model.get_user(user_id)

    do_reset     = process_request.do_reset
    chunk_size   = process_request.chunk_size
    overlap_size = process_request.overlap_size

    youtube_urls = user.youtube_urls   # ✔ list of URLs

    if not youtube_urls or len(youtube_urls) == 0:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, 
            content={
                "status": "error",
                "signal": ResponseSignal.NO_URLS_FOUND_FOR_USER.value
            }
        )

    results = []   # store result for each URL

    # Optional reset
    if do_reset == 1:
        prev_videos = await video_model.get_videos_by_user_id(user.id)
        for prev_video in prev_videos:
            await chunk_model.delete_chunks_by_video_id(prev_video.id)
        await video_model.delete_video_by_user_id(user.id)

    # Process each URL
    for youtube_url in youtube_urls:

        audio_path = video_controller.generate_audio_path(user_id)

        d_success, d_signal = video_controller.download_youtube_audio(
            youtube_url, audio_path
        )

        if not d_success:
            results.append({
                "url": youtube_url,
                "status": "download_error",
                "signal": d_signal
            })
            continue

        transcript_path = audio_controller.generate_transcript_path(user_id)
        t_success, t_signal = audio_controller.transcribe_audio(
            audio_path, transcript_path
        )

        if not t_success:
            results.append({
                "url": youtube_url,
                "status": "transcription_error",
                "signal": t_signal
            })
            continue

        file_content = chunk_controller.get_file_content(transcript_path)
        file_chunks = chunk_controller.process_file_content(
            file_content=file_content,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )

        if file_chunks is None or len(file_chunks) == 0:
            results.append({
                "url": youtube_url,
                "status": "chunking_failed"
            })
            continue

        # Insert video entry
        video = await video_model.insert_video(
            Video(
                video_user_id=user.id,
                youtube_url=youtube_url,
                audio_path=audio_path,
                transcript_path=transcript_path,
            )
        )

        # Prepare chunk docs
        file_chunks_docs = [
            Chunk(
                chunk_user_id=user.id,
                chunk_video_id=video.id,
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i+1
            )
            for i, chunk in enumerate(file_chunks)
        ]

        # Insert all chunks using bulk_write
        no_docs = await chunk_model.insert_many_chunks(file_chunks_docs)

        results.append({
            "url": youtube_url,
            "status": "success",
            "download_signal": d_signal,
            "transcription_signal": t_signal,
            "chunks_inserted": no_docs
        })

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "completed",
            "processed_urls": results
        }
    )

from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from controllers.VideoController import VideoController 
from models.VideoModel import VideoModel
from models.db_schemas import Video
from .schemas.ingest import IngestRequest
from controllers.AudioController import AudioController
from models.UserModel import UserModel
from .schemas.process import ProcessRequest
from models.ChunkModel import ChunkModel
from models.db_schemas import Chunk
from controllers.ChunkController import ChunkController
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
            "video_user_id": str(user.id),
        }
    )


@ingest_router.post("/process/{user_id}")
async def process_audio(request: Request, user_id: str, process_request: ProcessRequest):
        
    video_controller = VideoController()
    user_model = UserModel(request.app.db_client)
    
    user = await user_model.get_user(user_id)
    
    do_reset = process_request.do_reset
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    
    youtube_url = user.youtube_url
    video_user_id = user.id

    audio_path = video_controller.generate_audio_path(user_id)

    d_success, d_signal = video_controller.download_youtube_audio(youtube_url, audio_path)
            
    if not d_success:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "signal": d_signal,
            }
        )
        
    audio_controller = AudioController()
    transcript_path = audio_controller.generate_transcript_path(user_id)
    t_success, t_signal = audio_controller.transcribe_audio(audio_path, transcript_path)
    
    if not t_success:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "signal": t_signal}
        )
        
    video_model = VideoModel(
        request.app.db_client
    )
    
    chunk_model = ChunkModel(
        db_client=request.app.db_client
    )
        
    chunk_controller = ChunkController()
    file_content = chunk_controller.get_file_content(transcript_path)
    
    file_chunks = chunk_controller.process_file_content(
        file_content=file_content,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )
    
    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROCESSING_FAILED.value
            }
        )
        
    if do_reset == 1:
        prev_videos = await video_model.get_videos_by_user_id(video_user_id)
        for prev_video in prev_videos:
            await chunk_model.delete_chunks_by_video_id(prev_video.id)
            
        await video_model.delete_video_by_user_id(video_user_id)
      
        
    video = await video_model.insert_video(
        Video(
            video_user_id=video_user_id,
            youtube_url=youtube_url,
            audio_path=audio_path,
            transcript_path=transcript_path,
        )
    )
        
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

    
    no_docs = await chunk_model.insert_many_chunks(
        chunks=file_chunks_docs
    )
    
    
    return JSONResponse( 
        status_code=status.HTTP_200_OK,
        content={ 
            "status": "success", 
            "signals": {
                "download": d_signal,
                "transcription": t_signal,
            }, 
            "no_docs": no_docs
            } 
    )
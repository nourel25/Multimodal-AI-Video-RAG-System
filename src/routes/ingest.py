from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from controllers.VideoController import VideoController 
from models.VideoModel import VideoModel
from models.db_schemas import Video
from .schemas.ingest import IngestRequest
from controllers.AudioController import AudioController
from models.UserModel import UserModel
from .schemas.process import ProcessRequest


ingest_router = APIRouter()


@ingest_router.post("/ingest/{user_id}")
async def ingest_urls(request: Request, user_id: str, ingest_request: IngestRequest):
    video_controller = VideoController()

    youtube_url = str(ingest_request.youtube_url)

    valid, v_signal = video_controller.validate_uploaded_video(youtube_url)
    
    if not valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "signal": v_signal,
            }
        )
        
    user_model = UserModel(request.app.db_client)
    user = await user_model.get_user_or_insert_one(
        user_id=user_id
    )
        
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "success",
            "signals": {
                "validation": v_signal,
            },
            "video_user_id": str(user.id),
            "youtube_url": youtube_url,
        }
    )


@ingest_router.post("/process/{user_id}")
async def process_audio(request: Request, user_id: str, process_request: ProcessRequest):
        
    video_controller = VideoController()
    user_model = UserModel(request.app.db_client)
    
    user = await user_model.get_user(user_id)
    
    do_reset = process_request.do_reset
    youtube_url = process_request.youtube_url
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
        
    video_model = VideoModel(request.app.db_client)

    if do_reset == 1:
        await video_model.delete_video_by_user_id(video_user_id)

    video = await video_model.insert_video(
        Video(
            video_user_id=video_user_id,
            youtube_url=youtube_url,
            audio_path=audio_path,
            transcript_path=transcript_path
        )
    )
    
    return JSONResponse( 
        status_code=status.HTTP_200_OK,
        content={ 
            "status": "success", 
            "signals": {
                "download": d_signal,
                "transcription": t_signal,
            }, 
            "audio_path": audio_path, 
            "transcript_path": transcript_path 
            } 
    )
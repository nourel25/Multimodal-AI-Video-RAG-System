from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from controllers.DataController import DataController 
from models.VideoModel import VideoModel
from models.db_schemas import Video
from .schemas.ingest import IngestRequest
from controllers.AudioController import AudioController
from models.UserModel import UserModel


ingest_router = APIRouter()


@ingest_router.post("/ingest/{user_id}")
async def ingest_urls(request: Request, user_id: str, ingest_request: IngestRequest):
    data_controller = DataController()

    youtube_url = str(ingest_request.url)
    do_reset = ingest_request.do_reset

    valid, v_signal = data_controller.validate_uploaded_video(youtube_url)
    
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
        
    audio_path = data_controller.generate_audio_path(user_id)

    d_success, d_signal = data_controller.download_youtube_audio(youtube_url, audio_path)
            
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
        await video_model.delete_video_by_user_id(user_id)

    video = await video_model.insert_video(
        Video(
            video_user_id=user.id,
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
                "validation": v_signal,
                "download": d_signal,
                "transcription": t_signal
            },
            "audio_path": audio_path,
            "transcript_path": transcript_path
        }
    )


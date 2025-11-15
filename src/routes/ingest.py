from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from controllers.DataController import DataController 
from models.VideoModel import VideoModel
from models.db_schemas import Video
from .schemas.ingest import IngestRequest

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
        
    audio_path = data_controller.generate_audio_path(user_id)

    success, d_signal = data_controller.download_youtube_audio(youtube_url, audio_path)
            
    if not success:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "signal": d_signal,
            }
        )
        
    video_model = VideoModel(request.app.db_client)

    if do_reset == 1:
        await video_model.delete_video_by_user_id(user_id)

    video = await video_model.insert_video(
        Video(
            user_id=user_id,
            youtube_url=youtube_url,
            file_path=audio_path
        )
    )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "success",
            "signal": d_signal,
            "audio_path": audio_path,
        }
    )

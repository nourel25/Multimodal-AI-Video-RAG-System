from .ChunkModel import ChunkModel
from .JobModel import JobModel
from .UserModel import UserModel
from .VideoModel import VideoModel


async def ensure_indexes(db_client) -> None:
    for model_cls in (UserModel, VideoModel, ChunkModel, JobModel):
        await model_cls(db_client).ensure_indexes()

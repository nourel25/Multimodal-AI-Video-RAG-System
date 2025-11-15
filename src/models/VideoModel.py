from .BaseDataModel import BaseDataModel
from .enums.DataBaseEnum import DataBaseEnum
from .db_schemas import Video
from bson import ObjectId

class VideoModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_VIDEO_NAME.value]
        
    async def insert_video(self, video: Video):
        result = await self.collection.insert_one(
            video.dict(by_alias=True, exclude_unset=True)
        )
        
        video.id = result.inserted_id
        
        return video
    
    async def delete_video_by_user_id(self, user_id: str):
        result = await self.collection.delete_many({
            'user_id': user_id
        })
        
        return result.deleted_count
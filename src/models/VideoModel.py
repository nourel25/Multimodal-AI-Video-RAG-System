from bson import ObjectId

from .BaseDataModel import BaseDataModel
from .db_schemas import Video
from .db_schemas.base import to_bson_value
from .enums import DataBaseEnum, VideoStatus


class VideoModel(BaseDataModel):
    collection_name = DataBaseEnum.COLLECTION_VIDEO_NAME.value

    async def ensure_indexes(self):
        await self.collection.create_index(
            [("video_user_id", 1), ("youtube_video_id", 1)], unique=True
        )

    async def insert_video(self, video: Video) -> Video:
        result = await self.collection.insert_one(video.to_mongo())
        video.id = result.inserted_id
        return video

    async def get_video(self, video_id: ObjectId) -> Video | None:
        record = await self.collection.find_one({"_id": video_id})
        return Video(**record) if record else None

    async def get_user_video(self, user_object_id: ObjectId, video_id: ObjectId) -> Video | None:
        record = await self.collection.find_one({"_id": video_id, "video_user_id": user_object_id})
        return Video(**record) if record else None

    async def find_by_youtube_id(self, user_object_id: ObjectId, youtube_video_id: str):
        record = await self.collection.find_one(
            {"video_user_id": user_object_id, "youtube_video_id": youtube_video_id}
        )
        return Video(**record) if record else None

    async def list_user_videos(
        self, user_object_id: ObjectId, statuses: list[VideoStatus] | None = None
    ) -> list[Video]:
        query: dict = {"video_user_id": user_object_id}
        if statuses:
            query["status"] = {"$in": [status.value for status in statuses]}
        docs = await self.collection.find(query).sort("created_at", 1).to_list(length=None)
        return [Video(**doc) for doc in docs]

    async def update_video(self, video_id: ObjectId, **fields) -> None:
        fields["updated_at"] = self.now()
        await self.collection.update_one({"_id": video_id}, {"$set": to_bson_value(fields)})

    async def claim_for_processing(self, video_ids: list[ObjectId]) -> list[ObjectId]:
        """Atomically mark videos as processing; returns only those this call claimed.

        A video already processing is skipped, so concurrent requests can't queue it twice.
        """
        claimed = []
        for video_id in video_ids:
            result = await self.collection.update_one(
                {"_id": video_id, "status": {"$ne": VideoStatus.PROCESSING.value}},
                {"$set": {"status": VideoStatus.PROCESSING.value, "error": None,
                          "updated_at": self.now()}},
            )
            if result.modified_count:
                claimed.append(video_id)
        return claimed

    async def fail_processing(self, video_ids: list[ObjectId], error: str) -> int:
        """Mark videos still 'processing' as failed (after a crash, restart or enqueue error)."""
        result = await self.collection.update_many(
            {"_id": {"$in": video_ids}, "status": VideoStatus.PROCESSING.value},
            {"$set": {"status": VideoStatus.FAILED.value, "error": error,
                      "updated_at": self.now()}},
        )
        return result.modified_count

    async def delete_video(self, video_id: ObjectId) -> int:
        result = await self.collection.delete_one({"_id": video_id})
        return result.deleted_count

from collections.abc import AsyncIterator

from bson import ObjectId
from pymongo import InsertOne

from .BaseDataModel import BaseDataModel
from .db_schemas import Chunk
from .enums import DataBaseEnum


class ChunkModel(BaseDataModel):
    collection_name = DataBaseEnum.COLLECTION_CHUNK_NAME.value

    async def ensure_indexes(self):
        await self.collection.create_index([("chunk_video_id", 1), ("chunk_order", 1)])
        await self.collection.create_index("chunk_user_id")

    async def insert_many_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> list[Chunk]:
        """Insert chunks and assign their generated ids (needed for vector point ids)."""
        for chunk in chunks:
            chunk.id = chunk.id or ObjectId()
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            await self.collection.bulk_write([InsertOne(chunk.to_mongo()) for chunk in batch])
        return chunks

    async def delete_chunks_by_video_id(self, video_id: ObjectId) -> int:
        result = await self.collection.delete_many({"chunk_video_id": video_id})
        return result.deleted_count

    async def get_video_chunks(self, video_id: ObjectId) -> list[Chunk]:
        cursor = self.collection.find({"chunk_video_id": video_id}).sort("chunk_order", 1)
        return [Chunk(**doc) for doc in await cursor.to_list(length=None)]

    async def iter_user_chunks(
        self, user_object_id: ObjectId, page_size: int = 200
    ) -> AsyncIterator[list[Chunk]]:
        """Yield the user's chunks page by page using keyset pagination on _id."""
        last_id = None
        while True:
            query: dict = {"chunk_user_id": user_object_id}
            if last_id is not None:
                query["_id"] = {"$gt": last_id}
            docs = await self.collection.find(query).sort("_id", 1).limit(page_size).to_list(
                length=None
            )
            if not docs:
                return
            last_id = docs[-1]["_id"]
            yield [Chunk(**doc) for doc in docs]

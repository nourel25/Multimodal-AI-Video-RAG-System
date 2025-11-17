from .BaseDataModel import BaseDataModel
from .db_schemas.chunk import Chunk
from .enums.DataBaseEnum import DataBaseEnum
from pymongo import InsertOne
from bson import ObjectId

class ChunkModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHUNK_NAME.value]
        
    async def insert_chunk(self, chunk: Chunk):
        result = await self.collection.insert_one(
            chunk.dict(by_alias=True, exclude_unset=True)
        )
        chunk.id = result.inserted_id
        return chunk
    
    async def get_chunk(self, chunk_id: str):
        result = await self.collection.find_one({
            "_id": ObjectId(chunk_id)
        })

        if result is None:
            return None
        
        return Chunk(**result)
      
    async def get_chunk_by_video_id(self, chunk_video_id: ObjectId):
        result = await self.collection.find_one({
            "chunk_video_id": chunk_video_id
        })

        if result is None:
            return None
        
        return Chunk(**result)  
    
     
    async def insert_many_chunks(self, chunks: list, batch_size: int=100):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            operations = [
                InsertOne(chunk.dict(by_alias=True, exclude_unset=True))
                for chunk in batch
            ]
            
            await self.collection.bulk_write(operations)
            
        return len(chunks)
            
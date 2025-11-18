from .BaseDataModel import BaseDataModel
from .db_schemas import User
from .enums.DataBaseEnum import DataBaseEnum


class UserModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_USER_NAME.value]
        
    async def insert_user(self, user: User):
        result = await self.collection.insert_one(user.dict(by_alias=True, exclude_unset=True))
        user.id = result.inserted_id

        return user
    
    async def get_user_or_insert_one(self, user_id: str, youtube_url: str):
        record = await self.collection.find_one({
            "user_id": user_id
        })
        
        if record is None:
            user = User(user_id=user_id, youtube_url=youtube_url)
            user = await self.insert_user(user=user)
            
            return user
        
        return User(**record)
    
    async def get_user(self, user_id: str):
        record = await self.collection.find_one({
            "user_id": user_id
        })
        
        if record is None:
            return None
        
        return User(**record)
from pymongo import ReturnDocument

from .BaseDataModel import BaseDataModel
from .db_schemas import User
from .enums import DataBaseEnum


class UserModel(BaseDataModel):
    collection_name = DataBaseEnum.COLLECTION_USER_NAME.value

    async def ensure_indexes(self):
        await self.collection.create_index("user_id", unique=True)

    async def get_user(self, user_id: str) -> User | None:
        record = await self.collection.find_one({"user_id": user_id})
        return User(**record) if record else None

    async def get_or_create_user(self, user_id: str) -> User:
        # Validate before touching the database so bad ids never get stored.
        new_user = User(user_id=user_id)
        record = await self.collection.find_one_and_update(
            {"user_id": user_id},
            {"$setOnInsert": new_user.to_mongo()},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return User(**record)

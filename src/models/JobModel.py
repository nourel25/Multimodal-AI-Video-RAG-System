from bson import ObjectId
from bson.errors import InvalidId

from .BaseDataModel import BaseDataModel
from .db_schemas import Job, JobVideoResult
from .db_schemas.base import to_bson_value
from .enums import DataBaseEnum


class JobModel(BaseDataModel):
    collection_name = DataBaseEnum.COLLECTION_JOB_NAME.value

    async def ensure_indexes(self):
        await self.collection.create_index([("job_user_id", 1), ("created_at", -1)])

    async def insert_job(self, job: Job) -> Job:
        result = await self.collection.insert_one(job.to_mongo())
        job.id = result.inserted_id
        return job

    async def get_job(self, job_id: str) -> Job | None:
        try:
            object_id = ObjectId(job_id)
        except (InvalidId, TypeError):
            return None
        record = await self.collection.find_one({"_id": object_id})
        return Job(**record) if record else None

    async def list_user_jobs(self, user_object_id: ObjectId, limit: int = 10) -> list[Job]:
        cursor = self.collection.find({"job_user_id": user_object_id}).sort("created_at", -1)
        return [Job(**doc) for doc in await cursor.limit(limit).to_list(length=None)]

    async def update_job(self, job_id: ObjectId, **fields) -> None:
        fields["updated_at"] = self.now()
        await self.collection.update_one({"_id": job_id}, {"$set": to_bson_value(fields)})

    async def add_result(self, job_id: ObjectId, result: JobVideoResult) -> None:
        await self.collection.update_one(
            {"_id": job_id},
            {
                "$push": {"results": to_bson_value(result.model_dump())},
                "$set": {"updated_at": self.now()},
            },
        )

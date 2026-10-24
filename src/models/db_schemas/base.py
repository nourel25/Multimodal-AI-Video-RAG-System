from datetime import UTC, datetime
from enum import Enum
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_bson_value(value: Any) -> Any:
    """Recursively convert enums (which BSON can't encode) to their raw values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: to_bson_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_bson_value(item) for item in value]
    return value


class MongoDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: ObjectId | None = Field(None, alias="_id")

    def to_mongo(self) -> dict:
        return to_bson_value(self.model_dump(by_alias=True, exclude_none=True))

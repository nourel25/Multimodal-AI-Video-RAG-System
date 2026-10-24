import re
from datetime import datetime

from pydantic import Field, field_validator

from .base import MongoDocument, utc_now

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,64}$")


class User(MongoDocument):
    user_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        if not USER_ID_PATTERN.match(value):
            raise ValueError("user_id must be 3-64 letters, numbers or underscores")
        return value

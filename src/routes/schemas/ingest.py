from pydantic import BaseModel, HttpUrl

class IngestRequest(BaseModel):
    youtube_urls: list[HttpUrl]

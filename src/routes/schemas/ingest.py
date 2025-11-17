from pydantic import BaseModel, HttpUrl
import re

class IngestRequest(BaseModel):
    youtube_url: HttpUrl

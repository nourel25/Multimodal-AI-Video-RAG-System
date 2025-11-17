from pydantic import BaseModel
from typing import Optional
from bson import ObjectId

class ProcessRequest(BaseModel):
    youtube_url: str
    do_reset: Optional[int] = 0
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
     

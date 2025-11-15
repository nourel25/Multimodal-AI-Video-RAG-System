from pydantic import BaseModel, HttpUrl
from typing import Optional

class IngestRequest(BaseModel):
    url: HttpUrl
    do_reset: Optional[int] = 0
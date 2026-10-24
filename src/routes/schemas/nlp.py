from typing import Literal

from pydantic import BaseModel, Field

from helpers.youtube import format_timestamp, watch_url
from models.db_schemas import RetrievedDocument
from models.enums import Modality
from stores.vectordb.VectorDBEnums import SearchMode


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(5, ge=1, le=20)
    mode: SearchMode | None = Field(None, description="dense | sparse | hybrid")
    rerank: bool | None = None
    video_ids: list[str] | None = Field(None, description="Restrict search to these videos")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class AnswerRequest(SearchRequest):
    language: str | None = Field(None, description="Prompt language; detected when omitted")
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class SourceOut(BaseModel):
    index: int
    chunk_id: str
    video_id: str
    youtube_video_id: str
    video_title: str
    text: str
    modality: Modality
    start: float
    end: float
    timestamp: str
    url: str
    score: float

    @classmethod
    def from_document(cls, index: int, doc: RetrievedDocument) -> "SourceOut":
        return cls(
            index=index,
            timestamp=format_timestamp(doc.start),
            url=watch_url(doc.youtube_video_id, doc.start),
            **doc.model_dump(),
        )


def to_sources(documents: list[RetrievedDocument]) -> list[SourceOut]:
    return [SourceOut.from_document(i + 1, doc) for i, doc in enumerate(documents)]


class SearchResponse(BaseModel):
    signal: str
    results: list[SourceOut]


class AnswerResponse(BaseModel):
    signal: str
    answer: str
    language: str
    sources: list[SourceOut]


class RebuildIndexResponse(BaseModel):
    signal: str
    indexed_chunks: int

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from stores.embedding.EmbeddingInterface import SparseVector

from .VectorDBEnums import SearchMode


@dataclass
class VectorRecord:
    id: str | int
    dense: list[float]
    sparse: SparseVector | None = None
    payload: dict = field(default_factory=dict)


@dataclass
class ScoredRecord:
    id: str | int
    score: float
    payload: dict


class VectorDBInterface(ABC):
    @abstractmethod
    async def connect(self): ...

    @abstractmethod
    async def disconnect(self): ...

    @abstractmethod
    async def is_collection_existed(self, collection_name: str) -> bool: ...

    @abstractmethod
    async def list_all_collections(self) -> list[str]: ...

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool: ...

    @abstractmethod
    async def create_collection(self, collection_name: str, embedding_size: int,
                                do_reset: bool = False) -> bool: ...

    @abstractmethod
    async def upsert_records(self, collection_name: str, records: list[VectorRecord],
                             batch_size: int = 64) -> int: ...

    @abstractmethod
    async def delete_by_video(self, collection_name: str, video_id: str) -> None: ...

    @abstractmethod
    async def count(self, collection_name: str) -> int: ...

    @abstractmethod
    async def search(self, collection_name: str, *, dense_vector: list[float] | None,
                     sparse_vector: SparseVector | None, limit: int,
                     mode: SearchMode = SearchMode.HYBRID,
                     video_ids: list[str] | None = None) -> list[ScoredRecord]: ...

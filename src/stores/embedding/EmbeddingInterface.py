from abc import ABC, abstractmethod
from dataclasses import dataclass


class EmbeddingInterface(ABC):
    size: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class RerankerInterface(ABC):
    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document; higher is more relevant."""


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

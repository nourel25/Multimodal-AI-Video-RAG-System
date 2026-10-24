from enum import Enum, StrEnum


class VectorDBEnums(Enum):
    QDRANT = "QDRANT"


class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"


class SearchMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

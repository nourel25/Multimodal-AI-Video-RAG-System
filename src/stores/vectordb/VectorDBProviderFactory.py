from helpers.config import Settings

from .providers import QdrantDBProvider
from .VectorDBEnums import VectorDBEnums
from .VectorDBInterface import VectorDBInterface


class VectorDBProviderFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create(self, provider: str) -> VectorDBInterface:
        s = self.settings
        if provider.upper() == VectorDBEnums.QDRANT.value:
            if s.QDRANT_URL:
                return QdrantDBProvider(url=s.QDRANT_URL, api_key=s.QDRANT_API_KEY,
                                        distance_method=s.VECTOR_DB_DISTANCE_METHOD)
            db_path = s.assets_path / "database" / s.VECTOR_DB_PATH
            db_path.mkdir(parents=True, exist_ok=True)
            return QdrantDBProvider(path=str(db_path),
                                    distance_method=s.VECTOR_DB_DISTANCE_METHOD)

        raise ValueError(f"Unsupported vector DB backend: {provider!r}")

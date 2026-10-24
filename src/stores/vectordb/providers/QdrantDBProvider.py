import logging

from qdrant_client import AsyncQdrantClient, models

from stores.embedding.EmbeddingInterface import SparseVector

from ..VectorDBEnums import DistanceMethodEnums, SearchMode
from ..VectorDBInterface import ScoredRecord, VectorDBInterface, VectorRecord

DENSE = "dense"
SPARSE = "sparse"

_DISTANCES = {
    DistanceMethodEnums.COSINE.value: models.Distance.COSINE,
    DistanceMethodEnums.DOT.value: models.Distance.DOT,
}


class QdrantDBProvider(VectorDBInterface):
    """Qdrant with a named dense vector and a BM25-style sparse vector per point.

    Hybrid search runs both queries as prefetches and fuses them with Reciprocal Rank Fusion
    inside Qdrant, so only one round trip is needed.
    """

    def __init__(self, *, url: str | None = None, path: str | None = None,
                 api_key: str | None = None, distance_method: str = "cosine"):
        if not url and not path:
            raise ValueError("QdrantDBProvider needs either a server url or a local path")
        self.url = url
        self.path = path
        self.api_key = api_key or None
        self.distance = _DISTANCES.get(distance_method, models.Distance.COSINE)
        self.client: AsyncQdrantClient | None = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        if self.url:
            self.client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        elif self.path == ":memory:":
            self.client = AsyncQdrantClient(location=":memory:")
        else:
            self.client = AsyncQdrantClient(path=self.path)

    async def disconnect(self):
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def is_collection_existed(self, collection_name: str) -> bool:
        return await self.client.collection_exists(collection_name=collection_name)

    async def list_all_collections(self) -> list[str]:
        response = await self.client.get_collections()
        return [collection.name for collection in response.collections]

    async def delete_collection(self, collection_name: str) -> bool:
        if await self.is_collection_existed(collection_name):
            return await self.client.delete_collection(collection_name=collection_name)
        return False

    async def create_collection(self, collection_name: str, embedding_size: int,
                                do_reset: bool = False) -> bool:
        if do_reset:
            await self.delete_collection(collection_name)
        if await self.is_collection_existed(collection_name):
            return False

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config={DENSE: models.VectorParams(size=embedding_size,
                                                       distance=self.distance)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF)},
        )
        if self.url:  # payload indexes only exist on a Qdrant server, not the embedded mode
            await self.client.create_payload_index(
                collection_name=collection_name, field_name="video_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        return True

    async def upsert_records(self, collection_name: str, records: list[VectorRecord],
                             batch_size: int = 64) -> int:
        for i in range(0, len(records), batch_size):
            points = []
            for record in records[i:i + batch_size]:
                vector: dict = {DENSE: record.dense}
                if record.sparse is not None and record.sparse.indices:
                    vector[SPARSE] = models.SparseVector(
                        indices=record.sparse.indices, values=record.sparse.values)
                points.append(models.PointStruct(id=record.id, vector=vector,
                                                 payload=record.payload))
            await self.client.upsert(collection_name=collection_name, points=points)
        return len(records)

    async def delete_by_video(self, collection_name: str, video_id: str) -> None:
        if not await self.is_collection_existed(collection_name):
            return
        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=self._video_filter([video_id])),
        )

    async def count(self, collection_name: str) -> int:
        if not await self.is_collection_existed(collection_name):
            return 0
        return (await self.client.count(collection_name=collection_name)).count

    @staticmethod
    def _video_filter(video_ids: list[str] | None) -> models.Filter | None:
        if not video_ids:
            return None
        return models.Filter(must=[
            models.FieldCondition(key="video_id", match=models.MatchAny(any=video_ids))
        ])

    async def search(self, collection_name: str, *, dense_vector: list[float] | None,
                     sparse_vector: SparseVector | None, limit: int,
                     mode: SearchMode = SearchMode.HYBRID,
                     video_ids: list[str] | None = None) -> list[ScoredRecord]:
        if not await self.is_collection_existed(collection_name):
            return []

        query_filter = self._video_filter(video_ids)
        sparse_query = (
            models.SparseVector(indices=sparse_vector.indices, values=sparse_vector.values)
            if sparse_vector is not None and sparse_vector.indices else None
        )
        # A query with no indexable terms (e.g. only stopwords) degrades to dense search.
        if mode != SearchMode.DENSE and sparse_query is None:
            mode = SearchMode.DENSE

        if mode == SearchMode.DENSE and dense_vector is None:
            return []  # sparse-only query with no indexable terms: nothing to rank by
        if mode == SearchMode.DENSE:
            response = await self.client.query_points(
                collection_name=collection_name, query=dense_vector, using=DENSE,
                query_filter=query_filter, limit=limit, with_payload=True,
            )
        elif mode == SearchMode.SPARSE:
            response = await self.client.query_points(
                collection_name=collection_name, query=sparse_query, using=SPARSE,
                query_filter=query_filter, limit=limit, with_payload=True,
            )
        else:
            prefetch_limit = max(limit * 2, 20)
            response = await self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vector, using=DENSE, filter=query_filter,
                                    limit=prefetch_limit),
                    models.Prefetch(query=sparse_query, using=SPARSE, filter=query_filter,
                                    limit=prefetch_limit),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit, with_payload=True,
            )

        return [
            ScoredRecord(id=str(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]

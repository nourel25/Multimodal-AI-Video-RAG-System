import logging

from .EmbeddingInterface import EmbeddingInterface, RerankerInterface

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(EmbeddingInterface):
    """Local dense embeddings. The default (BAAI/bge-m3) is multilingual, so Arabic and
    English queries share one vector space."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16,
                 query_prefix: str = "", document_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.size = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


class CrossEncoderReranker(RerankerInterface):
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s on %s", model_name, device)
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self.model.predict([(query, doc) for doc in documents], show_progress_bar=False)
        return [float(score) for score in scores]

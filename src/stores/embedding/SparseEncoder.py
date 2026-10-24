import zlib
from collections import Counter

from helpers.text import tokenize

from .EmbeddingInterface import SparseVector


class BM25SparseEncoder:
    """BM25-style sparse vectors for keyword matching in Qdrant.

    Documents carry the BM25 term-frequency component; Qdrant applies IDF at query time
    (the collection's sparse vector uses `Modifier.IDF`), so the combined score is BM25.
    Terms are hashed to indices, which keeps the encoder stateless — no vocabulary to persist.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_doc_length: float = 120.0):
        self.k1 = k1
        self.b = b
        self.avg_doc_length = avg_doc_length

    @staticmethod
    def _index(term: str) -> int:
        return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF

    def _vector(self, weights: dict[int, float]) -> SparseVector:
        items = sorted(weights.items())
        return SparseVector(indices=[i for i, _ in items], values=[v for _, v in items])

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        length_norm = 1 - self.b + self.b * len(tokens) / self.avg_doc_length
        weights: dict[int, float] = {}
        for term, tf in Counter(tokens).items():
            index = self._index(term)
            weights[index] = weights.get(index, 0.0) + (
                tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
            )
        return self._vector(weights)

    def encode_query(self, text: str) -> SparseVector:
        return self._vector({self._index(term): 1.0 for term in set(tokenize(text))})

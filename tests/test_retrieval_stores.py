import pytest

from stores.embedding.SparseEncoder import BM25SparseEncoder
from stores.templates.template_parser import TemplateParser
from stores.vectordb.providers import QdrantDBProvider
from stores.vectordb.VectorDBEnums import SearchMode
from stores.vectordb.VectorDBInterface import VectorRecord
from tests.fakes import FakeEmbedder

DOCS = {
    "11111111-1111-1111-1111-111111111111": ("v1", "Pydantic validates request data"),
    "22222222-2222-2222-2222-222222222222": ("v1", "Starlette powers the web layer"),
    "33333333-3333-3333-3333-333333333333": ("v2", "غزوة تبوك كانت آخر غزوة"),
}


def test_sparse_encoder_query_hits_document_terms():
    encoder = BM25SparseEncoder()
    doc = encoder.encode_document("Pydantic validates data, pydantic is fast")
    query = encoder.encode_query("pydantic")
    assert set(query.indices) <= set(doc.indices)
    # Term frequency saturates: repeating a word helps less than linearly.
    weight = dict(zip(doc.indices, doc.values, strict=True))[query.indices[0]]
    assert 1.0 < weight < 2.2


def test_sparse_encoder_matches_arabic_spelling_variants():
    encoder = BM25SparseEncoder()
    assert encoder.encode_query("غزوة").indices == encoder.encode_query("غزوه").indices


@pytest.fixture
async def vectordb():
    db = QdrantDBProvider(path=":memory:")
    await db.connect()
    embedder, sparse = FakeEmbedder(), BM25SparseEncoder()
    await db.create_collection("c", embedder.size)
    await db.upsert_records("c", [
        VectorRecord(id=pid, dense=embedder.embed_query(text),
                     sparse=sparse.encode_document(text), payload={"video_id": vid, "text": text})
        for pid, (vid, text) in DOCS.items()
    ])
    yield db, embedder, sparse
    await db.disconnect()


@pytest.mark.parametrize("mode", list(SearchMode))
async def test_every_search_mode_finds_the_matching_document(vectordb, mode):
    db, embedder, sparse = vectordb
    query = "how does pydantic validate data"
    results = await db.search("c", dense_vector=embedder.embed_query(query),
                              sparse_vector=sparse.encode_query(query), limit=2, mode=mode)
    assert results[0].payload["text"] == "Pydantic validates request data"


async def test_search_can_be_restricted_to_videos(vectordb):
    db, embedder, sparse = vectordb
    query = "pydantic"
    results = await db.search("c", dense_vector=embedder.embed_query(query),
                              sparse_vector=sparse.encode_query(query), limit=5,
                              video_ids=["v2"])
    assert {r.payload["video_id"] for r in results} == {"v2"}


async def test_delete_by_video_and_reset(vectordb):
    db, embedder, _ = vectordb
    await db.delete_by_video("c", "v1")
    assert await db.count("c") == 1
    assert await db.create_collection("c", embedder.size) is False  # exists, untouched
    assert await db.create_collection("c", embedder.size, do_reset=True) is True
    assert await db.count("c") == 0


async def test_search_missing_collection_returns_empty(vectordb):
    db, embedder, _ = vectordb
    assert await db.search("nope", dense_vector=embedder.embed_query("x"), sparse_vector=None,
                           limit=3) == []


def test_template_parser_falls_back_to_default_language():
    parser = TemplateParser(language="fr", default_language="en")
    assert parser.language == "en"
    assert "[3]" in parser.get("rag", "document_prompt", {
        "doc_num": 3, "modality": "spoken", "video_title": "T", "timestamp": "00:10",
        "chunk_text": "hello"})
    assert parser.get("rag", "missing_key") is None
    assert "السؤال" in parser.get("rag", "footer_prompt", {"query": "q"}, language="ar")

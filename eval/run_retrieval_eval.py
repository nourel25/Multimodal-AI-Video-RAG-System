"""Compare retrieval configurations on the labelled question set.

Indexes eval/corpus with the app's own chunker, embedder, sparse encoder and Qdrant provider
(in memory), then reports Hit@k / MRR for dense, sparse, hybrid and hybrid + rerank.

    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --chunk-chars 400 --no-rerank
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from controllers.ChunkController import ChunkController  # noqa: E402
from eval.metrics import GoldWindow, Hit, first_relevant_rank, summarize  # noqa: E402
from helpers.config import Settings  # noqa: E402
from stores.embedding.SentenceTransformerEmbedding import (  # noqa: E402
    CrossEncoderReranker,
    SentenceTransformerEmbedding,
)
from stores.embedding.SparseEncoder import BM25SparseEncoder  # noqa: E402
from stores.transcription import TranscriptSegment  # noqa: E402
from stores.vectordb.providers import QdrantDBProvider  # noqa: E402
from stores.vectordb.VectorDBEnums import SearchMode  # noqa: E402
from stores.vectordb.VectorDBInterface import VectorRecord  # noqa: E402

CORPUS_DIR = ROOT / "eval" / "corpus"
QUESTIONS_FILE = ROOT / "eval" / "datasets" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
COLLECTION = "eval"
TOP_K = 10


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def build_index(settings: Settings, embedder, sparse, chunk_chars: int, overlap: int):
    db = QdrantDBProvider(path=":memory:")
    await db.connect()
    await db.create_collection(COLLECTION, embedder.size)

    chunker = ChunkController()
    records, n_chunks = [], 0
    for path in sorted(CORPUS_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        segments = [TranscriptSegment(**s) for s in doc["segments"]]
        chunks = chunker.build_chunks(segments, max_chars=chunk_chars, overlap_segments=overlap)
        texts = [c.text for c in chunks]
        dense = embedder.embed_documents(texts)
        for i, chunk in enumerate(chunks):
            records.append(VectorRecord(
                id=n_chunks + i, dense=dense[i], sparse=sparse.encode_document(chunk.text),
                payload={"video_id": doc["youtube_video_id"], "text": chunk.text,
                         "start": chunk.start, "end": chunk.end},
            ))
        n_chunks += len(chunks)
    await db.upsert_records(COLLECTION, records)
    return db, n_chunks


async def evaluate(db, embedder, sparse, reranker, questions, mode: SearchMode, rerank: bool,
                   candidates: int):
    ranks, latencies, per_language = [], [], defaultdict(list)
    for q in questions:
        started = time.perf_counter()
        limit = candidates if rerank else TOP_K
        results = await db.search(
            COLLECTION,
            dense_vector=embedder.embed_query(q["question"]) if mode != SearchMode.SPARSE
            else None,
            sparse_vector=sparse.encode_query(q["question"]), limit=limit, mode=mode,
        )
        if rerank and results:
            scores = reranker.score(q["question"], [r.payload["text"] for r in results])
            results = [r for _, r in sorted(zip(scores, results, strict=True),
                                            key=lambda pair: pair[0], reverse=True)]
        latencies.append((time.perf_counter() - started) * 1000)

        hits = [Hit(r.payload["video_id"], r.payload["start"], r.payload["end"])
                for r in results[:TOP_K]]
        rank = first_relevant_rank(hits, [GoldWindow(**w) for w in q["windows"]])
        ranks.append(rank)
        per_language[q["language"]].append(rank)
        for tag in q.get("tags", []):
            per_language[tag].append(rank)

    return {
        **summarize(ranks),
        "latency_ms": mean(latencies),
        "by_language": {lang: summarize(r) for lang, r in per_language.items()},
        "misses": [q["id"] for q, r in zip(questions, ranks, strict=True) if r is None or r > 5],
    }


def to_markdown(rows: list[tuple[str, dict]], meta: dict) -> str:
    lines = [
        f"Questions: {meta['questions']} ({meta['languages']}) over {meta['videos']} videos, "
        f"{meta['chunks']} chunks of <= {meta['chunk_chars']} chars. "
        f"Embedding: `{meta['embedding_model']}`, reranker: `{meta['reranker']}`.",
        "",
        "| Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Arabic Hit@5 | English Hit@5 "
        "| Latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rows:
        ar = r["by_language"].get("ar", {}).get("hit@5")
        en = r["by_language"].get("en", {}).get("hit@5")
        fmt = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {name} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | "
                     f"{r['mrr@10']:.2f} | {fmt(ar)} | {fmt(en)} | {r['latency_ms']:.0f} |")
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--tag", default="", help="suffix for the results file names")
    args = parser.parse_args()

    settings = Settings(_env_file=str(ROOT / "src" / ".env"))
    chunk_chars = args.chunk_chars or settings.CHUNK_MAX_CHARS
    overlap = settings.CHUNK_OVERLAP_SEGMENTS if args.overlap is None else args.overlap

    questions = load_questions()
    embedder = SentenceTransformerEmbedding(
        settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE,
        query_prefix=settings.EMBEDDING_QUERY_PREFIX,
        document_prefix=settings.EMBEDDING_DOCUMENT_PREFIX)
    sparse = BM25SparseEncoder()
    reranker = None if args.no_rerank else CrossEncoderReranker(settings.RERANKER_MODEL,
                                                                device=settings.EMBEDDING_DEVICE)

    db, n_chunks = await build_index(settings, embedder, sparse, chunk_chars, overlap)
    configs = [("Dense (bge-m3)", SearchMode.DENSE, False),
               ("Sparse (BM25)", SearchMode.SPARSE, False),
               ("Hybrid (RRF)", SearchMode.HYBRID, False)]
    if reranker:
        configs.append(("Hybrid + rerank", SearchMode.HYBRID, True))

    rows = []
    for name, mode, rerank in configs:
        result = await evaluate(db, embedder, sparse, reranker, questions, mode, rerank,
                                settings.RETRIEVAL_CANDIDATES)
        rows.append((name, result))
        print(f"{name:18s} hit@1={result['hit@1']:.2f} hit@5={result['hit@5']:.2f} "
              f"mrr={result['mrr@10']:.2f} misses={result['misses']}")
    await db.disconnect()

    languages = defaultdict(int)
    for q in questions:
        languages[q["language"]] += 1
    meta = {
        "questions": len(questions),
        "languages": ", ".join(f"{n} {lang}" for lang, n in sorted(languages.items())),
        "videos": len(list(CORPUS_DIR.glob("*.json"))), "chunks": n_chunks,
        "chunk_chars": chunk_chars, "embedding_model": settings.EMBEDDING_MODEL,
        "reranker": settings.RERANKER_MODEL if reranker else "none",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    (RESULTS_DIR / f"retrieval{suffix}.md").write_text(to_markdown(rows, meta), encoding="utf-8")
    (RESULTS_DIR / f"retrieval{suffix}.json").write_text(
        json.dumps({"meta": meta, "results": dict(rows)}, indent=2), encoding="utf-8")
    print(to_markdown(rows, meta))


if __name__ == "__main__":
    asyncio.run(main())

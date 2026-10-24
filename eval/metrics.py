"""Retrieval metrics where relevance is defined by time overlap with gold windows.

Labelling questions with (video, start, end) windows instead of chunk ids keeps the dataset
valid when chunk sizes, overlap or the transcript itself change. A question may have several
windows when more than one video answers it.
"""

from dataclasses import dataclass
from statistics import mean


@dataclass
class GoldWindow:
    video_id: str
    start: float
    end: float


@dataclass
class Hit:
    video_id: str
    start: float
    end: float


def is_relevant(hit: Hit, gold: GoldWindow, min_overlap: float = 1.0) -> bool:
    if hit.video_id != gold.video_id:
        return False
    overlap = min(hit.end, gold.end) - max(hit.start, gold.start)
    # Very short gold windows count if the chunk fully contains them.
    return overlap > 0 and overlap >= min(min_overlap, gold.end - gold.start)


def first_relevant_rank(hits: list[Hit], golds: list[GoldWindow]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if any(is_relevant(hit, gold) for gold in golds):
            return rank
    return None


def summarize(ranks: list[int | None], ks=(1, 3, 5), mrr_cutoff: int = 10) -> dict:
    """Hit@k (share of questions with a relevant chunk in the top k) and MRR@cutoff."""
    if not ranks:
        return {}
    summary = {f"hit@{k}": mean(1.0 if r is not None and r <= k else 0.0 for r in ranks)
               for k in ks}
    summary[f"mrr@{mrr_cutoff}"] = mean(
        1.0 / r if r is not None and r <= mrr_cutoff else 0.0 for r in ranks)
    return summary

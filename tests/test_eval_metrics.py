import pytest
from eval.metrics import GoldWindow, Hit, first_relevant_rank, is_relevant, summarize

GOLD = GoldWindow("v1", 30.0, 40.0)


@pytest.mark.parametrize(("hit", "expected"), [
    (Hit("v1", 25.0, 35.0), True),    # partial overlap
    (Hit("v1", 0.0, 100.0), True),    # chunk contains the window
    (Hit("v1", 39.5, 50.0), False),   # touches by less than a second
    (Hit("v1", 40.0, 50.0), False),   # adjacent, no overlap
    (Hit("v2", 30.0, 40.0), False),   # other video
])
def test_is_relevant(hit, expected):
    assert is_relevant(hit, GOLD) is expected


def test_short_gold_window_inside_chunk_is_relevant():
    assert is_relevant(Hit("v1", 0.0, 20.0), GoldWindow("v1", 10.0, 10.5))


def test_first_relevant_rank_and_summary():
    hits = [Hit("v2", 0, 10), Hit("v1", 31, 33), Hit("v1", 35, 45)]
    assert first_relevant_rank(hits, [GOLD]) == 2
    assert first_relevant_rank(hits[:1], [GOLD]) is None
    # Any of several gold windows counts (questions answered by more than one video).
    assert first_relevant_rank(hits, [GoldWindow("v2", 5, 8), GOLD]) == 1

    summary = summarize([1, 2, None, 4])
    assert summary["hit@1"] == 0.25
    assert summary["hit@3"] == 0.5
    assert summary["hit@5"] == 0.75
    assert summary["mrr@10"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)

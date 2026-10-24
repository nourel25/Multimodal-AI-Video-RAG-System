from helpers.citations import StreamingCitationNormalizer, normalize_citations


def test_normalizes_fullwidth_citations():
    assert normalize_citations("VMs are bulky【1†L1-L5】【2†L1-L4】.") == "VMs are bulky[1][2]."
    assert normalize_citations("Already fine [3].") == "Already fine [3]."


def test_streaming_normalizer_handles_markers_split_across_tokens():
    tokens = ["Containers share", " a kernel", "【", "2", "†L1", "-L4】", " and start fast", "【1"]
    normalizer = StreamingCitationNormalizer()
    out = "".join(normalizer.feed(t) for t in tokens) + normalizer.flush()
    assert out == "Containers share a kernel[2] and start fast【1"  # unclosed tail kept as-is
    assert normalizer.flush() == ""

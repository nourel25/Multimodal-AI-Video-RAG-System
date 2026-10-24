"""Normalise model-specific citation markers to the plain [n] format the UI links to.

Some models ignore the prompt's format: gpt-oss, for example, writes 【1†L1-L5】.
"""

import re

_FULLWIDTH_CITATION = re.compile(r"【\s*(\d+)[^】]*】")


def normalize_citations(text: str) -> str:
    return _FULLWIDTH_CITATION.sub(r"[\1]", text)


class StreamingCitationNormalizer:
    """Applies normalize_citations to a token stream where a marker may span several tokens.

    Text is passed through immediately, except from an unclosed "【" onward, which is held
    back until its "】" arrives (or the stream ends).
    """

    def __init__(self):
        self._pending = ""

    def feed(self, token: str) -> str:
        text = self._pending + token
        start = text.rfind("【")
        if start != -1 and "】" not in text[start:]:
            self._pending = text[start:]
            text = text[:start]
        else:
            self._pending = ""
        return normalize_citations(text)

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return normalize_citations(text)

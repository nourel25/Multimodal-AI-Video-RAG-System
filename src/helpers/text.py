"""Language-aware text normalisation shared by the sparse (BM25-style) encoder."""

import re

_ARABIC_DIACRITICS = re.compile("[ً-ْٰـ]")  # tashkeel + tatweel
_ALEF_VARIANTS = re.compile("[آأإٱ]")  # آ أ إ ٱ
_TOKEN = re.compile(r"\w+", re.UNICODE)
_ARABIC_LETTER = re.compile("[؀-ۿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")

_STOPWORDS_TEXT = """
a an and are as at be but by for from has have he her his i if in into is it its me my no not
of on or our she so that the their them then there these they this to was we were what when
where which who why will with you your do does did can could would should about just than
في من على الى إلى عن ان أن إن هذا هذه ذلك تلك التي الذي الذين هو هي هم انا أنا نحن انت أنت
كان كانت يكون ما لا لم لن قد ثم او أو و يا مع كل بعض عند حتى اذا إذا لكن بل هل
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def normalize(text: str) -> str:
    text = text.lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")  # ى→ي, ة→ه


_NORMALIZED_STOPWORDS = frozenset(normalize(word) for word in STOPWORDS)


def detect_language(text: str) -> str:
    """Cheap script-based guess between the two supported prompt languages."""
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    return "ar" if arabic > latin else "en"


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN.findall(normalize(text))
        if len(token) > 1 and token not in _NORMALIZED_STOPWORDS
    ]

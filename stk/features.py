"""What the edge sends instead of text: indices and counts of hashed word and character pieces.

The text is masked first (stk/pii.py), then hashed here, and the words are dropped. The hashing is public: the
numbers are not sentences, but someone holding them could test whether a given word appears, which is why the
masking runs first. STK's servers read exactly this payload."""
from __future__ import annotations
from sklearn.feature_extraction.text import HashingVectorizer

N18 = 2 ** 18
_HW18 = HashingVectorizer(ngram_range=(1, 2), n_features=N18, alternate_sign=False, norm=None)
_HC18 = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=N18, alternate_sign=False, norm=None)


def hash_text(text: str) -> dict:
    """Indices and counts of word pieces (1-2 words) and character pieces (3-5 letters), for one masked text."""
    w = _HW18.transform([text]); c = _HC18.transform([text])
    return {"v": 1, "w": w.indices.tolist(), "wc": w.data.astype(int).tolist(),
            "c": c.indices.tolist(), "cc": c.data.astype(int).tolist()}

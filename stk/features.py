"""What the edge sends instead of text: indices and counts of hashed word and character pieces.

The text is masked first (stk/pii.py), then hashed here, and the words are dropped. The hashing itself is public, so
with STK_FEATURE_KEY (your center's feature key, from GET /v1/keys/feature with the admin key) the positions are
also moved by your own secret permutation before they leave: without the key, the numbers cannot be checked against
guessed words. Without STK_FEATURE_KEY, STK applies the same key the moment the numbers arrive."""
from __future__ import annotations
import hmac, os
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

N18 = 2 ** 18
_HW18 = HashingVectorizer(ngram_range=(1, 2), n_features=N18, alternate_sign=False, norm=None)
_HC18 = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=N18, alternate_sign=False, norm=None)


def hash_text(text: str) -> dict:
    """Indices and counts of word pieces (1-2 words) and character pieces (3-5 letters), for one masked text."""
    w = _HW18.transform([text]); c = _HC18.transform([text])
    return {"v": 1, "w": w.indices.tolist(), "wc": w.data.astype(int).tolist(),
            "c": c.indices.tolist(), "cc": c.data.astype(int).tolist()}


class Keyed:
    """The same permutation STK's servers use (4-round Feistel over the low 16 bits, HMAC-SHA256 round functions)."""
    def __init__(self, key: bytes):
        self.F = {kind: [np.array([hmac.digest(key, b"stk-feat|%s|%d|%d" % (kind, r, x), "sha256")[0] for x in range(256)],
                                  dtype=np.int64) for r in range(4)] for kind in (b"w", b"c")}

    def _perm(self, a, kind: bytes):
        a = np.asarray(a, dtype=np.int64)
        hi, L, R = a >> 16, (a >> 8) & 255, a & 255
        for r in range(4):
            L, R = R, L ^ self.F[kind][r][R]
        return ((hi << 16) | (L << 8) | R).tolist()

    def payload(self, p: dict) -> dict:
        return {"v": 1, "w": self._perm(p["w"], b"w"), "wc": p["wc"], "c": self._perm(p["c"], b"c"), "cc": p["cc"], "k": 1}


_KEY = os.environ.get("STK_FEATURE_KEY")
KEYED = Keyed(bytes.fromhex(_KEY)) if _KEY else None


def features(text: str) -> dict:
    """hash_text, then your own key when STK_FEATURE_KEY is set."""
    p = hash_text(text)
    return KEYED.payload(p) if KEYED else p

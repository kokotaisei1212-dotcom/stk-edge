"""Features: how a conversation is turned into numbers, and why STK's servers never need the words.

A customer's first messages become hashed counts of word pieces (1-2 words) and character pieces (3-5 letters
inside words), in a fixed numbering of 2^18 each. Hashing is one-way: the numbers cannot be turned back into the
sentence. The same function runs wherever the text lives (the edge shim in the customer's cloud, `stk.edge`), so
what reaches STK is a list of numbers; or on STK's side, from masked text, for centers that prefer to send text.

The numbering folds exactly onto the 2^16 numbering of the pooled judge (index mod 2^16), so one payload serves
both the center's own model and the shared second opinion.

Two forms of the same payload: v1 (plain lists: the wire form the edge sends) and v2 (the same arrays packed as
base64, 7x smaller: the form STK keeps in memory and in state files). `pack` turns v1 into v2; `rows` reads both.

An item is a dict with either "text" (masked) or "feat" (this payload) and a stable "key" (used for the exam split,
the self-judge folds and duplicate detection). `item_key` gives the key for either kind; identical texts give
identical keys in either form."""
from __future__ import annotations
import base64, hashlib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

N18 = 2 ** 18
N16 = 2 ** 16
_HW18 = HashingVectorizer(ngram_range=(1, 2), n_features=N18, alternate_sign=False, norm=None)
_HC18 = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=N18, alternate_sign=False, norm=None)
_FOLD = sp.csr_matrix((np.ones(N18, dtype=np.float32), (np.arange(N18), np.arange(N18) % N16)), shape=(N18, N16))
CHAR_MIN_DF = 2  # character pieces seen in fewer conversations than this are ignored (as the old char vectorizer did)
_EMPTY = np.zeros(0, dtype=np.int64)


def hash_text(text: str) -> dict:
    """The payload the edge sends instead of text: indices and counts of word and character pieces (v1, lists)."""
    w = _HW18.transform([text]); c = _HC18.transform([text])
    return {"v": 1, "w": w.indices.tolist(), "wc": w.data.astype(int).tolist(),
            "c": c.indices.tolist(), "cc": c.data.astype(int).tolist()}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _unb64(s: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s, validate=True), dtype=dtype).astype(np.int64)


def pack(p: dict) -> dict:
    """v2: the same payload with each array packed (uint32 indices, uint16 counts, base64). Idempotent."""
    if p.get("v") == 2:
        return p
    w, wc, c, cc = unpack(p)
    return {"v": 2, "w": _b64(w.astype(np.uint32)), "wc": _b64(wc.astype(np.uint16)),
            "c": _b64(c.astype(np.uint32)), "cc": _b64(cc.astype(np.uint16))}


def unpack(p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(word indices, word counts, char indices, char counts) as int64 arrays, from either form."""
    if p.get("v") == 2:
        return (_unb64(p["w"], np.uint32), _unb64(p["wc"], np.uint16), _unb64(p["c"], np.uint32), _unb64(p["cc"], np.uint16))
    g = lambda k: np.asarray(p.get(k) or [], dtype=np.int64) if p.get(k) else _EMPTY
    return g("w"), g("wc"), g("c"), g("cc")


def payload_key(p: dict) -> str:
    """A stable key for a payload: identical texts give identical keys (in either form), and nothing else can be read from it."""
    h = hashlib.sha1()
    for k, a in zip(("w", "wc", "c", "cc"), unpack(p)):
        h.update(k.encode()); h.update(a.tobytes())
    return "f:" + h.hexdigest()[:32]


def item_key(x: dict) -> str:
    k = x.get("key")
    if k:
        return k
    if "text" in x and x["text"] is not None:
        return x["text"]
    return payload_key(x["feat"])


def with_key(x: dict) -> dict:
    if "key" not in x:
        x["key"] = item_key(x)
    return x


def _valid(p: dict) -> bool:
    try:
        if p.get("v") == 2 and not all(isinstance(p[k], str) for k in ("w", "wc", "c", "cc")):
            return False
        w, wc, c, cc = unpack(p)
        return (len(w) == len(wc) and len(c) == len(cc)
                and (len(w) == 0 or (w.min() >= 0 and w.max() < N18 and wc.min() > 0))
                and (len(c) == 0 or (c.min() >= 0 and c.max() < N18 and cc.min() > 0)))
    except Exception:
        return False


def _csr(parts: list[tuple[np.ndarray, np.ndarray]], n: int) -> sp.csr_matrix:
    lens = np.fromiter((len(i) for i, _ in parts), dtype=np.int64, count=len(parts))
    indptr = np.zeros(n + 1, dtype=np.int64); np.cumsum(lens, out=indptr[1:])
    idx = np.concatenate([i for i, _ in parts]) if parts else _EMPTY
    dat = np.concatenate([d for _, d in parts]).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    M = sp.csr_matrix((dat, idx, indptr), shape=(n, N18))
    M.sum_duplicates()
    return M


def rows(items: list[dict]) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Hashed counts (2^18 word, 2^18 char) for a list of items of either kind."""
    n = len(items)
    empty = lambda: sp.csr_matrix((0, N18), dtype=np.float32)
    if n == 0:
        return empty(), empty()
    text_idx = [i for i, x in enumerate(items) if "feat" not in x]
    feat_idx = [i for i, x in enumerate(items) if "feat" in x]
    Ws, Cs = [], []
    if text_idx:
        texts = [items[i]["text"] or "" for i in text_idx]
        Ws.append(_HW18.transform(texts).tocsr().astype(np.float32)); Cs.append(_HC18.transform(texts).tocsr().astype(np.float32))
    if feat_idx:
        un = [unpack(items[i]["feat"]) for i in feat_idx]
        Ws.append(_csr([(w, wc) for w, wc, _, _ in un], len(feat_idx)))
        Cs.append(_csr([(c, cc) for _, _, c, cc in un], len(feat_idx)))
    W = sp.vstack(Ws).tocsr(); C = sp.vstack(Cs).tocsr()
    if text_idx and feat_idx:
        perm = np.argsort(np.asarray(text_idx + feat_idx))
        W, C = W[perm], C[perm]
    return W, C


def _log_norm(X):
    X = X.copy().astype(np.float32)
    X.data = 1 + np.log(X.data)
    return normalize(X)


def featurize16(items: list[dict]) -> sp.csr_matrix:
    """The fixed 2^16 numbering of the pooled judge: log counts, each block normalised, word then char."""
    W, C = rows(items)
    return sp.hstack([_log_norm(W @ _FOLD), _log_norm(C @ _FOLD)]).tocsr()


class Featurizer:
    """A center's own weighting of the fixed numbering: only the hashed pieces this center has actually seen are
    kept as columns (its vocabulary, as the old per-center vectorizers had; character pieces need two conversations),
    sublinear tf, smoothed idf from this center's own conversations, l2 per block. The same recipe as before on
    hashed counts, so the model needs no text, the numbering is the same everywhere, and the model stays small."""

    def __init__(self):
        self.cols_w = self.cols_c = None
        self.idf_w = self.idf_c = None

    @staticmethod
    def _select(X, min_df: int):
        df = np.bincount(X.indices, minlength=X.shape[1])
        cols = np.where(df >= min_df)[0]
        n = max(X.shape[0], 1)
        idf = (np.log((1 + n) / (1 + df[cols])) + 1.0).astype(np.float32)
        return cols, idf

    def fit(self, items: list[dict]):
        W, C = rows(items)
        self.cols_w, self.idf_w = self._select(W, 1)
        self.cols_c, self.idf_c = self._select(C, CHAR_MIN_DF)
        return self._apply(W, C)

    def _apply(self, W, C):
        W = W[:, self.cols_w].astype(np.float32); C = C[:, self.cols_c].astype(np.float32)
        W.data = 1 + np.log(W.data); C.data = 1 + np.log(C.data)
        W = W @ sp.diags(self.idf_w); C = C @ sp.diags(self.idf_c)
        return sp.hstack([normalize(W.tocsr()), normalize(C.tocsr())]).tocsr()

    def transform(self, items: list[dict]):
        W, C = rows(items)
        return self._apply(W, C)

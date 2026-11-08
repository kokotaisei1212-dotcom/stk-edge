"""Why customers came back: read in the center's own cloud, by the center's own model, one sentence to STK.

stk-edge keeps the words of each finished conversation sent through it for 7 days, in the center's own cloud, and then
deletes them, unless the customer came back: those it keeps until the week's reading, and deletes once read. Once a week STK lists, per cell (intent x language x channel x vendor), the AI-finished conversations
whose customer came back; stk-edge reads those it holds, asks the center's own model why the customers came back,
masks the answer and sends STK that one sentence. Nothing else leaves: no words, no conversation, no suggestion.

The model is the center's: STK_WHY_MODEL="<base url>|<api key>|<model>" (an OpenAI-compatible /chat/completions, or
https://api.anthropic.com for Anthropic's /v1/messages). Without it nothing is read, nothing fails, the rest goes on.
The same file runs in stk-edge and in STK's tests (kept identical in both repositories)."""
from __future__ import annotations
import json, os, re, sqlite3, threading, time

from .pii import mask

KEEP_S = 7 * 86400
WEEK_S = 7 * 86400
MAX_CONVERSATIONS = 10          # read per cell per week
MAX_CHARS = 4000                # of each conversation
MAX_SENTENCE = 240
LANGS = {"en": "English", "ja": "Japanese", "es": "Spanish", "pt": "Portuguese", "fr": "French", "de": "German",
         "it": "Italian", "ko": "Korean", "zh": "Chinese", "th": "Thai", "id": "Indonesian", "vi": "Vietnamese",
         "tl": "Tagalog", "ar": "Arabic", "hi": "Hindi"}


class Store:
    """The words of finished conversations, 7 days, in a file in the center's own cloud. Nothing else is kept."""

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(os.environ.get("STK_EDGE_DATA", "/tmp/stk-edge"), "words.db")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS words(id TEXT PRIMARY KEY, at REAL, turns TEXT, pinned INTEGER DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")

    def _db(self):
        return sqlite3.connect(self.path, timeout=10)

    def keep(self, conversation_id: str, turns: list, now: float | None = None):
        now = now or time.time()
        clean = [{"role": str(t.get("role", "")), "text": str(t.get("text", ""))} for t in (turns or []) if isinstance(t, dict)]
        if not conversation_id or not clean:
            return
        with self._lock, self._db() as db:
            db.execute("INSERT OR REPLACE INTO words(id, at, turns, pinned) VALUES(?,?,?,0)", (str(conversation_id), now, json.dumps(clean)))
            db.execute("DELETE FROM words WHERE at < ? AND pinned=0", (now - KEEP_S,))

    def get(self, ids: list[str], now: float | None = None) -> dict:
        now = now or time.time()
        with self._lock, self._db() as db:
            db.execute("DELETE FROM words WHERE at < ? AND pinned=0", (now - KEEP_S,))
            out = {}
            for i in ids:
                r = db.execute("SELECT turns FROM words WHERE id=?", (str(i),)).fetchone()
                if r:
                    out[str(i)] = json.loads(r[0])
            return out

    def pin(self, ids: list[str]):
        """The customer came back: keep these words until the week's reading (the 7 days no longer apply)."""
        with self._lock, self._db() as db:
            db.executemany("UPDATE words SET pinned=1 WHERE id=?", [(str(i),) for i in ids])

    def forget(self, ids: list[str]):
        """Read: deleted."""
        with self._lock, self._db() as db:
            db.executemany("DELETE FROM words WHERE id=?", [(str(i),) for i in ids])

    def count(self) -> int:
        with self._lock, self._db() as db:
            return db.execute("SELECT COUNT(*) FROM words").fetchone()[0]

    def last_week(self) -> int | None:
        with self._lock, self._db() as db:
            r = db.execute("SELECT v FROM meta WHERE k='week'").fetchone()
            return int(r[0]) if r else None

    def done_week(self, week: int):
        with self._lock, self._db() as db:
            db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('week', ?)", (str(int(week)),))


def model_from_env(value: str | None = None) -> dict | None:
    """STK_WHY_MODEL="<base url>|<api key>|<model>". Anything else: None (and nothing is read)."""
    v = (value if value is not None else os.environ.get("STK_WHY_MODEL", "")).strip()
    parts = [p.strip() for p in v.split("|")]
    if len(parts) != 3 or not all(parts) or not parts[0].startswith(("https://", "http://")):
        return None
    return {"url": parts[0].rstrip("/"), "key": parts[1], "model": parts[2]}


def _lines(turns: list) -> str:
    who = {"customer": "Customer", "ai": "AI", "assistant": "AI", "agent": "Agent", "human": "Agent"}
    s = "\n".join(f"{who.get(t['role'], t['role'] or 'Turn')}: {t['text']}" for t in turns)
    return s[:MAX_CHARS]


def prompt(items: list[dict], language: str) -> tuple[str, str]:
    lang = LANGS.get((language or "en")[:2], "English")
    system = (f"You read customer-service conversations that an AI assistant finished on its own, after which the same "
              f"customer came back within 7 days about the same problem. Answer with ONE sentence of at most 25 words, "
              f"in {lang}: why the customers came back, that is, what the AI did wrong. Describe the failure only: "
              f"no advice, no fix, no names, no numbers that identify anyone, no quotes.")
    parts = []
    for n, it in enumerate(items, 1):
        parts.append(f"Conversation {n}, finished by the AI:\n{_lines(it['finished'])}")
        if it.get("returned"):
            parts.append(f"The same customer came back:\n{_lines(it['returned'])}")
    return system, "\n\n".join(parts)


def ask(cfg: dict, system: str, user: str, post=None, timeout: float = 60) -> str:
    """One call to the center's own model: Anthropic's messages API, or any OpenAI-compatible chat completions."""
    import httpx
    post = post or httpx.post
    if "anthropic.com" in cfg["url"]:
        r = post(cfg["url"] + "/v1/messages", timeout=timeout,
                 headers={"x-api-key": cfg["key"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                 json={"model": cfg["model"], "max_tokens": 120, "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    r = post(cfg["url"] + "/chat/completions", timeout=timeout, headers={"Authorization": f"Bearer {cfg['key']}"},
             json={"model": cfg["model"], "max_tokens": 120, "temperature": 0,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def one_sentence(raw: str) -> str:
    """The first sentence only, masked, and never longer than the limit."""
    s = re.sub(r"\s+", " ", (raw or "")).strip().strip('"“”\'')
    m = re.search(r"^(.+?[.!?。！？])(\s|$)", s)
    s = (m.group(1) if m else s).strip()
    s = mask(s)
    return s if len(s) <= MAX_SENTENCE else s[:MAX_SENTENCE - 1].rstrip() + "…"


def pin_from_queue(stk_get, store: Store) -> int:
    """Every hour: the conversations STK says came back are kept until they are read."""
    q = stk_get("/v1/edge/why-queue")
    ids = [x for cell in q.get("cells", []) for c in cell.get("conversations", []) for x in (c.get("id"), c.get("returned_as")) if x]
    store.pin(ids)
    return len(ids)


def read_week(stk_get, stk_post, store: Store, cfg: dict | None, now: float | None = None, ask_fn=None) -> list[dict]:
    """The week's reading: STK's list of cells, the words this cloud holds for them, one sentence per cell back to STK.
    stk_get(path) -> dict and stk_post(path, json) -> response: how this side talks to STK (stk-edge: its key)."""
    if not cfg:
        return []
    now = now or time.time()
    q = stk_get("/v1/edge/why-queue")
    week, language, sent = q.get("week"), q.get("language", "en"), []
    for cell in q.get("cells", []):
        convs = cell.get("conversations", [])[:MAX_CONVERSATIONS]
        words = store.get([c["id"] for c in convs] + [c["returned_as"] for c in convs if c.get("returned_as")], now)
        items = [{"finished": words[c["id"]], "returned": words.get(c.get("returned_as"))} for c in convs if c["id"] in words]
        if not items:
            continue          # none of these passed through this stk-edge: nothing to read
        system, user = prompt(items, language)
        try:
            raw = (ask_fn or ask)(cfg, system, user)
        except Exception:
            continue          # the model did not answer: this cell waits for next week; nothing fails
        s = one_sentence(raw)
        if s:
            stk_post("/v1/edge/why", {"week": week, "cell": cell["cell"], "sentence": s, "n": len(items)})
            sent.append({"cell": cell["cell"], "sentence": s, "n": len(items)})
            store.forget(list(words))     # read: deleted
    if week is not None:
        store.done_week(week)
    return sent


def due(store: Store, now: float | None = None) -> bool:
    """Once a week, on the last day of STK's week (so the week's returns are in), and once only."""
    now = now or time.time()
    week = int(now // WEEK_S)
    return now - week * WEEK_S >= 6 * 86400 and store.last_week() != week

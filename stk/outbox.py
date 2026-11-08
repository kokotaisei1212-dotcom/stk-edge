"""The outbox: finished conversations STK did not accept yet, kept in this cloud and sent again until it does.

A /v1/conversations post that does not reach STK (no answer, 5xx, 429) is not lost: its rewritten body (features, no
words: `turns` and `known_pii` are taken out before it gets here) goes into a file under STK_EDGE_DATA (outbox.db) and
is sent again after 30 s, 2 min, 10 min, 1 h, then every hour, until STK accepts it. STK accepts conversations up to 30
days late and counts each conversation_id once, so after 30 days an item is dropped, and sending one twice is harmless.
A 4xx other than 429 means STK will never accept it: dropped (and logged), not sent again. At most MAX_ITEMS
conversations are kept; beyond that the oldest are dropped and counted."""
from __future__ import annotations
import json, logging, os, sqlite3, threading, time

log = logging.getLogger("stk.edge")

BACKOFF_S = [30, 120, 600, 3600]          # then hourly
MAX_AGE_S = 30 * 86400
MAX_ITEMS = 200_000


class Outbox:
    def __init__(self, path: str | None = None, max_items: int = MAX_ITEMS):
        self.path = path or os.path.join(os.environ.get("STK_EDGE_DATA", "/tmp/stk-edge"), "outbox.db")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.max_items = max_items
        self.backoff = list(BACKOFF_S)
        self.max_age = MAX_AGE_S
        self.tick = 5.0                   # how often the worker looks for items that are due
        self._lock = threading.Lock()     # one writer, and one sender at a time: each item is sent by one thread only
        self._send_lock = threading.Lock()
        self._pause_until = 0.0
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS q(id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, next REAL, tries INTEGER,"
                       " path TEXT, key TEXT, n INTEGER, body TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v INTEGER)")

    def _db(self):
        return sqlite3.connect(self.path, timeout=10)

    def _count(self, db, k: str, n: int):
        db.execute("INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v = v + excluded.v", (k, n))

    def put(self, path: str, body: list, key: str | None = None, now: float | None = None) -> int:
        """Keep one rewritten post (a list of conversations, no words) to send again. Returns how many conversations."""
        if any(isinstance(it, dict) and ("turns" in it or "known_pii" in it) for it in body):
            raise ValueError("the outbox holds rewritten bodies only")      # the words never go to disk here
        now = now or time.time()
        n = len(body)
        with self._lock, self._db() as db:
            db.execute("INSERT INTO q(at, next, tries, path, key, n, body) VALUES(?,?,0,?,?,?,?)",
                       (now, now + self.backoff[0], path, key, n, json.dumps(body)))
            total = db.execute("SELECT COALESCE(SUM(n), 0) FROM q").fetchone()[0]
            while total > self.max_items:                                    # full: the oldest go
                r = db.execute("SELECT id, n FROM q ORDER BY id LIMIT 1").fetchone()
                db.execute("DELETE FROM q WHERE id=?", (r[0],))
                self._count(db, "dropped_full", r[1])
                total -= r[1]
                log.warning("stk-edge outbox full (%d): dropped %d oldest conversations", self.max_items, r[1])
        return n

    def length(self) -> int:
        """Conversations waiting to be sent."""
        with self._lock, self._db() as db:
            return db.execute("SELECT COALESCE(SUM(n), 0) FROM q").fetchone()[0]

    def stats(self) -> dict:
        with self._lock, self._db() as db:
            out = {"queued": db.execute("SELECT COALESCE(SUM(n), 0) FROM q").fetchone()[0]}
            out.update({k: v for k, v in db.execute("SELECT k, v FROM meta")})
            return out

    def _delay(self, tries: int) -> float:
        """The wait after the tries-th retry failed (the first wait, after the original post, is backoff[0])."""
        return self.backoff[min(tries, len(self.backoff) - 1)]

    def drain(self, send, now: float | None = None) -> dict:
        """Send what is due, oldest first. send(path, body, key) -> HTTP status, or None when STK did not answer.
        Stops at the first send STK did not take (it is down: the rest waits for the next tick)."""
        done = {"sent": 0, "dropped": 0, "expired": 0, "failed": 0}
        with self._send_lock:
            now = now or time.time()
            if now < self._pause_until:
                return done
            with self._lock, self._db() as db:
                old = db.execute("SELECT COALESCE(SUM(n), 0) FROM q WHERE at < ?", (now - self.max_age,)).fetchone()[0]
                if old:
                    db.execute("DELETE FROM q WHERE at < ?", (now - self.max_age,))
                    self._count(db, "expired", old)
                    log.warning("stk-edge outbox: %d conversations older than 30 days dropped (STK no longer accepts them)", old)
                    done["expired"] = old
                due = db.execute("SELECT id, tries, path, key, n, body FROM q WHERE next <= ? ORDER BY id LIMIT 100", (now,)).fetchall()
            for rid, tries, path, key, n, body in due:
                try:
                    status = send(path, json.loads(body), key)
                except Exception:
                    status = None
                with self._lock, self._db() as db:
                    if status is not None and 200 <= status < 300:
                        db.execute("DELETE FROM q WHERE id=?", (rid,))
                        done["sent"] += n
                        continue
                    if status is not None and 400 <= status < 500 and status != 429:
                        db.execute("DELETE FROM q WHERE id=?", (rid,))
                        self._count(db, "dropped_rejected", n)
                        log.error("stk-edge outbox: STK refused %d conversations (HTTP %d): dropped, not sent again", n, status)
                        done["dropped"] += n
                        continue
                    delay = self._delay(tries + 1)
                    db.execute("UPDATE q SET tries=?, next=? WHERE id=?", (tries + 1, now + delay, rid))
                self._pause_until = now + min(delay, self.backoff[0])
                done["failed"] += n
                log.warning("stk-edge outbox: STK did not take %d conversations (%s); next try in %gs", n, status or "no answer", delay)
                break
        return done

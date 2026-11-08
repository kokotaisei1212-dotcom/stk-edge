"""The edge: runs in your own cloud when no words may leave it. It masks personal data and turns the customer's
first messages into hashed counts (see stk.features), then forwards the same JSON STK expects, with `features` in
place of `turns`. The conversation text does not leave your cloud: STK receives numbers, not words.

Use it as a library:
    from stk.edge import prepare
    body = prepare(turns, known_pii=["Crystal Minh"])          # -> {"features": {...}}
    httpx.post(STK + "/v1/decide", json=[{**body, "customer_id": "c-1"}], headers={"X-STK-Key": KEY})

Or as a shim: point your platform at this container instead of STK; it rewrites and forwards.
    STK_URL=https://api.stk-evidence.com STK_KEY=stk_ai_... STK_FEATURE_KEY=<hex> uvicorn stk.edge:app --port 8001
It keeps one thing: the words of the finished conversations sent through it (POST /v1/conversations with `turns`),
7 days, in this cloud (STK_EDGE_DATA), for the week's "why" (stk.why): with STK_WHY_MODEL set, once a week your own
model reads the conversations whose customer came back and stk-edge sends STK one masked sentence per cell. Without
STK_WHY_MODEL nothing is read. The words never leave this cloud.
A finished conversation STK does not accept at once (no answer, 5xx, 429) is answered 202 {"queued": n} and kept,
rewritten and without words, in STK_EDGE_DATA/outbox.db, and sent again until STK accepts it (stk.outbox)."""
from __future__ import annotations
import os
from .pii import mask
from .features import features
from . import why as _why

READ_MESSAGES = int(os.environ.get("STK_READ_MESSAGES", "5"))


def customer_text(turns: list[dict], first_n: int = READ_MESSAGES) -> str:
    return " ".join([t.get("text", "") for t in turns if t.get("role") == "customer"][:first_n])


def prepare(turns: list[dict], known_pii: list[str] | None = None, read_messages: int = READ_MESSAGES) -> dict:
    """Mask here, hash here, send numbers."""
    return {"features": features(mask(customer_text(turns, read_messages), known_pii))}


def rewrite(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        it = dict(it)
        if it.get("turns") is not None and it.get("features") is None:
            it.update(prepare(it.pop("turns"), it.pop("known_pii", None)))
        it.pop("turns", None); it.pop("known_pii", None)
        out.append(it)
    return out


try:  # the shim is optional: the library part works without FastAPI
    import json
    from fastapi import FastAPI, Header, Request, Response
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
    import httpx
    from . import outbox as _outbox_mod

    app = FastAPI(title="STK edge", version="0.15.0")

    VERSION = "0.15.0"
    WHY_CFG = _why.model_from_env()
    STORE = None

    def _store():
        global STORE
        if STORE is None:
            STORE = _why.Store()
        return STORE
    _LAST = {"ok": None, "at": None}

    def _beat():
        """Once a minute, only the version: STK emails you (and STK) if stk-edge goes quiet for an hour."""
        import time as _t
        while True:
            try:
                r = httpx.post(f"{STK_URL}/v1/edge/heartbeat", json={"version": VERSION, "why": bool(WHY_CFG)}, headers={"X-STK-Key": STK_KEY}, timeout=10)
                _LAST.update(ok=r.status_code == 200, at=_t.time())
            except httpx.HTTPError:
                _LAST.update(ok=False, at=_t.time())
            _t.sleep(60)

    @app.on_event("startup")
    def _warm():
        # the name model takes about 2 seconds to load: load it now, so the first call is not slower than 300 ms
        import threading
        threading.Thread(target=lambda: mask("warm up with John Smith"), daemon=True).start()
        if STK_KEY:
            threading.Thread(target=_beat, daemon=True).start()
        if STK_KEY and WHY_CFG:
            threading.Thread(target=_why_loop, daemon=True).start()

    def _why_loop():
        """Once an hour: on the last day of STK's week, the week's reading (once). Nothing here can stop the proxy."""
        import time as _t
        get = lambda path: httpx.get(f"{STK_URL}{path}", headers={"X-STK-Key": STK_KEY}, timeout=30).json()
        post = lambda path, body: httpx.post(f"{STK_URL}{path}", json=body, headers={"X-STK-Key": STK_KEY}, timeout=30)
        while True:
            try:
                _why.pin_from_queue(get, _store())       # came back: kept until read
                if _why.due(_store()):
                    _why.read_week(get, post, _store(), WHY_CFG)
            except Exception:
                pass
            _t.sleep(3600)

    @app.get("/health")
    def health():
        """For your own monitoring: stk-edge is up, whether its last heartbeat reached STK, and how many finished
        conversations wait in the outbox for STK to accept them."""
        try:
            ob = _outbox().stats()
        except Exception:
            ob = {"queued": None}
        return {"ok": True, "version": VERSION, "stk_reachable": _LAST["ok"], "last_heartbeat": _LAST["at"],
                "queued": ob.get("queued"), "queue_dropped": ob.get("dropped_full", 0) + ob.get("dropped_rejected", 0) + ob.get("expired", 0)}
    STK_URL = os.environ.get("STK_URL", "https://api.stk-evidence.com").rstrip("/")
    STK_KEY = os.environ.get("STK_KEY", "")
    _UNAVAILABLE = b'{"error": {"code": "unavailable", "message": "STK did not answer.", "fix": "Escalate on your own line, as before STK."}}'

    # one client, kept alive, for every forwarded call: a new HTTPS connection per call does not fit in 300 ms.
    # It belongs to the event loop that made it (uvicorn has one; a test client may start a new one per request).
    _CLIENT: list = [None, None]          # [loop, client]

    def _client() -> "httpx.AsyncClient":
        import asyncio
        loop = asyncio.get_running_loop()
        if _CLIENT[1] is None or _CLIENT[0] is not loop or _CLIENT[1].is_closed:
            _CLIENT[:] = [loop, httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=200, max_keepalive_connections=50, keepalive_expiry=120))]
        return _CLIENT[1]

    @app.on_event("startup")
    async def _open_client():
        _client()
        _outbox_worker()                  # what an earlier run could not send goes out now

    @app.on_event("shutdown")
    async def _close_client():
        cli, _CLIENT[:] = _CLIENT[1], [None, None]
        if cli is not None:
            await cli.aclose()

    import threading as _threading
    OUTBOX = None
    _WORKER: list = [None]
    _OUTBOX_LOCK = _threading.Lock()

    def _outbox() -> _outbox_mod.Outbox:
        global OUTBOX
        if OUTBOX is None:
            with _OUTBOX_LOCK:
                if OUTBOX is None:
                    OUTBOX = _outbox_mod.Outbox()
        return OUTBOX

    def _send_sync(cli: "httpx.Client", path: str, body, key: str | None):
        try:
            r = cli.post(f"{STK_URL}/v1/{path}", json=body, headers={"X-STK-Key": key or STK_KEY}, timeout=60)
            return r.status_code
        except httpx.HTTPError:
            return None

    def _outbox_loop():
        """Sends again what STK did not accept, with backoff. Nothing here can stop the proxy."""
        import time as _t
        with httpx.Client() as cli:
            while True:
                ob = None
                try:
                    ob = _outbox()
                    ob.drain(lambda path, body, key: _send_sync(cli, path, body, key))
                except Exception:
                    _outbox_mod.log.exception("stk-edge outbox: sending again failed; trying again shortly")
                _t.sleep(min(ob.tick, 1.0) if ob is not None else 1.0)   # short naps: what is due is found within a second

    def _outbox_worker():
        with _OUTBOX_LOCK:
            if _WORKER[0] is None or not _WORKER[0].is_alive():
                _WORKER[0] = _threading.Thread(target=_outbox_loop, daemon=True, name="stk-edge-outbox")
                _WORKER[0].start()

    def _prepare(path: str, body):
        """Runs in a worker thread (masking and hashing are CPU work): keep the words, then take them out."""
        if path == "conversations" and isinstance(body, list):   # the words stay here, 7 days, for the week's "why"
            for it in body:
                if isinstance(it, dict) and it.get("turns") and it.get("conversation_id"):
                    try:
                        _store().keep(str(it["conversation_id"]), it["turns"])
                    except Exception:
                        pass
        if path in ("decide", "conversations", "history") and isinstance(body, list):
            return rewrite(body), True
        return body, False

    def _answer(r: "httpx.Response") -> Response:
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    @app.post("/v1/{path:path}")
    async def forward(path: str, request: Request, x_stk_key: str | None = Header(None)):
        raw = await request.body()
        try:
            body = await run_in_threadpool(json.loads, raw) if raw else None
        except ValueError:
            body = None                            # not JSON: forwarded as it came, nothing to rewrite
        body, rewritten = await run_in_threadpool(_prepare, path, body)
        headers = {"X-STK-Key": x_stk_key or STK_KEY}
        sig = request.headers.get("X-STK-Signature")
        if sig and not rewritten:
            headers["X-STK-Signature"] = sig       # a signature over the original body cannot match a rewritten one: not sent then
        if rewritten:
            send = {"json": body, "headers": headers}
        else:                                      # not rewritten: the same bytes, so a signature over them still matches
            send = {"content": raw, "headers": {**headers, "Content-Type": request.headers.get("content-type", "application/json")}}
        # /v1/decide waits at most 2 s: your code stops at 300 ms anyway; conversations 15 s (then the outbox); history 60 s
        timeout = {"decide": 2, "conversations": 15}.get(path, 60)
        queueable = path == "conversations" and rewritten
        try:
            r = await _client().post(f"{STK_URL}/v1/{path}", timeout=timeout, **send)
        except httpx.HTTPError:
            r = None
        if queueable and (r is None or r.status_code >= 500 or r.status_code == 429):
            # STK did not take it: kept here without words, sent again until it does; your code need not send it again
            n = await run_in_threadpool(_outbox().put, path, body, x_stk_key if x_stk_key and x_stk_key != STK_KEY else None)
            _outbox_worker()
            return JSONResponse({"queued": n}, status_code=202)
        if r is None:
            # STK did not answer: say so at once; your code escalates on its own line, as before STK
            return Response(content=_UNAVAILABLE, status_code=503, media_type="application/json")
        return _answer(r)

    async def _get(url: str, request: Request, x_stk_key: str | None):
        try:
            r = await _client().get(url, params=dict(request.query_params), headers={"X-STK-Key": x_stk_key or STK_KEY}, timeout=10)
        except httpx.HTTPError:
            return Response(content=_UNAVAILABLE, status_code=503, media_type="application/json")
        return _answer(r)

    @app.get("/v2/{path:path}")
    async def forward_get_v2(path: str, request: Request, x_stk_key: str | None = Header(None)):
        return await _get(f"{STK_URL}/v2/{path}", request, x_stk_key)

    @app.get("/v1/{path:path}")
    async def forward_get(path: str, request: Request, x_stk_key: str | None = Header(None)):
        return await _get(f"{STK_URL}/v1/{path}", request, x_stk_key)
except ImportError:  # pragma: no cover
    app = None

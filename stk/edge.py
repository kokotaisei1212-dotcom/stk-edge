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
STK_WHY_MODEL nothing is read. The words never leave this cloud."""
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
    from fastapi import FastAPI, Header, Request, Response
    import httpx

    app = FastAPI(title="STK edge", version="0.14.0")

    VERSION = "0.14.0"
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
        """For your own monitoring: stk-edge is up, and whether its last heartbeat reached STK."""
        return {"ok": True, "version": VERSION, "stk_reachable": _LAST["ok"], "last_heartbeat": _LAST["at"]}
    STK_URL = os.environ.get("STK_URL", "https://api.stk-evidence.com").rstrip("/")
    STK_KEY = os.environ.get("STK_KEY", "")

    @app.post("/v1/{path:path}")
    async def forward(path: str, request: Request, x_stk_key: str | None = Header(None)):
        body = await request.json()
        if path == "conversations" and isinstance(body, list):   # the words stay here, 7 days, for the week's "why"
            for it in body:
                if isinstance(it, dict) and it.get("turns") and it.get("conversation_id"):
                    try:
                        _store().keep(str(it["conversation_id"]), it["turns"])
                    except Exception:
                        pass
        if path in ("decide", "conversations", "history") and isinstance(body, list):
            body = rewrite(body)
        headers = {"X-STK-Key": x_stk_key or STK_KEY}
        if request.headers.get("X-STK-Signature"):
            headers["X-STK-Signature"] = request.headers["X-STK-Signature"]  # note: a signature over the original body will not match the rewritten one
        # /v1/decide waits at most 2 s: your code stops at 300 ms anyway; the rest (history, conversations) up to 60 s
        try:
            async with httpx.AsyncClient(timeout=2 if path == "decide" else 60) as cli:
                r = await cli.post(f"{STK_URL}/v1/{path}", json=body, headers=headers)
        except httpx.HTTPError:
            # STK did not answer: say so at once; your code escalates on its own line, as before STK
            return Response(content=b'{"error": {"code": "unavailable", "message": "STK did not answer.", "fix": "Escalate on your own line, as before STK."}}',
                            status_code=503, media_type="application/json")
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    @app.get("/v2/{path:path}")
    async def forward_get_v2(path: str, request: Request, x_stk_key: str | None = Header(None)):
        try:
            async with httpx.AsyncClient(timeout=10) as cli:
                r = await cli.get(f"{STK_URL}/v2/{path}", params=dict(request.query_params), headers={"X-STK-Key": x_stk_key or STK_KEY})
        except httpx.HTTPError:
            return Response(content=b'{"error": {"code": "unavailable", "message": "STK did not answer.", "fix": "Escalate on your own line, as before STK."}}',
                            status_code=503, media_type="application/json")
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    @app.get("/v1/{path:path}")
    async def forward_get(path: str, request: Request, x_stk_key: str | None = Header(None)):
        try:
            async with httpx.AsyncClient(timeout=10) as cli:
                r = await cli.get(f"{STK_URL}/v1/{path}", params=dict(request.query_params), headers={"X-STK-Key": x_stk_key or STK_KEY})
        except httpx.HTTPError:
            return Response(content=b'{"error": {"code": "unavailable", "message": "STK did not answer.", "fix": "Escalate on your own line, as before STK."}}',
                            status_code=503, media_type="application/json")
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
except ImportError:  # pragma: no cover
    app = None

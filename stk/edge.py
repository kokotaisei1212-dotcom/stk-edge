"""The edge: runs in your own cloud when no words may leave it. It masks personal data and turns the customer's
first messages into hashed counts (see stk.features), then forwards the same JSON STK expects, with `features` in
place of `turns`. The conversation text does not leave your cloud: STK receives numbers, not words.

Use it as a library:
    from stk.edge import prepare
    body = prepare(turns, known_pii=["Crystal Minh"])          # -> {"features": {...}}
    httpx.post(STK + "/v1/decide", json=[{**body, "customer_id": "c-1"}], headers={"X-STK-Key": KEY})

Or as a shim: point your platform at this container instead of STK; it rewrites and forwards.
    STK_URL=https://api.stk-evidence.com STK_KEY=stk_ai_... STK_FEATURE_KEY=<hex> uvicorn stk.edge:app --port 8001
No state, no storage: it holds nothing between requests. Only the pii rules and the hashing run here."""
from __future__ import annotations
import os
from .pii import mask
from .features import features

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

    app = FastAPI(title="STK edge", version="0.9.0")
    STK_URL = os.environ.get("STK_URL", "https://api.stk-evidence.com").rstrip("/")
    STK_KEY = os.environ.get("STK_KEY", "")

    @app.post("/v1/{path:path}")
    async def forward(path: str, request: Request, x_stk_key: str | None = Header(None)):
        body = await request.json()
        if path in ("decide", "conversations", "history") and isinstance(body, list):
            body = rewrite(body)
        headers = {"X-STK-Key": x_stk_key or STK_KEY}
        if request.headers.get("X-STK-Signature"):
            headers["X-STK-Signature"] = request.headers["X-STK-Signature"]  # note: a signature over the original body will not match the rewritten one
        async with httpx.AsyncClient(timeout=60) as cli:
            r = await cli.post(f"{STK_URL}/v1/{path}", json=body, headers=headers)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    @app.get("/v2/{path:path}")
    async def forward_get_v2(path: str, request: Request, x_stk_key: str | None = Header(None)):
        async with httpx.AsyncClient(timeout=60) as cli:
            r = await cli.get(f"{STK_URL}/v2/{path}", params=dict(request.query_params), headers={"X-STK-Key": x_stk_key or STK_KEY})
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    @app.get("/v1/{path:path}")
    async def forward_get(path: str, request: Request, x_stk_key: str | None = Header(None)):
        async with httpx.AsyncClient(timeout=60) as cli:
            r = await cli.get(f"{STK_URL}/v1/{path}", params=dict(request.query_params), headers={"X-STK-Key": x_stk_key or STK_KEY})
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
except ImportError:  # pragma: no cover
    app = None

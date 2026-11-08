"""A finished conversation STK does not take at once is kept (without words) and sent again until STK takes it;
/v1/decide never waits for that; one kept-alive client carries every forwarded call."""
import http.server, json, os, socket, tempfile, threading, time

import pytest

MARK = "QUOKKA-TEAL-9031"
RECV = []            # (path, body, headers) of every POST STK received
PLAN = {}            # path -> statuses STK answers next (then 200)


class FakeSTK(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        RECV.append((self.path, json.loads(raw or b"null"), {k.lower(): v for k, v in self.headers.items()}))
        code = PLAN.get(self.path, []).pop(0) if PLAN.get(self.path) else 200
        b = json.dumps({"accepted": 1} if code == 200 else {"error": {"code": str(code)}}).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        b = b"{}"; self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)


def _serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeSTK)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


def _closed_port_url():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return f"http://127.0.0.1:{port}"


UP, DOWN = _serve(), _closed_port_url()


@pytest.fixture
def edge(monkeypatch):
    os.environ.setdefault("STK_PII_NER", "0")
    from stk import edge as e, why, outbox            # imported here: the other test file sets the environment first
    data = tempfile.mkdtemp()
    ob = outbox.Outbox(os.path.join(data, "outbox.db"))
    ob.backoff, ob.tick = [0.2, 0.3], 0.05
    e.OUTBOX = ob                                      # not put back after: the outbox thread would make a default one meanwhile
    monkeypatch.setattr(e, "STORE", why.Store(os.path.join(data, "words.db")))
    monkeypatch.setattr(e, "STK_URL", UP)
    monkeypatch.setattr(e, "STK_KEY", "")             # no heartbeat or weekly threads from startup in these tests
    RECV.clear(); PLAN.clear()
    e._data = data
    yield e
    e._CLIENT[:] = [None, None]


def _conv(cid):
    return [{"conversation_id": cid, "customer_id": "h9", "handled_by": "ai", "label": "refund_status", "known_pii": ["Ana Ruiz"],
             "turns": [{"role": "customer", "text": f"Ana Ruiz here, my refund {MARK} never came"}, {"role": "ai", "text": "Issued."}]}]


def _wait(cond, s=5.0):
    end = time.time() + s
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _delivered(cid):
    return [b for p, b, h in RECV if p == "/v1/conversations" and isinstance(b, list) and any(it.get("conversation_id") == cid for it in b)]


def _no_words_anywhere(e):
    for root, _, files in os.walk(e._data):
        for f in files:
            if f.startswith("words.db"):
                continue                                # the words store: the words are meant to be there
            with open(os.path.join(root, f), "rb") as fh:
                data = fh.read()
            assert MARK.encode() not in data and b"Ana Ruiz" not in data, f
    sent = json.dumps([b for p, b, h in RECV])
    assert MARK not in sent and "Ana Ruiz" not in sent and '"turns"' not in sent and "known_pii" not in sent


def test_stk_down_queued_202_then_delivered_exactly_once(edge):
    from fastapi.testclient import TestClient
    with TestClient(edge.app) as c:
        edge.STK_URL = DOWN
        r = c.post("/v1/conversations", json=_conv("q1"), headers={"X-STK-Key": "stk_ai_center"})
        assert r.status_code == 202 and r.json() == {"queued": 1}
        assert c.get("/health").json()["queued"] == 1
        time.sleep(0.6)                                  # retried against a dead STK: still queued
        assert edge._outbox().length() == 1 and edge._outbox().stats()  # nothing lost
        assert edge.STORE.get(["q1"])["q1"][0]["text"].endswith("never came")   # words kept here, as before
        edge.STK_URL = UP
        assert _wait(lambda: edge._outbox().length() == 0)
        time.sleep(0.5)
        got = _delivered("q1")
        assert len(got) == 1                             # exactly once
        assert [h.get("x-stk-key") for p, b, h in RECV if p == "/v1/conversations"] == ["stk_ai_center"]
        assert c.get("/health").json()["queued"] == 0
    _no_words_anywhere(edge)


@pytest.mark.parametrize("code", [500, 503, 429])
def test_stk_5xx_or_429_queued_and_retried(edge, code):
    from fastapi.testclient import TestClient
    PLAN["/v1/conversations"] = [code, code, code]
    with TestClient(edge.app) as c:
        r = c.post("/v1/conversations", json=_conv("r1"))
        assert r.status_code == 202 and r.json() == {"queued": 1}
        assert _wait(lambda: edge._outbox().length() == 0)
        time.sleep(0.4)
    assert len(_delivered("r1")) == 4                    # the original, two refused retries, the accepted one
    assert edge._outbox().stats().get("dropped_rejected", 0) == 0
    _no_words_anywhere(edge)


def test_stk_400_is_not_retried(edge):
    from fastapi.testclient import TestClient
    PLAN["/v1/conversations"] = [400]
    with TestClient(edge.app) as c:
        r = c.post("/v1/conversations", json=_conv("b1"))
        assert r.status_code == 400                      # STK's answer, to the center's code, as it came
        assert edge._outbox().length() == 0
        PLAN["/v1/conversations"] = [500, 422]           # refused for good while waiting in the outbox: dropped
        assert c.post("/v1/conversations", json=_conv("b2")).status_code == 202
        assert _wait(lambda: edge._outbox().length() == 0)
        time.sleep(0.6)
    assert len(_delivered("b1")) == 1 and len(_delivered("b2")) == 2
    assert edge._outbox().stats()["dropped_rejected"] == 1


def test_the_outbox_never_holds_the_words(edge):
    from stk import outbox
    from fastapi.testclient import TestClient
    with TestClient(edge.app) as c:
        edge.STK_URL = DOWN
        for i in range(5):
            assert c.post("/v1/conversations", json=_conv(f"w{i}")).status_code == 202
        assert edge._outbox().length() == 5
        _no_words_anywhere(edge)
        edge.STK_URL = UP
        assert _wait(lambda: edge._outbox().length() == 0)
    _no_words_anywhere(edge)
    with pytest.raises(ValueError):
        outbox.Outbox(os.path.join(edge._data, "x.db")).put("conversations", _conv("raw"))


def test_outbox_is_bounded_oldest_dropped_and_old_items_expire(edge):
    from stk import outbox
    ob = outbox.Outbox(os.path.join(edge._data, "b.db"), max_items=3)
    for i in range(5):
        ob.put("conversations", [{"conversation_id": f"x{i}"}])
    assert ob.length() == 3 and ob.stats()["dropped_full"] == 2
    sent = []
    ob.drain(lambda p, b, k: sent.append(b[0]["conversation_id"]) or 200, now=time.time() + 31)
    assert sent == ["x2", "x3", "x4"]                    # the oldest went; the rest, in order
    ob.put("conversations", [{"conversation_id": "late"}], now=time.time() - 31 * 86400)
    ob.drain(lambda p, b, k: 200)
    assert ob.length() == 0 and ob.stats()["expired"] == 1
    assert [ob._delay(t) for t in (1, 2, 3, 4, 9)] == [120, 600, 3600, 3600, 3600]


def test_decide_with_stk_down_is_503_at_once_and_never_queued(edge):
    from fastapi.testclient import TestClient
    with TestClient(edge.app) as c:
        edge.STK_URL = DOWN
        t = time.time()
        r = c.post("/v1/decide", json=[{"customer_id": "c1", "turns": [{"role": "customer", "text": f"refund {MARK}"}]}])
        assert r.status_code == 503 and time.time() - t < 1.0
        edge.STK_URL = UP
        PLAN["/v1/decide"] = [500]
        assert c.post("/v1/decide", json=[{"customer_id": "c1"}]).status_code == 500
    assert edge._outbox().length() == 0


def test_one_shared_client_and_no_signature_on_rewritten_bodies(edge, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    made = []
    real = httpx.AsyncClient

    class Counting(real):
        def __init__(self, *a, **k):
            made.append(1); super().__init__(*a, **k)
    monkeypatch.setattr(httpx, "AsyncClient", Counting)
    with TestClient(edge.app) as c:
        for i in range(5):
            assert c.post("/v1/decide", json=[{"customer_id": f"c{i}", "turns": [{"role": "customer", "text": "hi"}]}],
                          headers={"X-STK-Signature": "sig", "X-STK-Key": "k1"}).status_code == 200
        assert c.post("/v1/conversations", json=_conv("s1"), headers={"X-STK-Signature": "sig"}).status_code == 200
        assert c.get("/v1/anything").status_code == 200
        assert c.post("/v1/other", content=b'{"a": 1}', headers={"X-STK-Signature": "sig", "Content-Type": "application/json"}).status_code == 200
        cli = edge._CLIENT[1]
    assert len(made) == 1                                 # one client for every call
    assert cli.is_closed                                  # closed on shutdown
    rewritten = [h for p, b, h in RECV if p in ("/v1/decide", "/v1/conversations")]
    assert len(rewritten) == 6 and not any("x-stk-signature" in h for h in rewritten)
    assert all(h.get("x-stk-key") == "k1" for p, b, h in RECV if p == "/v1/decide")
    other = [h for p, b, h in RECV if p == "/v1/other"]
    assert other[0].get("x-stk-signature") == "sig"       # not rewritten: the same bytes, the signature still matches

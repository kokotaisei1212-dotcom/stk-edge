"""stk-edge keeps the words of /v1/conversations for 7 days in its own cloud, forwards them to STK without the words,
and once a week asks the center's own model why customers came back; STK receives one sentence and nothing else."""
import http.server, json, os, tempfile, threading, time

MARK = "ZEBRA-PLUM-4417"
SEEN = {"stk": [], "model": []}


def _serve(handler):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


class FakeSTK(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        SEEN["stk"].append(("GET", self.path, None))
        if self.path.startswith("/v1/edge/why-queue"):
            return self._json({"week": int(time.time() // (7 * 86400)), "language": "en",
                               "cells": [{"cell": "refund_status|chat|en|sierra", "score": 40.0, "conversations": [{"id": "a1", "returned_as": "b1"}]}]})
        self._json({})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
        SEEN["stk"].append(("POST", self.path, body))
        self._json({"accepted": 1} if self.path == "/v1/conversations" else {"ok": True})


class FakeModel(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        SEEN["model"].append(body)
        b = json.dumps({"choices": [{"message": {"content": "The AI cut its answers off mid-sentence. Customers had to ask again."}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)


STK_URL, MODEL_URL = _serve(FakeSTK), _serve(FakeModel)
os.environ.update({"STK_URL": STK_URL, "STK_KEY": "stk_ai_test", "STK_EDGE_DATA": tempfile.mkdtemp(), "STK_PII_NER": "0",
                   "STK_WHY_MODEL": f"{MODEL_URL}/v1|model-key|small-model"})
from fastapi.testclient import TestClient  # noqa: E402
from stk import edge, why  # noqa: E402

c = TestClient(edge.app)


def test_the_words_stay_here_seven_days_and_stk_gets_them_never():
    turns = [{"role": "customer", "text": f"My refund {MARK} never came"}, {"role": "ai", "text": "Your refund was issued and"}]
    r = c.post("/v1/conversations", json=[{"conversation_id": "a1", "customer_id": "h1", "handled_by": "ai", "label": "refund_status", "turns": turns}])
    assert r.status_code == 200
    sent = [b for m, p, b in SEEN["stk"] if p == "/v1/conversations"][-1]
    assert "turns" not in sent[0] and MARK not in json.dumps(sent)           # STK: no words
    assert edge._store().get(["a1"])["a1"] == turns                            # here: the words, for the week
    edge._store().keep("old", turns, now=time.time() - 8 * 86400)
    assert edge._store().get(["old"]) == {}                                     # after 7 days: gone


def test_the_week_reads_here_with_the_centers_model_and_stk_gets_one_sentence():
    c.post("/v1/conversations", json=[{"conversation_id": "b1", "customer_id": "h1", "handled_by": "human", "label": "refund_status",
                                       "turns": [{"role": "customer", "text": "Your answer stopped halfway, so I am asking again"}]}])
    get = lambda path: json.loads(__import__("httpx").get(STK_URL + path).text)
    post = lambda path, body: __import__("httpx").post(STK_URL + path, json=body)
    out = why.read_week(get, post, edge._store(), why.model_from_env())
    assert out == [{"cell": "refund_status|chat|en|sierra", "sentence": "The AI cut its answers off mid-sentence.", "n": 1}]
    prompt = json.dumps(SEEN["model"][-1])
    assert MARK in prompt and "stopped halfway" in prompt                       # the model read the words, here
    posted = [b for m, p, b in SEEN["stk"] if p == "/v1/edge/why"][-1]
    assert posted == {"week": int(time.time() // (7 * 86400)), "cell": "refund_status|chat|en|sierra", "sentence": "The AI cut its answers off mid-sentence.", "n": 1}
    assert MARK not in json.dumps([b for m, p, b in SEEN["stk"]])               # STK never saw a word
    assert edge._store().get(["a1", "b1"]) == {}                                # read: deleted


def test_without_a_model_nothing_is_read_and_nothing_fails():
    assert why.model_from_env("") is None
    assert why.read_week(None, None, edge._store(), None) == []


def test_a_conversation_that_came_back_is_kept_until_read_and_deleted_once_read():
    st = why.Store(os.path.join(tempfile.mkdtemp(), "w.db"))
    now = time.time()
    st.keep("p1", [{"role": "customer", "text": "kept"}], now=now - 6 * 86400)
    st.keep("q1", [{"role": "customer", "text": "not back"}], now=now - 6 * 86400)
    st.pin(["p1"])
    later = now + 2 * 86400                                   # 8 days after both
    assert st.get(["p1", "q1"], later) == {"p1": [{"role": "customer", "text": "kept"}]}
    st.forget(["p1"])
    assert st.count() == 0

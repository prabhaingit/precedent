"""Tests run with a small stand-in model, so they need neither TabPFN weights nor SALT."""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from precedent import Engine, Ledger, build_task
from precedent.server import build_server


class StandIn:
    """Behaves like an in-context model: fit on the rows it is shown, then predict."""

    def fit(self, X, y):
        self.m = make_pipeline(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                               RandomForestClassifier(n_estimators=20, random_state=0)).fit(X.astype(str), y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X.astype(str))


def make_frame(n=4000, seed=1):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "CREATIONDATE": pd.date_range("2018-01-01", periods=n, freq="h").astype(str),
        "SOLDTOPARTY": r.choice([f"C{i}" for i in range(60)], n),
        "SALESORGANIZATION": r.choice(["A", "B", "C"], n),
        "QTY": r.gamma(2, 5, n),
        "DOCID": [f"D{i}" for i in range(n)],
    })
    by_customer = df.SOLDTOPARTY.str[1:].astype(int) % 3
    df["TERMS"] = np.where(r.random(n) < 0.1, "99", by_customer.map({0: "03", 1: "32", 2: "54"}))
    df.loc[r.random(n) < 0.03, "TERMS"] = ""
    return df


@pytest.fixture(scope="module")
def engine():
    task = build_task(make_frame(), "TERMS", n_context=600, n_new=100)
    return Engine(task, model_factory=lambda model: StandIn())


def test_task_is_clean(engine):
    t = engine.task
    assert "" not in set(t.y_ctx) and "DOCID" not in t.features and "TERMS" not in t.features
    assert len(t.X_ctx) == 600 and len(t.X_new) == 100


def test_predict(engine):
    p = engine.predict(order_id=3)
    assert 0 < p["confidence"] <= 1 and p["predicted"] in set(engine.task.y_ctx)
    with pytest.raises(ValueError):
        engine.predict(order_id=3, model="plus")
    with pytest.raises(ValueError):
        engine.predict()


def test_predict_custom_fields(engine):
    fields = {"SOLDTOPARTY": "C3", "SALESORGANIZATION": "A", "QTY": 4.0}
    p = engine.predict(fields=fields)
    assert p["order_key"].startswith("custom:") and p["true_value"] is None
    with pytest.raises(ValueError):
        engine.predict(fields={"NOPE": 1})


def test_explain_then_verify(engine):
    ex = engine.explain(order_id=3, candidates=40, rounds=80)
    assert len(ex["precedents"]) == 10 and ex["status"] == "unverified"
    assert ex["precedents"][0]["influence"] >= ex["precedents"][-1]["influence"]
    v = engine.verify(explanation_id=ex["explanation_id"], n_random=4)
    assert len(v["random_fall_points"]) == 4
    assert v["verified"] == (v["fall_points"] >= 5 and v["fall_points"] > max(v["random_fall_points"]))
    assert v["strength"] in {"strong", "moderate", "weak", "none"}
    assert set(v["precedents_removed"]) == {p["past_order"] for p in ex["precedents"]}


def test_random_removals_never_include_precedents(engine):
    ex = engine.explain(order_id=5, candidates=30, rounds=60)
    stored = engine._explanations[ex["explanation_id"]]
    others = np.setdiff1d(stored["candidates"], stored["rows"])
    assert len(np.intersect1d(others, stored["rows"])) == 0 and len(others) == 20


def test_verify_by_order_id_creates_explanation(engine):
    v = engine.verify(order_id=9)
    assert v["order_id"] == 9 and "summary" in v


def test_ledger_chain(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("predict", {"order_id": 1}, {"order_id": 1, "predicted": "03"})
    led.append("verify", {"order_id": 1}, {"order_id": 1, "verified": True})
    assert led.check_chain() == {"intact": True, "first_bad_seq": None}
    assert len(led.read(order_id=1)) == 2 and len(led.read(tool="verify")) == 1
    lines = led.path.read_text().splitlines()
    first = json.loads(lines[0]); first["output"]["predicted"] = "32"
    led.path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    assert led.check_chain()["intact"] is False


def test_mcp_tools_end_to_end(engine, tmp_path):
    mcp = pytest.importorskip("mcp")
    if not hasattr(mcp, "Client"):
        pytest.skip("in-process client needs MCP SDK v2")
    server = build_server(lambda: engine, Ledger(tmp_path / "ledger.jsonl"))

    async def go():
        async with mcp.Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == {"list_orders", "predict", "explain", "verify", "ledger_entries"}

            def data(res):
                return res.structured_content if getattr(res, "structured_content", None) else json.loads(res.content[0].text)

            p = data(await client.call_tool("predict", {"order_id": 2}))
            ex = data(await client.call_tool("explain", {"order_id": 2, "candidates": 30, "rounds": 60}))
            v = data(await client.call_tool("verify", {"explanation_id": ex["explanation_id"]}))
            bad = data(await client.call_tool("predict", {"order_id": 2, "model": "plus"}))
            led = data(await client.call_tool("ledger_entries", {"last_n": 10}))
            return p, ex, v, bad, led

    p, ex, v, bad, led = asyncio.run(go())
    assert p["ledger_seq"] == 1 and ex["ledger_seq"] == 2 and v["ledger_seq"] == 3
    assert "error" in bad and "hosted API" in bad["error"]
    assert led["chain"]["intact"] and [e["tool"] for e in led["entries"]] == ["predict", "explain", "verify", "predict"]

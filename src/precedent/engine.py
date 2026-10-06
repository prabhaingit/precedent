"""The Precedent engine: predict, explain, verify.

predict  - ask the full model (all past rows) for an answer.
explain  - find the past rows the answer seems to rest on (the precedents).
verify   - remove the precedents from the full model and check that the answer moves.

The method needs only a model that can be given past rows and asked for class
probabilities, so the model is created through a small factory and can be swapped.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from .data import Task

# TabPFN repeats these on every call; show the CPU notice once and hide the free-text one.
warnings.filterwarnings("once", message="Running on CPU with more than 1000 samples")
warnings.filterwarnings("ignore", message="These columns look like free text")

MODELS = {"fast": "TabPFN-3.5-Fast (local weights)", "base": "TabPFN-3.5 (local weights)"}

# Descriptive bands for the size of the fall, in percentage points. These are
# reporting conventions chosen from the run-3 results, not validated thresholds.
STRENGTH_BANDS = [(40.0, "strong"), (15.0, "moderate"), (5.0, "weak")]
MIN_FALL_POINTS = 5.0


def tabpfn_factory(n_estimators: int = 1, seed: int = 0) -> Callable[[str], object]:
    """Default model factory: local TabPFN-3.5 weights."""
    def make(model: str):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        version = {"fast": ModelVersion.V3_5_FAST, "base": ModelVersion.V3_5}[model]
        return TabPFNClassifier.create_default_for_version(version, n_estimators=n_estimators, random_state=seed)
    return make


def strength_of(fall_points: float) -> str:
    for threshold, label in STRENGTH_BANDS:
        if fall_points >= threshold:
            return label
    return "negligible"


class Engine:
    def __init__(self, task: Task, model_factory: Callable[[str], object] | None = None,
                 n_estimators: int = 1, seed: int = 0):
        self.task = task
        self.seed = seed
        self.n_estimators = n_estimators
        self._make = model_factory or tabpfn_factory(n_estimators, seed)
        self._full: dict[str, object] = {}
        self._answers: dict[tuple[str, str], dict] = {}
        self._explanations: dict[str, dict] = {}
        self._latest: dict[tuple[str, str], str] = {}
        self._all_rows = np.arange(len(task.X_ctx))
        self._nn = NearestNeighbors().fit(task.E_ctx)

    # ---------- helpers

    def _check_model(self, model: str) -> None:
        if model not in MODELS:
            raise ValueError(
                f"Unknown model '{model}'. Available: {sorted(MODELS)}. "
                "TabPFN-3.5-Plus and Thinking run only on the hosted API and are not supported yet.")

    def _resolve(self, order_id: int | None, fields: dict | None):
        """Return (key, one-row frame, encoded row, true answer or None)."""
        if (order_id is None) == (fields is None):
            raise ValueError("Give exactly one of order_id or fields.")
        if order_id is not None:
            n = len(self.task.X_new)
            if not 0 <= order_id < n:
                raise ValueError(f"order_id must be between 0 and {n - 1}.")
            return (f"order:{order_id}", self.task.X_new.iloc[[order_id]],
                    self.task.E_new[[order_id]], str(self.task.y_new[order_id]))
        frame = self.task.row_from_fields(fields)
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:10]
        return f"custom:{digest}", frame, self.task.encode(frame), None

    def _probabilities(self, rows: np.ndarray, x_row: pd.DataFrame, model: str) -> dict[str, float]:
        """Class probabilities for x_row when the model sees only these past rows."""
        ys = self.task.y_ctx[rows]
        classes = np.unique(ys)
        if len(classes) == 1:
            return {str(classes[0]): 1.0}
        m = self._make(model).fit(self.task.X_ctx.iloc[rows], ys)
        return {str(c): float(p) for c, p in zip(m.classes_, m.predict_proba(x_row)[0])}

    def _full_answer(self, key: str, x_row: pd.DataFrame, model: str) -> dict:
        if (model, key) not in self._answers:
            if model not in self._full:
                self._full[model] = self._make(model).fit(self.task.X_ctx, self.task.y_ctx)
            m = self._full[model]
            probs = {str(c): float(p) for c, p in zip(m.classes_, m.predict_proba(x_row)[0])}
            self._answers[(model, key)] = probs
        return self._answers[(model, key)]

    # ---------- the three operations

    def list_orders(self, start: int = 0, count: int = 5) -> dict:
        """A few held-out orders with their fields, so a caller can pick one."""
        t = self.task
        stop = min(start + count, len(t.X_new))
        return {"dataset": t.name, "target": t.target, "total_orders": len(t.X_new),
                "orders": [{"order_id": i, "fields": _plain(t.X_new.iloc[i].to_dict())} for i in range(start, stop)]}

    def predict(self, order_id: int | None = None, fields: dict | None = None, model: str = "fast") -> dict:
        self._check_model(model)
        key, x_row, _, truth = self._resolve(order_id, fields)
        probs = self._full_answer(key, x_row, model)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        return {
            "order_id": order_id, "order_key": key, "target": self.task.target,
            "model": MODELS[model], "past_orders_used": len(self.task.X_ctx),
            "predicted": ranked[0][0], "confidence": round(ranked[0][1], 4),
            "top_alternatives": [{"value": c, "probability": round(p, 4)} for c, p in ranked[1:4]],
            "true_value": truth,
        }

    def explain(self, order_id: int | None = None, fields: dict | None = None, model: str = "fast",
                candidates: int = 60, rounds: int = 180, top_k: int = 10) -> dict:
        """Find the past orders the full model's answer rests on."""
        self._check_model(model)
        t0 = time.time()
        key, x_row, e_row, truth = self._resolve(order_id, fields)
        probs = self._full_answer(key, x_row, model)
        answer = max(probs, key=probs.get)
        seed = self.seed + (order_id if order_id is not None else int(key[-6:], 16) % 10_000)
        rng = np.random.default_rng(seed)

        k = min(candidates, len(self._all_rows))
        top_k = min(top_k, k)
        cand = self._nn.kneighbors(e_row, n_neighbors=k, return_distance=False)[0]

        masks = np.zeros((rounds, k), dtype=bool)
        outs = np.zeros(rounds)
        for r in range(rounds):
            masks[r, rng.choice(k, size=k // 2, replace=False)] = True
            outs[r] = self._probabilities(cand[masks[r]], x_row, model).get(answer, 0.0)

        scores = Ridge(alpha=1.0).fit(masks.astype(float), outs).coef_
        best = np.argsort(-scores)[:top_k]
        rows = cand[best]
        positive = np.clip(scores, 0, None)
        new = x_row.iloc[0]
        precedents = []
        for rank, (row, s) in enumerate(zip(rows, scores[best]), start=1):
            past = self.task.X_ctx.iloc[row]
            matching = [c for c in self.task.categorical if past[c] == new[c]]
            precedents.append({
                "rank": rank, "past_order": int(row), "influence": round(float(s), 4),
                "value": str(self.task.y_ctx[row]),
                "same_as_prediction": bool(str(self.task.y_ctx[row]) == answer),
                "matching_fields": matching,
                "fields": _plain(past[self.task.categorical].to_dict()),
            })

        flat = bool(outs.std() < 1e-6)
        explanation_id = uuid.uuid4().hex[:12]
        result = {
            "explanation_id": explanation_id, "order_id": order_id, "order_key": key,
            "model": MODELS[model], "predicted": answer, "confidence": round(probs[answer], 4),
            "true_value": truth,
            "settings": {"candidates": k, "rounds": rounds, "top_k": top_k, "seed": seed,
                         "n_estimators": self.n_estimators},
            "round_confidence": {"min": round(float(outs.min()), 4), "max": round(float(outs.max()), 4)},
            "top_precedent_share": round(float(positive[best[0]] / positive.sum()), 3) if positive.sum() > 0 else 0.0,
            "precedents": precedents,
            "status": "unverified",
            "note": ("Confidence was the same in every round, so no past order stands out."
                     if flat else "These are candidate reasons. Call verify before presenting them as the explanation."),
            "seconds": round(time.time() - t0, 1),
        }
        self._explanations[explanation_id] = {"result": result, "model": model, "x_row": x_row, "key": key,
                                             "answer": answer, "rows": rows, "candidates": cand,
                                             "rng": np.random.default_rng(seed + 1)}
        self._latest[(model, key)] = explanation_id
        return result

    def verify(self, explanation_id: str | None = None, order_id: int | None = None,
               model: str = "fast", n_random: int = 5) -> dict:
        """Remove the precedents from the full model and check that the answer moves."""
        if explanation_id is None:
            if order_id is None:
                raise ValueError("Give an explanation_id, or an order_id that has been explained.")
            self._check_model(model)
            explanation_id = self._latest.get((model, f"order:{order_id}"))
            if explanation_id is None:
                explanation_id = self.explain(order_id=order_id, model=model)["explanation_id"]
        if explanation_id not in self._explanations:
            raise ValueError(f"Unknown explanation_id '{explanation_id}'. Explanations last only while the server runs.")
        ex = self._explanations[explanation_id]
        t0 = time.time()
        model, x_row, answer, rows, cand, rng = ex["model"], ex["x_row"], ex["answer"], ex["rows"], ex["candidates"], ex["rng"]
        before = self._full_answer(ex["key"], x_row, model)[answer]

        after_probs = self._probabilities(np.setdiff1d(self._all_rows, rows), x_row, model)
        after = after_probs.get(answer, 0.0)
        answer_after = max(after_probs, key=after_probs.get)
        fall = (before - after) * 100

        # Random comparison: draw only from rows that are NOT precedents.
        others = np.setdiff1d(cand, rows)
        if len(others) < len(rows):
            others = np.setdiff1d(self._all_rows, rows)
        random_falls = []
        for _ in range(n_random):
            removed = rng.choice(others, size=len(rows), replace=False)
            p = self._probabilities(np.setdiff1d(self._all_rows, removed), x_row, model).get(answer, 0.0)
            random_falls.append((before - p) * 100)
        largest_random = max(random_falls) if random_falls else 0.0

        if fall < MIN_FALL_POINTS:
            verified, reason = False, "fall_below_threshold"
        elif fall <= largest_random:
            verified, reason = False, "random_removal_matched"
        else:
            verified, reason = True, None
        strength = strength_of(fall) if verified else "none"

        if verified:
            random_text = (f"The largest of {n_random} random removals lowered it by {largest_random:.0f} points."
                           if largest_random >= 0.5 else f"None of the {n_random} random removals lowered it.")
            summary = (f"Verified ({strength}). Removing the {len(rows)} precedents lowered the model's confidence in "
                       f"'{answer}' from {before:.0%} to {after:.0%}, a fall of {fall:.0f} points. {random_text}")
        elif reason == "fall_below_threshold":
            change = f"lowered it by only {fall:.1f} points" if fall >= 0 else f"raised it by {-fall:.1f} points"
            summary = (f"Not verified. Removing the {len(rows)} precedents from the model's context {change} "
                       f"(confidence in '{answer}' went from {before:.0%} to {after:.0%}). The answer does not rest "
                       "on these orders alone; do not present them as the reason.")
        else:
            summary = (f"Not verified. Removing the precedents lowered confidence by {fall:.0f} points, but a random "
                       f"removal lowered it by {largest_random:.0f} points. The prediction is unstable, so the "
                       "precedents cannot be singled out.")
        if verified and answer_after != answer:
            summary += f" Without them the model's answer becomes '{answer_after}'."

        result = {
            "explanation_id": explanation_id, "order_id": ex["result"]["order_id"], "order_key": ex["key"],
            "model": MODELS[model], "predicted": answer,
            "confidence_before": round(before, 4), "confidence_after": round(after, 4),
            "fall_points": round(fall, 1), "answer_after": answer_after,
            "answer_changed": bool(answer_after != answer),
            "random_fall_points": [round(f, 1) for f in random_falls],
            "margin_over_random_points": round(fall - largest_random, 1),
            "verified": verified, "strength": strength, "reason_not_verified": reason,
            "precedents_removed": [int(r) for r in rows],
            "rule": (f"Verified if the fall is at least {MIN_FALL_POINTS:.0f} points and larger than every one of "
                     f"{n_random} random removals drawn from non-precedent candidates."),
            "summary": summary, "seconds": round(time.time() - t0, 1),
        }
        ex["result"]["status"] = "verified" if verified else "not_verified"
        return result


def _plain(d: dict) -> dict:
    """Make values JSON-friendly."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = None if np.isnan(v) else float(v)
        out[str(k)] = v
    return out

"""Data preparation: turn a table into a prediction task (context rows + new rows).

The SALT recipe below is the same one used in the Kaggle experiments (runs 2 and 3
and the walk-through), so order numbers should line up with those notebooks.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

SALT_TARGETS = [
    "SALESOFFICE", "SALESGROUP", "CUSTOMERPAYMENTTERMS", "SHIPPINGCONDITION",
    "PLANT", "SHIPPINGPOINT", "HEADERINCOTERMSCLASSIFICATION", "ITEMINCOTERMSCLASSIFICATION",
]
SALT_ID_COLUMNS = ["SALESDOCUMENT", "SALESDOCUMENTITEM"]


@dataclass
class Task:
    """A prediction task: past rows with known answers, and new rows to predict."""

    name: str
    target: str
    features: list[str]
    categorical: list[str]
    X_ctx: pd.DataFrame          # past rows, as given to the model
    y_ctx: np.ndarray            # their known answers
    X_new: pd.DataFrame          # new rows (held out)
    y_new: np.ndarray            # their true answers, for reference only
    E_ctx: np.ndarray = field(repr=False, default=None)   # numeric copy, for nearest-row search
    E_new: np.ndarray = field(repr=False, default=None)
    encoder: OrdinalEncoder = field(repr=False, default=None)
    scaler: StandardScaler = field(repr=False, default=None)

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        """Numeric, scaled copy of rows; used only to find similar rows."""
        out = frame[self.features].copy()
        if self.categorical:
            out[self.categorical] = self.encoder.transform(out[self.categorical])
        out = out.astype(float).fillna(-1)
        return self.scaler.transform(out)

    def row_from_fields(self, fields: dict) -> pd.DataFrame:
        """Build a one-row frame from a dict of field values. Missing fields are left empty."""
        unknown = sorted(set(fields) - set(self.features))
        if unknown:
            raise ValueError(f"Unknown fields {unknown}. Known fields: {self.features}")
        row = {}
        for c in self.features:
            v = fields.get(c)
            if c in self.categorical:
                row[c] = "missing" if v is None else str(v)
            else:
                row[c] = np.nan if v is None else float(v)
        frame = pd.DataFrame([row], columns=self.features)
        for c in self.categorical:
            frame[c] = frame[c].astype(object)
        return frame


def _is_text(col: pd.Series) -> bool:
    t = col.dtype
    return (pd.api.types.is_object_dtype(t) or pd.api.types.is_string_dtype(t)
            or isinstance(t, pd.CategoricalDtype))


def build_task(df: pd.DataFrame, target: str, *, name: str = "table", drop: list[str] | None = None,
               n_context: int = 2000, n_new: int = 1000, max_classes: int = 10,
               date_column: str | None = "auto", seed: int = 0) -> Task:
    """Clean a table and split it into past rows (context) and new rows.

    If a date column is found the split is by time (oldest 80% are the past);
    otherwise it is random.
    """
    df = df.copy()
    df[target] = df[target].astype(object)
    df = df[df[target].notna()]
    df[target] = df[target].astype(str).str.strip()
    df = df[df[target] != ""]
    df = df.drop(columns=[c for c in (drop or []) if c in df.columns])
    df = df.drop(columns=[c for c in df.columns if c.startswith("__")])

    top = df[target].value_counts().index[:max_classes]
    df = df[df[target].isin(top)]

    if date_column == "auto":
        date_column = next((c for c in df.columns if "DATE" in c.upper() and c != target), None)
    if date_column is not None:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.sort_values(date_column)
    else:
        df = df.sample(frac=1.0, random_state=seed)
    cut = int(len(df) * 0.8)
    past, future = df.iloc[:cut], df.iloc[cut:]

    ctx = past.sample(n=min(n_context, len(past)), random_state=seed)
    new = future[future[target].isin(ctx[target].unique())]
    new = new.sample(n=min(n_new, len(new)), random_state=seed)

    features = [c for c in df.columns if c != target]
    for c in list(features):
        nun = ctx[c].nunique(dropna=True)
        if nun <= 1 or (_is_text(ctx[c]) and nun > 0.95 * len(ctx)):
            features.remove(c)

    def prep(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[features].copy()
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = (out[c] - pd.Timestamp("1970-01-01")).dt.days
            elif _is_text(out[c]):
                out[c] = out[c].astype(object).where(out[c].notna(), "missing").astype(str).astype(object)
        return out.reset_index(drop=True)

    X_ctx, X_new = prep(ctx), prep(new)
    categorical = [c for c in features if _is_text(X_ctx[c])]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if categorical:
        encoder.fit(X_ctx[categorical])
    task = Task(name=name, target=target, features=features, categorical=categorical,
                X_ctx=X_ctx, y_ctx=ctx[target].to_numpy(), X_new=X_new, y_new=new[target].to_numpy(),
                encoder=encoder)
    raw = X_ctx.copy()
    if categorical:
        raw[categorical] = encoder.transform(raw[categorical])
    task.scaler = StandardScaler().fit(raw.astype(float).fillna(-1))
    task.E_ctx, task.E_new = task.encode(X_ctx), task.encode(X_new)
    return task


def _salt_frame(sample_rows: int, seed: int) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("SAP/SALT", "joined_table", split="train", token=os.environ.get("HF_TOKEN"))
    df = ds.to_pandas()
    return df.sample(n=min(sample_rows, len(df)), random_state=seed)


def load_task(dataset: str = "salt", *, target: str | None = None, csv_path: str | None = None,
              n_context: int = 2000, n_new: int = 1000, seed: int = 0,
              cache_dir: str | None = None) -> Task:
    """Load a task, using a disk cache so the server starts quickly after the first run."""
    cache = Path(cache_dir or os.environ.get("PRECEDENT_CACHE_DIR", Path.home() / ".cache" / "precedent"))
    cache.mkdir(parents=True, exist_ok=True)
    key_src = f"{dataset}|{target}|{csv_path}|{n_context}|{n_new}|{seed}|v1"
    if csv_path:
        key_src += f"|{os.path.getmtime(csv_path)}"
    path = cache / f"task_{hashlib.sha256(key_src.encode()).hexdigest()[:16]}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    if dataset == "salt":
        target = target or "CUSTOMERPAYMENTTERMS"
        df = _salt_frame(300_000, seed)
        drop = [c for c in SALT_TARGETS if c != target] + SALT_ID_COLUMNS
        task = build_task(df, target, name="SAP SALT", drop=drop, n_context=n_context, n_new=n_new, seed=seed)
    elif dataset == "csv":
        if not csv_path or not target:
            raise ValueError("dataset='csv' needs both csv_path and target.")
        task = build_task(pd.read_csv(csv_path), target, name=Path(csv_path).name,
                          n_context=n_context, n_new=n_new, seed=seed)
    else:
        raise ValueError("dataset must be 'salt' or 'csv'.")
    with open(path, "wb") as f:
        pickle.dump(task, f)
    return task

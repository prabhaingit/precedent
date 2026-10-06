"""Append-only audit ledger.

One JSON object per line. Each entry carries the hash of the entry before it, so
editing or deleting an earlier line breaks the chain and `check_chain` reports it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class Ledger:
    def __init__(self, path: str | os.PathLike = "precedent_ledger.jsonl"):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, tool: str, inputs: dict, output: dict) -> dict:
        """Add one entry and return it."""
        with self._lock:
            entries = self._entries()
            entry = {
                "seq": len(entries) + 1,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tool": tool,
                "input": inputs,
                "output": output,
                "prev_hash": entries[-1]["hash"] if entries else None,
            }
            entry["hash"] = _digest(entry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            return entry

    def check_chain(self) -> dict:
        """Recompute every hash. Returns whether the ledger is intact."""
        prev = None
        for e in self._entries():
            if e.get("prev_hash") != prev or e.get("hash") != _digest(e):
                return {"intact": False, "first_bad_seq": e.get("seq")}
            prev = e["hash"]
        return {"intact": True, "first_bad_seq": None}

    def read(self, last_n: int = 10, order_id: int | None = None, tool: str | None = None) -> list[dict]:
        entries = self._entries()
        if order_id is not None:
            entries = [e for e in entries if e.get("output", {}).get("order_id") == order_id]
        if tool is not None:
            entries = [e for e in entries if e.get("tool") == tool]
        return entries[-last_n:]

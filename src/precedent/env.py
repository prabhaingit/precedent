"""Load secrets from a .env file, so tokens need not be typed into the terminal."""

from __future__ import annotations

import os
from pathlib import Path


def candidate_paths() -> list[Path]:
    paths = []
    if os.environ.get("PRECEDENT_ENV_FILE"):
        paths.append(Path(os.environ["PRECEDENT_ENV_FILE"]))
    paths.append(Path.cwd() / ".env")
    paths.append(Path(__file__).resolve().parents[2] / ".env")   # repo root
    return paths


def load_env(path: str | os.PathLike | None = None) -> Path | None:
    """Load the first .env file found. Returns its path, or None if there is none."""
    for p in ([Path(path)] if path else candidate_paths()):
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].strip()
            if key and key not in os.environ:
                os.environ[key] = value
        return p
    return None
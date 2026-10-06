"""Precedent: row-level, removal-verified explanations for in-context tabular models."""

from .data import Task, build_task, load_task
from .engine import Engine
from .ledger import Ledger

__all__ = ["Task", "build_task", "load_task", "Engine", "Ledger"]
__version__ = "0.1.0"

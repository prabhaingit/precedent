"""A Precedent MCP server backed by a stand-in model and synthetic data. For tests only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_precedent import StandIn, make_frame          # noqa: E402

from precedent import Engine, Ledger, build_task         # noqa: E402
from precedent.server import build_server                # noqa: E402

if __name__ == "__main__":
    ledger = sys.argv[sys.argv.index("--ledger") + 1] if "--ledger" in sys.argv else "standin_ledger.jsonl"
    engine = Engine(build_task(make_frame(), "TERMS", n_context=600, n_new=100), model_factory=lambda model: StandIn())
    build_server(lambda: engine, Ledger(ledger)).run()

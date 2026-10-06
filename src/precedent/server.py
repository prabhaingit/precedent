"""Precedent MCP server.

Run:  precedent-mcp            (SAP SALT task, local TabPFN-3.5-Fast)
      precedent-mcp --dataset csv --csv my.csv --target my_column

The server speaks MCP over stdio. Nothing may be printed to stdout except
protocol messages, so library output is redirected to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys

try:                                   # MCP Python SDK v2
    from mcp.server.mcpserver import MCPServer
except ImportError:                    # MCP Python SDK v1
    from mcp.server.fastmcp import FastMCP as MCPServer

from .data import load_task
from .env import load_env
from .engine import MODELS, Engine
from .ledger import Ledger

log = logging.getLogger("precedent")

INSTRUCTIONS = """Precedent explains predictions of the TabPFN-3.5 tabular foundation model by naming
the past rows a prediction rests on ("precedents"), and checks each explanation by removing those rows
and asking the model again.

Typical flow: list_orders -> predict -> explain -> verify -> (ledger to review).
Always call verify before telling a user why the model gave an answer. If verify returns
verified=false, say that the explanation could not be confirmed and do not present the
precedents as the reason. Report the size of the fall along with the verdict."""


def build_server(engine_loader, ledger: Ledger) -> MCPServer:
    """Create the MCP server. `engine_loader` returns the Engine (loaded on first use)."""
    server = MCPServer("precedent", instructions=INSTRUCTIONS)

    def run(tool: str, inputs: dict, fn):
        with contextlib.redirect_stdout(sys.stderr):
            try:
                output = fn(engine_loader())
            except ValueError as e:
                output = {"error": str(e)}
        entry = ledger.append(tool, inputs, _for_ledger(tool, output))
        output["ledger_seq"] = entry["seq"]
        return output

    @server.tool()
    def list_orders(start: int = 0, count: int = 5) -> dict:
        """List new (held-out) orders with their fields, so you can choose one to predict or explain.

        Args:
            start: index of the first order to list.
            count: how many orders to list (keep this small).
        """
        with contextlib.redirect_stdout(sys.stderr):
            return engine_loader().list_orders(start, min(count, 25))

    @server.tool()
    def predict(order_id: int | None = None, fields: dict | None = None, model: str = "fast") -> dict:
        """Predict the target value for one order with the full model (all past orders in context).

        Give either order_id (a held-out order, see list_orders) or fields (a dict of field values
        for a new order). Returns the predicted value, the model's confidence and the runners-up.

        Args:
            order_id: index of a held-out order.
            fields: field values for a new order, as an alternative to order_id.
            model: "fast" (TabPFN-3.5-Fast) or "base" (TabPFN-3.5). Both run locally.
        """
        return run("predict", {"order_id": order_id, "fields": fields, "model": model},
                   lambda e: e.predict(order_id, fields, model))

    @server.tool()
    def explain(order_id: int | None = None, fields: dict | None = None, model: str = "fast",
                candidates: int = 60, rounds: int = 180, top_k: int = 10) -> dict:
        """Find the past orders that the model's prediction seems to rest on (the precedents).

        Takes the most similar past orders as candidates, repeatedly hides a random half and
        re-asks the model, then scores each candidate by how much it raises the model's confidence.
        Slow: roughly one minute on a GPU with the defaults. The result is UNVERIFIED; call
        verify with the returned explanation_id before presenting the precedents as the reason.

        Args:
            order_id: index of a held-out order.
            fields: field values for a new order, as an alternative to order_id.
            model: "fast" or "base".
            candidates: how many similar past orders to consider.
            rounds: hide-and-ask rounds; about three per candidate gives stable scores.
            top_k: how many precedents to return.
        """
        return run("explain", {"order_id": order_id, "fields": fields, "model": model,
                               "candidates": candidates, "rounds": rounds, "top_k": top_k},
                   lambda e: e.explain(order_id, fields, model, candidates, rounds, top_k))

    @server.tool()
    def verify(explanation_id: str | None = None, order_id: int | None = None,
               model: str = "fast", n_random: int = 5) -> dict:
        """Check an explanation against the full model.

        Removes the precedents from all past orders and asks the model again, then does the same
        with random non-precedent orders for comparison. Returns the fall in confidence, a
        verified true/false verdict, a strength label, and a plain-language summary to relay.

        Args:
            explanation_id: id returned by explain. Preferred.
            order_id: alternatively, a held-out order; its latest explanation is used, or one is created.
            model: used only with order_id. "fast" or "base".
            n_random: number of random removals to compare against.
        """
        return run("verify", {"explanation_id": explanation_id, "order_id": order_id,
                              "model": model, "n_random": n_random},
                   lambda e: e.verify(explanation_id, order_id, model, n_random))

    @server.tool()
    def ledger_entries(last_n: int = 10, order_id: int | None = None, tool: str | None = None) -> dict:
        """Read the audit ledger: every predict, explain and verify call, in order.

        Also reports whether the ledger's hash chain is intact (it breaks if an earlier
        entry was edited or deleted).

        Args:
            last_n: how many of the most recent entries to return.
            order_id: only entries about this order.
            tool: only entries from this tool ("predict", "explain" or "verify").
        """
        return {"path": str(ledger.path), "chain": ledger.check_chain(),
                "entries": ledger.read(min(last_n, 50), order_id, tool)}

    return server


def _for_ledger(tool: str, output: dict) -> dict:
    """Keep ledger entries compact: drop the bulky per-precedent field dumps."""
    if tool == "explain" and "precedents" in output:
        slim = dict(output)
        slim["precedents"] = [{k: p[k] for k in ("rank", "past_order", "influence", "value", "matching_fields")}
                              for p in output["precedents"]]
        return slim
    return dict(output)


def main() -> None:
    env_file = load_env()
    parser = argparse.ArgumentParser(description="Precedent MCP server")
    parser.add_argument("--dataset", default=os.environ.get("PRECEDENT_DATASET", "salt"), choices=["salt", "csv"])
    parser.add_argument("--csv", default=os.environ.get("PRECEDENT_CSV"), help="path to a CSV file (dataset=csv)")
    parser.add_argument("--target", default=os.environ.get("PRECEDENT_TARGET"), help="column to predict")
    parser.add_argument("--ledger", default=os.environ.get("PRECEDENT_LEDGER", "precedent_ledger.jsonl"))
    parser.add_argument("--n-context", type=int, default=int(os.environ.get("PRECEDENT_N_CONTEXT", 2000)))
    parser.add_argument("--n-new", type=int, default=int(os.environ.get("PRECEDENT_N_NEW", 1000)))
    parser.add_argument("--n-estimators", type=int, default=int(os.environ.get("PRECEDENT_N_ESTIMATORS", 1)))
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="precedent: %(message)s")
    log.info("tokens loaded from %s", env_file) if env_file else log.info("no .env file found; using the environment")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    state: dict = {}

    def engine_loader() -> Engine:
        if "engine" not in state:
            log.info("loading task (%s); the first run downloads data and model weights", args.dataset)
            task = load_task(args.dataset, target=args.target, csv_path=args.csv,
                             n_context=args.n_context, n_new=args.n_new)
            state["engine"] = Engine(task, n_estimators=args.n_estimators)
            log.info("ready: %d past orders, %d new orders, models %s",
                     len(task.X_ctx), len(task.X_new), sorted(MODELS))
        return state["engine"]

    build_server(engine_loader, Ledger(args.ledger)).run()


if __name__ == "__main__":
    main()

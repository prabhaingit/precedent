"""Precedent agent: a small LangGraph workflow on top of the MCP server.

    question -> resolve -> predict -> explain -> verify -+-> report (verified)
                                                         +-> report (not verified)

The control flow is fixed, so the agent cannot skip the check. An LLM is used only to
word the final answer, and only from facts the tools returned. When an explanation is
not verified, the precedent rows are withheld from the LLM, so it cannot present them
as the reason. Without an LLM key the agent falls back to a plain template.

Tracing: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY (for example in .env) and
every run, node and MCP tool call is recorded in LangSmith.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .env import load_env
from .guardrails import check_patterns, check_scope, load_guardrails, prompt_rules

try:
    from langsmith import traceable
except ImportError:                                    # tracing is optional
    def traceable(*_a, **_k):
        return lambda fn: fn

DEFAULT_QUESTION = "Why did the model give this answer?"
DEFAULT_LLM = "anthropic:claude-opus-5-5"
TOOL_TIMEOUT_SECONDS = 900
MCP_SERVER_NAME = "precedent"


# ---------------------------------------------------------------- run trace (what the UI shows while the agent works)

def _plain(x: Any) -> Any:
    """A JSON-safe copy, so a trace can be stored and rendered without surprises."""
    try:
        return json.loads(json.dumps(x, default=str))
    except (TypeError, ValueError):
        return str(x)


class TraceLog:
    """Thread-safe record of one run: LLM turns, agent tool calls and MCP calls, in order.

    kind is "llm", "agent_tool" or "mcp". The agent runs on a background thread and the UI
    reads snapshots from its own thread.
    """

    def __init__(self):
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self.t0 = time.time()

    def start(self, kind: str, name: str, input: Any = None) -> int:
        with self._lock:
            self._events.append({"kind": kind, "name": name, "input": _plain(input), "output": None,
                                 "status": "running", "t0": time.time(), "seconds": None})
            return len(self._events) - 1

    def end(self, i: int, output: Any = None, error: str | None = None) -> None:
        with self._lock:
            e = self._events[i]
            e["seconds"] = round(time.time() - e["t0"], 1)
            e["output"] = _plain(output) if error is None else {"error": error}
            e["status"] = "error" if error else "done"

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(e) for e in self._events]


def _trace_handler(trace: TraceLog):
    """A LangChain callback that logs the chat model's turns and the agent's tool calls."""
    from langchain_core.callbacks import BaseCallbackHandler

    class Handler(BaseCallbackHandler):
        def __init__(self):
            self.ids: dict = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            self.ids[run_id] = trace.start("llm", "LLM turn", {"messages_in_context": len(messages[0])})

        def on_llm_end(self, response, *, run_id, **kwargs):
            i = self.ids.pop(run_id, None)
            if i is None:
                return
            msg = response.generations[0][0].message
            calls = [{"tool": c["name"], "args": c["args"]} for c in (getattr(msg, "tool_calls", None) or [])]
            trace.end(i, {"decided_to_call": calls} if calls else {"wrote_answer": _message_text(msg)[:300]})

        def on_llm_error(self, error, *, run_id, **kwargs):
            i = self.ids.pop(run_id, None)
            if i is not None:
                trace.end(i, error=f"{type(error).__name__}: {error}")

        def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
            name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
            self.ids[run_id] = trace.start("agent_tool", name, inputs if inputs is not None else input_str)

        def on_tool_end(self, output, *, run_id, **kwargs):
            i = self.ids.pop(run_id, None)
            if i is not None:
                out = getattr(output, "content", output)
                trace.end(i, out)

        def on_tool_error(self, error, *, run_id, **kwargs):
            i = self.ids.pop(run_id, None)
            if i is not None:
                trace.end(i, error=f"{type(error).__name__}: {error}")

    return Handler()


# ---------------------------------------------------------------- MCP backend

class MCPBackend:
    """Starts the Precedent MCP server as a subprocess and calls its tools."""

    def __init__(self, ledger: str | None = None, extra_args: list[str] | None = None,
                 command: list[str] | None = None):
        self.trace: TraceLog | None = None                 # set by AgentRuntime for the run in progress
        cmd = list(command or [sys.executable, "-m", "precedent.server"])
        if ledger:
            cmd += ["--ledger", ledger]
        self._cmd = cmd + (extra_args or [])
        self._stack = None
        self._call = None

    async def __aenter__(self) -> "MCPBackend":
        from contextlib import AsyncExitStack
        import mcp
        params = mcp.StdioServerParameters(command=self._cmd[0], args=self._cmd[1:], env=dict(os.environ))
        self._stack = AsyncExitStack()
        if hasattr(mcp, "Client"):                                     # MCP SDK v2
            client = await self._stack.enter_async_context(mcp.Client(params))
            self._call = lambda tool, args: client.call_tool(tool, args, read_timeout_seconds=TOOL_TIMEOUT_SECONDS)
        else:                                                          # MCP SDK v1
            from datetime import timedelta
            from mcp.client.stdio import stdio_client
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(mcp.ClientSession(read, write))
            await session.initialize()
            self._call = lambda tool, args: session.call_tool(
                tool, args, read_timeout_seconds=timedelta(seconds=TOOL_TIMEOUT_SECONDS))
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def call(self, tool: str, **arguments: Any) -> dict:
        arguments = {k: v for k, v in arguments.items() if v is not None}

        @traceable(run_type="tool", name=f"mcp:{tool}")
        async def _go(arguments: dict) -> dict:
            res = await self._call(tool, arguments)
            data = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
            if data is None:
                text = res.content[0].text if res.content else "{}"
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = {"error": text}
            if isinstance(data, dict) and set(data) == {"result"} and isinstance(data["result"], dict):
                data = data["result"]
            return data

        trace = self.trace
        i = trace.start("mcp", f"{MCP_SERVER_NAME}.{tool}", arguments) if trace else None
        try:
            data = await _go(arguments)
        except BaseException as e:
            if trace:
                trace.end(i, error=f"{type(e).__name__}: {e}")
            raise
        if trace:
            err = data.get("error") if isinstance(data, dict) else None
            trace.end(i, data, error=str(err) if err else None)
        return data


# ---------------------------------------------------------------- LLM wording

def get_llm(spec: str | None = "auto"):
    """Return a chat model, or None to use the template.

    spec: "auto" (PRECEDENT_LLM, else Anthropic if a key is set), "none", or a
    LangChain model string such as "anthropic:claude-haiku-4-5-20251001" or "openai:gpt-4o-mini".
    """
    if spec in (None, "none", "off"):
        return None
    if spec == "auto":
        spec = os.environ.get("PRECEDENT_LLM") or (DEFAULT_LLM if os.environ.get("ANTHROPIC_API_KEY") else None)
        if not spec or spec in ("none", "off"):
            return None
    from langchain.chat_models import init_chat_model
    return init_chat_model(spec, max_tokens=1500)     # no temperature: newer Claude models reject it


WRITER_RULES = (
    "You write the final answer for a tool that explains predictions of a tabular model. "
    "Use ONLY the facts in the JSON below; never add causes, business reasons or numbers that are not there. "
    "Write 3 to 5 plain sentences for a business reader. Always state whether the explanation was verified "
    "and the size of the fall in confidence, in points. Do not use bullet points or headings. "
    "Accuracy rules: (1) The check removed ALL the precedents together (see precedents_removed_together), "
    "never one order alone, so never attribute the fall to a single past order. (2) The random removals "
    "removed the same number of random past ORDERS, not fields. (3) Give percentages as whole numbers. "
    "(4) If strength is 'weak', say plainly that the effect is small. (5) A passed check is supporting "
    "evidence, not proof: do not write 'confirms', 'proves', 'demonstrates' or 'genuine'.")

VERIFIED_TASK = (
    "The explanation WAS verified. Say what the model predicted and how sure it was, name the main precedent "
    "(past order number, its value, and which fields it shares with the new order), then report the check: "
    "confidence before and after removal, the fall, and how the random removals compared. If answer_changed "
    "is true, say what the answer became.")

UNVERIFIED_TASK = (
    "The explanation was NOT verified. Say what the model predicted and how sure it was, then say clearly that "
    "the candidate precedents could not be confirmed as the reason, and why (use reason_not_verified and the "
    "numbers). Do not describe or name any past orders.")


def _template(state: dict) -> str:
    p, v = state["prediction"], state["verification"]
    head = f"The model predicted '{p['predicted']}' for this order and was {p['confidence']:.0%} sure. "
    if v.get("verified"):
        top = state["explanation"]["precedents"][0]
        shared = ", ".join(top["matching_fields"][:6]) or "no fields"
        return (head + f"The answer rests mainly on past order {top['past_order']} (value '{top['value']}'), "
                f"which shares {shared} with the new order. " + v["summary"])
    return head + v["summary"]


async def _write(llm, task: str, facts: dict, state: dict) -> tuple[str, str]:
    if llm is None:
        return _template(state), "template"
    try:
        msg = await llm.ainvoke([("system", WRITER_RULES + " " + task),
                                 ("human", f"Question: {state['question']}\n\nFacts:\n{json.dumps(facts, indent=1)}")])
        text = msg.content if isinstance(msg.content, str) else "".join(
            part.get("text", "") for part in msg.content if isinstance(part, dict))
        return text.strip(), "llm"
    except Exception as e:                              # never lose the result because wording failed
        return _template(state) + f" (LLM wording unavailable: {type(e).__name__}.)", "template"


# ---------------------------------------------------------------- the graph

class AgentState(TypedDict, total=False):
    question: str
    order_id: int | None
    model: str
    order: dict
    prediction: dict
    explanation: dict
    verification: dict
    answer: str
    writer: str
    error: str
    steps: list


def build_graph(backend, llm=None):
    def step(state: AgentState, name: str, t0: float, ledger_seq: int | None = None) -> list:
        return state.get("steps", []) + [{"step": name, "seconds": round(time.time() - t0, 1), "ledger_seq": ledger_seq}]

    async def resolve(state: AgentState) -> dict:
        t0 = time.time()
        order_id = state.get("order_id")
        if order_id is None:
            m = re.search(r"order\s*(?:number|no\.?|#)?\s*(\d+)", state.get("question", ""), re.I)
            if not m:
                return {"error": "No order number found. Ask about a specific order, for example 'order 7'."}
            order_id = int(m.group(1))
        listing = await backend.call("list_orders", start=order_id, count=1)
        if listing.get("error") or not listing.get("orders"):
            return {"error": listing.get("error") or f"Order {order_id} does not exist."}
        return {"order_id": order_id, "order": listing["orders"][0], "steps": step(state, "resolve", t0)}

    async def predict(state: AgentState) -> dict:
        t0 = time.time()
        out = await backend.call("predict", order_id=state["order_id"], model=state.get("model", "fast"))
        if out.get("error"):
            return {"error": out["error"]}
        return {"prediction": out, "steps": step(state, "predict", t0, out.get("ledger_seq"))}

    async def explain(state: AgentState) -> dict:
        t0 = time.time()
        out = await backend.call("explain", order_id=state["order_id"], model=state.get("model", "fast"))
        if out.get("error"):
            return {"error": out["error"]}
        return {"explanation": out, "steps": step(state, "explain", t0, out.get("ledger_seq"))}

    async def verify(state: AgentState) -> dict:
        t0 = time.time()
        out = await backend.call("verify", explanation_id=state["explanation"]["explanation_id"])
        if out.get("error"):
            return {"error": out["error"]}
        return {"verification": out, "steps": step(state, "verify", t0, out.get("ledger_seq"))}

    async def report_verified(state: AgentState) -> dict:
        t0 = time.time()
        ex, v = state["explanation"], state["verification"]
        facts = {
            "prediction": {k: state["prediction"][k] for k in ("predicted", "confidence", "true_value", "model")},
            "main_precedent": {k: ex["precedents"][0][k] for k in ("past_order", "influence", "value", "matching_fields")},
            "main_precedent_share_of_influence": ex["top_precedent_share"],
            "other_precedents": [{"past_order": p["past_order"], "influence": p["influence"], "value": p["value"]}
                                 for p in ex["precedents"][1:4]],
            "check": {k: v[k] for k in ("confidence_before", "confidence_after", "fall_points", "answer_after",
                                        "answer_changed", "random_fall_points", "verified", "strength")},
            "precedents_removed_together": len(v["precedents_removed"]),
        }
        text, writer = await _write(llm, VERIFIED_TASK, facts, state)
        return {"answer": text, "writer": writer, "steps": step(state, "report (verified)", t0)}

    async def report_unverified(state: AgentState) -> dict:
        t0 = time.time()
        v = state["verification"]
        facts = {   # the precedent rows are deliberately left out
            "prediction": {k: state["prediction"][k] for k in ("predicted", "confidence", "true_value", "model")},
            "check": {k: v[k] for k in ("confidence_before", "confidence_after", "fall_points", "random_fall_points",
                                        "verified", "reason_not_verified")},
        }
        text, writer = await _write(llm, UNVERIFIED_TASK, facts, state)
        return {"answer": text, "writer": writer, "steps": step(state, "report (not verified)", t0)}

    async def failed(state: AgentState) -> dict:
        return {"answer": f"I could not complete the request: {state['error']}", "writer": "template"}

    def ok(nxt: str):
        return lambda state: "failed" if state.get("error") else nxt

    g = StateGraph(AgentState)
    for name, fn in [("resolve", resolve), ("predict", predict), ("explain", explain), ("verify", verify),
                     ("report_verified", report_verified), ("report_unverified", report_unverified), ("failed", failed)]:
        g.add_node(name, fn)
    g.add_edge(START, "resolve")
    g.add_conditional_edges("resolve", ok("predict"), ["predict", "failed"])
    g.add_conditional_edges("predict", ok("explain"), ["explain", "failed"])
    g.add_conditional_edges("explain", ok("verify"), ["verify", "failed"])
    g.add_conditional_edges(
        "verify",
        lambda s: "failed" if s.get("error") else ("report_verified" if s["verification"]["verified"] else "report_unverified"),
        ["report_verified", "report_unverified", "failed"])
    for end in ("report_verified", "report_unverified", "failed"):
        g.add_edge(end, END)
    return g.compile()


async def run_agent(backend, question: str = DEFAULT_QUESTION, order_id: int | None = None,
                    model: str = "fast", llm="auto", trace: TraceLog | None = None) -> dict:
    """Run the workflow once and return the final state."""
    chat = get_llm(llm) if isinstance(llm, (str, type(None))) else llm
    if chat is not None and question != DEFAULT_QUESTION:       # free-text questions get the same topic check
        refusal = await _check_guardrails(chat, [{"role": "user", "content": question}], load_guardrails(), trace)
        if refusal:
            return {"answer": refusal, "error": "out_of_scope", "order_id": order_id, "steps": []}
    graph = build_graph(backend, chat)
    config = {"run_name": "precedent-agent", "tags": ["precedent", f"model:{model}"],
              "metadata": {"order_id": order_id, "model": model, "llm": getattr(chat, "model", None) or "template"}}
    if trace is not None:
        config["callbacks"] = [_trace_handler(trace)]
    return await graph.ainvoke({"question": question, "order_id": order_id, "model": model, "steps": []}, config)


def tracing_status() -> dict:
    on = os.environ.get("LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", "")).lower() == "true"
    key = bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))
    return {"enabled": on and key, "project": os.environ.get("LANGSMITH_PROJECT", "default")}


# ---------------------------------------------------------------- conversational agent

CHAT_RULES = """You are the Precedent assistant. You answer questions about predictions made by the TabPFN-3.5
tabular model on sales orders, using the tools. Each order has a number (0 to 999).

Tools: list_orders (see orders and their fields), predict_order (the model's answer only),
explain_order (the full checked explanation: predict, find precedents, verify), read_ledger (the audit log).

Rules:
- For any "why" question about an order, call explain_order. It always runs the check; you cannot skip it.
- Use only what the tools return. Never invent reasons, fields or numbers.
- Always report the verdict and the fall in confidence, in points. Give percentages as whole numbers.
- If verified is false, say the explanation could not be confirmed. No precedents are returned in that case.
- The check removes ALL the precedents together, never one order alone. Random removals remove random past
  orders. A passed check is supporting evidence, not proof. If strength is "weak", say the effect is small.
- If the user gives no order number, ask for one or offer to list some orders.
- Past orders "carry" or "have" a value; they do not "predict". Do not say precedents "pushed", "drove" or
  "caused" the answer. Say the answer "rests partly on" them, and for a weak result say most of it is unexplained.
- A message that only names an order (for example "order 5") is unclear: ask whether the user wants the
  prediction or the explanation.
- Keep answers short: a few plain sentences, no headings.
- Earlier tool calls and their results appear in the conversation history. Rely on them. Never say that you
  did or did not call a tool, or that you made something up, unless the history shows it.

Definitions you may use: a precedent is a past order with high influence on the model's answer. Influence is
how much the model's confidence rises when that past order is in view. Verified means removing the precedents
from the model's context lowered its confidence by at least 5 points and by more than every random removal."""


def build_chat_agent(backend, llm, model: str = "fast", collected: list | None = None, rules: str = CHAT_RULES):
    """A tool-calling agent for free-form questions. Its explain tool is the fixed, checked workflow."""
    from langchain_core.tools import tool
    workflow = build_graph(backend, None)            # template wording inside; the chat LLM writes the reply

    @tool
    async def list_orders(start: int = 0, count: int = 5) -> dict:
        """List held-out orders and their fields. Use to find or inspect orders."""
        return await backend.call("list_orders", start=start, count=min(count, 10))

    @tool
    async def predict_order(order_id: int) -> dict:
        """The model's predicted value and confidence for one order. No explanation."""
        return await backend.call("predict", order_id=order_id, model=model)

    @tool
    async def explain_order(order_id: int) -> dict:
        """Explain why the model gave its answer for one order. Runs predict, finds precedents and
        verifies them against the full model. Slow (about a minute). Returns the verdict and, only if
        verified, the precedents."""
        state = await workflow.ainvoke({"question": DEFAULT_QUESTION, "order_id": order_id, "model": model, "steps": []})
        if state.get("error"):
            return {"error": state["error"]}
        if collected is not None:
            collected.append(state)
        v, ex = state["verification"], state["explanation"]
        out = {"order_id": order_id,
               "prediction": {k: state["prediction"][k] for k in ("predicted", "confidence", "true_value")},
               "check": {k: v[k] for k in ("verified", "strength", "confidence_before", "confidence_after", "fall_points",
                                           "answer_after", "answer_changed", "random_fall_points", "reason_not_verified")},
               "precedents_removed_together": len(v["precedents_removed"])}
        if v["verified"]:                              # precedents are released only when the check passes
            out["top_precedent_share_of_influence"] = ex["top_precedent_share"]
            out["precedents"] = [{k: p[k] for k in ("rank", "past_order", "influence", "value", "matching_fields")}
                                 for p in ex["precedents"][:5]]
        return out

    @tool
    async def read_ledger(last_n: int = 10, order_id: int | None = None) -> dict:
        """Read the audit ledger: past predict, explain and verify calls, and whether its hash chain is intact."""
        led = await backend.call("ledger_entries", last_n=min(last_n, 20), order_id=order_id)
        return {"chain": led["chain"], "entries": [
            {"seq": e["seq"], "time": e["time"], "tool": e["tool"], "order_id": e["output"].get("order_id"),
             "result": e["output"].get("summary") or e["output"].get("predicted") or e["output"].get("error")}
            for e in led["entries"]]}

    tools = [list_orders, predict_order, explain_order, read_ledger]
    try:
        from langchain.agents import create_agent
        return create_agent(llm, tools, system_prompt=rules)
    except ImportError:                                # older LangChain / LangGraph
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(llm, tools, prompt=rules)


def _turn_text(m) -> str:
    return m["content"] if isinstance(m, dict) else _message_text(m)


async def _check_guardrails(llm, messages: list, g, trace: TraceLog | None = None) -> str | None:
    """Returns the refusal message if the latest message is off topic, else None. Logged in the trace."""
    if not g.enabled:
        return None
    latest = _turn_text(messages[-1])
    context = [t for m in messages[:-1]
               if (m.get("role") if isinstance(m, dict) else getattr(m, "type", None)) in ("user", "human", "assistant", "ai")
               and (t := _turn_text(m))]
    i = trace.start("guardrail", "scope check", {"rules_file": g.source}) if trace else None
    reason = check_patterns(latest, g)
    if reason is None and not await check_scope(llm, latest, context, g):
        reason = "outside the Precedent topic"
    if trace:
        trace.end(i, {"allowed": reason is None, **({"reason": reason} if reason else {}),
                      **({"warning": g.problem} if g.problem else {})})
    return g.refusal_message if reason else None


def _message_text(msg) -> str:
    """The visible text of a model message, ignoring thinking and tool-call blocks."""
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    return "".join(part.get("text", "") for part in content
                   if isinstance(part, dict) and part.get("type", "text") == "text").strip()


async def run_chat(backend, messages: list[dict], model: str = "fast", llm="auto",
                   trace: TraceLog | None = None) -> dict:
    """Answer the latest message in a conversation. `messages` is [{"role": "user"|"assistant", "content": str}]."""
    chat = get_llm(llm) if isinstance(llm, (str, type(None))) else llm
    if chat is None:
        return {"answer": "The chat needs an LLM. Add ANTHROPIC_API_KEY (or PRECEDENT_LLM) to your .env file. "
                          "The 'Explain an order' tab works without one.", "results": []}
    g = load_guardrails()                              # re-read each time, so edits to the file apply at once
    refusal = await _check_guardrails(chat, messages, g, trace)
    if refusal:                                        # no "history" key: the refused message stays out of the context
        return {"answer": refusal, "results": [], "blocked": True}
    collected: list = []
    agent = build_chat_agent(backend, chat, model, collected, CHAT_RULES + "\n\n" + prompt_rules(g))
    config = {"run_name": "precedent-chat", "tags": ["precedent", "chat", f"model:{model}"], "recursion_limit": 12}
    if trace is not None:
        config["callbacks"] = [_trace_handler(trace)]
    turns = [(m["role"], m["content"]) if isinstance(m, dict) else m for m in messages]
    out = await agent.ainvoke({"messages": turns}, config)
    text = _message_text(out["messages"][-1])
    if not text and collected:          # the model gave no wording: fall back to the checked summary
        text = collected[-1]["answer"]
    if not text:
        text = "I ran the request but did not get a written answer. Please ask again."
    return {"answer": text, "results": collected, "history": out["messages"]}


# ---------------------------------------------------------------- long-lived runtime (used by the UI)

class AgentRuntime:
    """Keeps one MCP server process alive on a background thread, so the model loads once."""

    def __init__(self, ledger: str | None = None, extra_args: list[str] | None = None, command: list[str] | None = None):
        self._args = dict(ledger=ledger, extra_args=extra_args, command=command)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop: asyncio.Event | None = None
        self.backend: MCPBackend | None = None
        threading.Thread(target=self._thread, daemon=True, name="precedent-agent").start()
        self._ready.wait(timeout=120)
        if self._error:
            raise RuntimeError(f"Could not start the Precedent MCP server: {self._error}")

    def _thread(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with MCPBackend(**self._args) as backend:
                self.backend = backend
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:                        # surface start-up failures to the caller
            self._error = e
            self._ready.set()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _run_traced(self, make_coro, on_progress=None) -> dict:
        """Run a coroutine on the agent thread, calling `on_progress(events)` from this thread while it works.
        The result carries the finished trace under "trace"."""
        trace = TraceLog()
        self.backend.trace = trace
        try:
            fut = asyncio.run_coroutine_threadsafe(make_coro(trace), self._loop)
            while True:
                try:
                    out = fut.result(timeout=0.4)
                    break
                except concurrent.futures.TimeoutError:
                    if on_progress:
                        on_progress(trace.snapshot())
        finally:
            self.backend.trace = None
        out["trace"] = trace.snapshot()
        if on_progress:
            on_progress(out["trace"])
        return out

    def ask(self, question: str = DEFAULT_QUESTION, order_id: int | None = None, model: str = "fast", llm="auto",
            on_progress=None) -> dict:
        return self._run_traced(lambda t: run_agent(self.backend, question, order_id, model, llm, t), on_progress)

    def call(self, tool: str, **arguments) -> dict:
        return self._run(self.backend.call(tool, **arguments))

    def chat(self, messages: list[dict], model: str = "fast", llm="auto", on_progress=None) -> dict:
        return self._run_traced(lambda t: run_chat(self.backend, messages, model, llm, t), on_progress)

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)


# ---------------------------------------------------------------- command line

def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Ask the Precedent agent about one order.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--order", type=int, help="order number (or mention 'order N' in the question)")
    parser.add_argument("--model", default="fast", choices=["fast", "base"])
    parser.add_argument("--llm", default="auto", help="'auto', 'none', or a LangChain model string")
    parser.add_argument("--ledger", default=os.environ.get("PRECEDENT_LEDGER", "precedent_ledger.jsonl"))
    args = parser.parse_args()

    async def go() -> dict:
        async with MCPBackend(ledger=args.ledger) as backend:
            return await run_agent(backend, args.question, args.order, args.model, args.llm)

    state = asyncio.run(go())
    print("\n" + state["answer"] + "\n")
    for s in state.get("steps", []):
        seq = f"  ledger #{s['ledger_seq']}" if s.get("ledger_seq") else ""
        print(f"  {s['step']:<22} {s['seconds']:>6.1f}s{seq}")
    t = tracing_status()
    print(f"\n  wording: {state.get('writer')}   tracing: {'on, project ' + t['project'] if t['enabled'] else 'off'}")


if __name__ == "__main__":
    main()

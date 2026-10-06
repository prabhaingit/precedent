"""Precedent demo UI.

    streamlit run src/precedent/app.py

Pick an order, press Run, and the LangGraph agent calls the MCP server
(predict -> explain -> verify) and writes the answer. The Ledger tab shows the audit log.
"""

from __future__ import annotations

import html
from datetime import datetime
import json
import os
import time

import pandas as pd
import streamlit as st

from precedent.agent import DEFAULT_QUESTION, AgentRuntime, get_llm, tracing_status
from precedent.env import load_env

st.set_page_config(page_title="Precedent", page_icon="🔎", layout="wide")
load_env()

CSS = """
<style>
.block-container { padding-top: 4.5rem; max-width: 1180px; }
.pc-kicker { font: 600 0.74rem/1 ui-monospace, Menlo, Consolas, monospace; letter-spacing: .12em;
             text-transform: uppercase; color: #4F46E5; margin-bottom: .35rem; }
.pc-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .2rem; }
.pc-sub { color: #6B7280; margin: 0 0 1.2rem; max-width: 62ch; }
.pc-verdict { border-radius: 14px; padding: 16px 20px; margin: 4px 0 14px; border: 1.5px solid; }
.pc-verdict .label { font: 700 0.78rem/1 ui-monospace, Menlo, Consolas, monospace; letter-spacing: .1em;
                     text-transform: uppercase; margin-bottom: 8px; }
.pc-verdict .text { font-size: 1.04rem; line-height: 1.55; }
.pc-ok  { background: rgba(79,70,229,.08); border-color: #4F46E5; } .pc-ok .label  { color: #4F46E5; }
.pc-no  { background: rgba(220,38,38,.07); border-color: #DC2626; } .pc-no .label  { color: #DC2626; }
.pc-bars { margin: 6px 0 4px; }
.pc-row { display: grid; grid-template-columns: 230px 1fr 64px; gap: 12px; align-items: center; margin: 7px 0;
          font-size: .9rem; }
.pc-track { background: rgba(127,127,127,.16); border-radius: 6px; height: 22px; overflow: hidden; }
.pc-fill { height: 100%; border-radius: 6px; }
.pc-num { font: 600 .9rem ui-monospace, Menlo, Consolas, monospace; text-align: right; }
.pc-cap { color: #6B7280; font-size: .86rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Starting the Precedent MCP server…")
def runtime() -> AgentRuntime:
    command = json.loads(os.environ["PRECEDENT_SERVER_CMD"]) if os.environ.get("PRECEDENT_SERVER_CMD") else None
    return AgentRuntime(ledger=os.environ.get("PRECEDENT_LEDGER", "precedent_ledger.jsonl"), command=command)


def llm_name() -> str | None:
    try:
        llm = get_llm("auto")
    except Exception:
        return None
    return None if llm is None else str(getattr(llm, "model", None) or getattr(llm, "model_name", "LLM"))


def bar(label: str, value: float, color: str) -> str:
    width = max(0.0, min(100.0, value * 100))
    return (f'<div class="pc-row"><div>{html.escape(label)}</div>'
            f'<div class="pc-track"><div class="pc-fill" style="width:{width:.1f}%;background:{color}"></div></div>'
            f'<div class="pc-num">{value:.0%}</div></div>')


def precedent_table(explanation: dict) -> pd.DataFrame:
    rows = []
    for p in explanation["precedents"]:
        row = {"#": p["rank"], "Past order": p["past_order"], "Influence": p["influence"],
               "Value": p["value"], "Fields matching": len(p["matching_fields"])}
        for field, value in p.get("fields", {}).items():
            row[field] = f"{value}  ✓" if field in p["matching_fields"] else str(value)
        rows.append(row)
    return pd.DataFrame(rows)


def show_result(state: dict, banner: bool = True) -> None:
    if state.get("error") or "verification" not in state:
        st.error(state.get("answer", "The run did not finish."))
        return
    pred, ex, v = state["prediction"], state["explanation"], state["verification"]
    verified = v["verified"]
    label = f"Verified · {v['strength']}" if verified else "Not verified"
    if banner:
        st.markdown(f'<div class="pc-verdict {"pc-ok" if verified else "pc-no"}"><div class="label">{label}</div>'
                    f'<div class="text">{html.escape(state["answer"])}</div></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model's answer", pred["predicted"], f"true value {pred['true_value']}", delta_color="off")
    c2.metric("Confidence", f"{v['confidence_before']:.0%}")
    c3.metric("After removing precedents", f"{v['confidence_after']:.0%}")
    c4.metric("Fall", f"{v['fall_points']:.0f} points",
              f"answer becomes {v['answer_after']}" if v["answer_changed"] else "answer unchanged", delta_color="off")

    st.subheader("The check")
    before = v["confidence_before"]
    bars = bar("Before: all past orders", before, "#9CA3AF")
    bars += bar(f"{len(v['precedents_removed'])} precedents removed", v["confidence_after"], "#4F46E5" if verified else "#DC2626")
    for n, fall in enumerate(v["random_fall_points"], start=1):
        bars += bar(f"Random removal {n}", before - fall / 100, "#D1D5DB")
    st.markdown(f'<div class="pc-bars">{bars}</div><div class="pc-cap">Model confidence in '
                f"'{html.escape(str(pred['predicted']))}' after each removal. {html.escape(v['rule'])}</div>",
                unsafe_allow_html=True)

    title = "Precedents" if verified else "Candidate precedents (not confirmed as the reason)"
    with st.expander(title, expanded=verified):
        st.caption(f"A tick marks a field with the same value as the new order. The top precedent holds "
                   f"{ex['top_precedent_share']:.0%} of the positive influence.")
        st.dataframe(precedent_table(ex), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left, st.expander("The new order's fields"):
        st.dataframe(pd.DataFrame(state["order"]["fields"].items(), columns=["Field", "Value"]).astype(str),
                     hide_index=True, width="stretch")
    with right, st.expander("Agent steps"):
        st.dataframe(pd.DataFrame(state["steps"]).rename(
            columns={"step": "Step", "seconds": "Seconds", "ledger_seq": "Ledger entry"}),
            hide_index=True, width="stretch")
        st.caption(f"Answer wording: {'LLM' if state.get('writer') == 'llm' else 'template'}.")
    with st.expander("Raw tool results"):
        st.json({"prediction": pred, "explanation": ex, "verification": v}, expanded=False)


TRACE_LAYER = {"guardrail": "Guardrail", "llm": "LLM", "agent_tool": "Agent tool", "mcp": "MCP tool"}


def trace_line(e: dict) -> str:
    icon = {"running": "⏳", "done": "✅", "error": "❌"}[e["status"]]
    secs = e["seconds"] if e["seconds"] is not None else round(time.time() - e["t0"], 1)
    args = ", ".join(f"{k}={v}" for k, v in e["input"].items()) if isinstance(e["input"], dict) else ""
    return f"{icon} **{TRACE_LAYER[e['kind']]}** · `{e['name']}`" + (f" ({args})" if args and e["kind"] not in ("llm", "guardrail") else "") + f" · {secs:.1f}s"


def show_trace(events: list[dict], details: bool = True) -> None:
    """One line per step. With details, each step also has a collapsed input/output viewer."""
    for e in events:
        st.markdown(trace_line(e))
        if details and e["status"] != "running":
            st.json({"input": e["input"], "output": e["output"]}, expanded=False)


def run_with_progress(label: str, call) -> dict:
    """Run `call(on_progress)` while a live status panel lists each LLM turn, agent tool and MCP call."""
    with st.status(label, expanded=True) as status:
        live = st.empty()

        def on_progress(events: list[dict]) -> None:
            with live.container():
                show_trace(events, details=False)

        result = call(on_progress)
        events = result.get("trace", [])
        total = round(time.time() - events[0]["t0"], 1) if events else 0
        status.update(label=f"Finished: {len(events)} steps, {total}s. Details are kept below.",
                      state="complete", expanded=False)
    return result


def trace_expander(events: list[dict]) -> None:
    if events:
        mcp = sum(e["kind"] == "mcp" for e in events)
        with st.expander(f"Agent run: {len(events)} steps ({mcp} MCP calls)"):
            show_trace(events)


st.markdown('<div class="pc-kicker">TabPFN-3.5 · MCP · LangGraph</div><div class="pc-title">Precedent</div>'
            '<p class="pc-sub">Which past orders does a prediction rest on, and does that explanation survive '
            'a check against the real model?</p>', unsafe_allow_html=True)

name = llm_name()
trace = tracing_status()
top_model, top_status = st.columns([1, 3], vertical_alignment="bottom")
with top_model:
    model = st.selectbox("Model", ["fast", "base"], format_func={"fast": "TabPFN-3.5-Fast", "base": "TabPFN-3.5"}.get)
with top_status:
    st.caption(f"LLM: {name or 'none configured (template wording)'}  ·  "
               f"LangSmith tracing: {'on, project ' + trace['project'] if trace['enabled'] else 'off'}")

tab_chat, tab_run, tab_ledger = st.tabs(["Ask the agent", "Explain an order", "Ledger"])

with tab_chat:
    st.caption("Ask in your own words, for example: \"Why did order 7 get these payment terms?\", "
               "\"What does the model predict for order 12?\", \"Show the ledger for order 7.\"")
    history = st.session_state.setdefault("chat", [])
    for n, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            trace_expander(msg.get("trace", []))
            for k, res in enumerate(msg.get("results", [])):
                verdict = f"verified, {res['verification']['strength']}" if res["verification"]["verified"] else "not verified"
                with st.expander(f"Details for order {res['order_id']} ({verdict})"):
                    show_result(res, banner=False)
    prompt = st.chat_input("Ask about an order")
    if prompt:
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        prior = st.session_state.get("chat_history", [])            # includes earlier tool calls and results
        reply = run_with_progress("The agent is working. An explanation takes about a minute.",
                                  lambda cb: runtime().chat(prior + [{"role": "user", "content": prompt}], model,
                                                            on_progress=cb))
        st.session_state["chat_history"] = reply.get("history", prior)
        history.append({"role": "assistant", "content": reply["answer"], "results": reply["results"],
                        "trace": reply.get("trace", [])})
        st.rerun()
    if history and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.session_state["chat_history"] = []
        st.rerun()

with tab_run:
    c_order, c_question, c_llm, c_run = st.columns([1, 3, 2, 1], vertical_alignment="bottom")
    order_id = c_order.number_input("Order number", min_value=0, max_value=999, value=7, step=1)
    question = c_question.text_input("Question", DEFAULT_QUESTION)
    use_llm = c_llm.toggle("Word the answer with an LLM", value=name is not None, disabled=name is None)
    go = c_run.button("Run", type="primary", width="stretch")
    if go:
        st.session_state["result"] = run_with_progress(
            "Predicting, finding precedents and checking them. This takes about a minute.",
            lambda cb: runtime().ask(question, int(order_id), model, "auto" if use_llm else "none", on_progress=cb))
    if "result" in st.session_state:
        show_result(st.session_state["result"])
        trace_expander(st.session_state["result"].get("trace", []))
    else:
        st.info("Choose an order above and press Run. Order 7 is a good first example.")

with tab_ledger:
    st.caption("Every predict, explain and verify call, in order. Each entry carries the hash of the one before it. "
               "Times are stored in UTC and shown here in this computer's local time.")
    if st.button("Load ledger"):
        st.session_state["ledger"] = runtime().call("ledger_entries", last_n=50)
    led = st.session_state.get("ledger")
    if led:
        chain = led["chain"]
        (st.success if chain["intact"] else st.error)(
            "Hash chain intact: no entry has been edited or deleted." if chain["intact"]
            else f"Hash chain broken at entry {chain['first_bad_seq']}.")
        rows = []
        for e in reversed(led["entries"]):
            out = e.get("output", {})
            note = (out.get("summary") or out.get("error")
                    or (f"'{out.get('predicted')}' at {out.get('confidence', 0):.0%}" if e["tool"] == "predict" else "")
                    or (f"explanation {out.get('explanation_id')}" if e["tool"] == "explain" else ""))
            local = datetime.fromisoformat(e["time"]).astimezone()      # stored in UTC, shown in local time
            rows.append({"Entry": e["seq"], "Time (local)": local.strftime("%Y-%m-%d %H:%M:%S"), "Tool": e["tool"],
                         "Order": out.get("order_id"), "Result": note, "Hash": e["hash"][:10]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"File: {led['path']}")

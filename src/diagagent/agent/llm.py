"""LLM backends.

- "ollama":    local model through Ollama (default). Needs `ollama serve` and a
               tool-calling model, for example `ollama pull qwen2.5:7b`.
- "anthropic": optional API backend (pip install diagagent[anthropic], ANTHROPIC_API_KEY).
- "offline":   RuleBasedPlanner, a fixed script that is NOT an LLM. It calls the
               same tools in a sensible order so the graph, approval gate and demo
               can run in CI and without a GPU. Its final text is a template.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def get_llm(backend: str = "ollama", model: str | None = None, **kw: Any) -> BaseChatModel:
    backend = backend.lower()
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        # Ollama defaults to a 4096-token context. One agent run carries the system
        # prompt, all tool schemas and every tool result, so give it 8192.
        kw.setdefault("num_ctx", int(os.getenv("DIAGAGENT_NUM_CTX", "8192")))
        return ChatOllama(model=model or os.getenv("DIAGAGENT_MODEL", "qwen2.5:7b"),
                          temperature=0, **kw)
    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or os.getenv("DIAGAGENT_MODEL", "claude-sonnet-5-5"),
                             temperature=0, **kw)
    if backend == "offline":
        return RuleBasedPlanner()
    raise ValueError(f"unknown backend '{backend}'")


class RuleBasedPlanner(BaseChatModel):
    """Deterministic stand-in for an LLM. Reads the tool results so far and
    decides the next tool call. Handles two tasks: diagnose a device, or
    analyze a log file whose path appears in the user message."""

    @property
    def _llm_type(self) -> str:
        return "rule-based-planner"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "RuleBasedPlanner":
        return self

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _results(messages: list[BaseMessage]) -> list[tuple[str, Any]]:
        out = []
        for m in messages:
            if isinstance(m, ToolMessage):
                try:
                    payload = json.loads(m.content) if isinstance(m.content, str) else m.content
                except json.JSONDecodeError:
                    payload = {"raw": m.content}
                out.append((m.name or "", payload))
        return out

    @staticmethod
    def _call(n: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"name": name, "args": args, "id": f"call_{n}_{name}", "type": "tool_call"}

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        msg = self._decide(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # -- policy ------------------------------------------------------------- #
    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        user = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
        res = self._results(messages)
        n = len(res)
        names = [r[0] for r in res]

        log_path = next((w for w in str(user).split() if w.endswith((".csv", ".parquet", ".tsv"))), None)
        if log_path:
            if "load_log_file" not in names:
                return AIMessage("", tool_calls=[self._call(n, "load_log_file", {"path": log_path})])
            ds = res[names.index("load_log_file")][1]["dataset_id"]
            if "analyze_control_loop" not in names:
                return AIMessage("", tool_calls=[self._call(n, "analyze_control_loop", {"dataset_id": ds})])
            return AIMessage(self._final(res))

        if not res:
            return AIMessage("", tool_calls=[self._call(0, "list_devices", {})])
        if names == ["list_devices"]:
            dev = res[0][1][0]["device_id"]
            return AIMessage("", tool_calls=[
                self._call(n, "get_device_info", {"device_id": dev}),
                self._call(n + 1, "capture_step_test", {"device_id": dev}),
            ])
        dev = res[0][1][0]["device_id"]
        captures = [p["dataset_id"] for nm, p in res if nm == "capture_step_test"]
        analyses = [p for nm, p in res if nm == "analyze_control_loop"]
        last_name, last = res[-1]

        if last_name == "capture_step_test":
            return AIMessage("", tool_calls=[self._call(n, "analyze_control_loop", {
                "dataset_id": last["dataset_id"]})])
        if last_name == "analyze_control_loop" and len(analyses) == 1:
            if last["anomalies"]:
                return AIMessage("", tool_calls=[self._call(n, "suggest_pid_retune", {
                    "dataset_id": captures[0], "device_id": dev})])
            return AIMessage(self._final(res))
        if last_name == "suggest_pid_retune":
            prop = last.get("proposal")
            if not prop:
                return AIMessage(self._final(res))
            return AIMessage("", tool_calls=[self._call(n, "set_pid_gains", {
                "device_id": dev, "kp": prop["kp"], "ki": prop["ki"], "kd": prop["kd"],
                "reason": f"{prop['method']}: {prop['rationale']}"})])
        if last_name == "set_pid_gains":
            if last.get("status") == "applied":
                return AIMessage("", tool_calls=[self._call(n, "capture_step_test", {"device_id": dev})])
            return AIMessage(self._final(res))
        if last_name == "analyze_control_loop" and len(analyses) >= 2:
            return AIMessage("", tool_calls=[self._call(n, "compare_datasets", {
                "before_id": captures[0], "after_id": captures[-1]})])
        return AIMessage(self._final(res))

    @staticmethod
    def _final(res: list[tuple[str, Any]]) -> str:
        lines = ["[offline planner, template report]", ""]
        analyses = [p for nm, p in res if nm == "analyze_control_loop"]
        if analyses:
            a = analyses[0]
            lines.append("Anomalies found:" if a["anomalies"] else "No anomalies found.")
            for an in a["anomalies"]:
                lines.append(f"- {an['summary']}")
                for e in an["evidence"]:
                    lines.append(f"    evidence: {e}")
            if a["hypotheses"]:
                lines.append("Likely causes:")
                for h in a["hypotheses"]:
                    lines.append(f"- ({h['confidence']}) {h['cause']}. Fix: {h['suggested_fix']}")
        write = next((p for nm, p in res if nm == "set_pid_gains"), None)
        if write is not None:
            if write.get("status") == "applied":
                lines.append(f"Operator approved. Gains changed from {write['before']} to {write['after']}.")
            else:
                lines.append(f"Write not applied: {write.get('status', write)}. Device unchanged.")
        cmp_ = next((p for nm, p in res if nm == "compare_datasets"), None)
        if cmp_:
            b, a2 = cmp_["before"], cmp_["after"]
            lines.append("Before vs after:")
            for k in ("overshoot_pct", "settling_time_s", "oscillating", "n_anomalies"):
                lines.append(f"- {k}: {_fmt(b.get(k))} -> {_fmt(a2.get(k))}")
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "not settled"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

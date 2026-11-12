"""Bounded execution with trusted confirmation and resumable message history."""
from __future__ import annotations
import json
from .actions import parse_action
from .environment import OrderToolEnvironment


def run_agent(generate, user_prompt, *, environment=None, max_steps=4, messages=None):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    env = environment if environment is not None else OrderToolEnvironment()
    history = list(messages) if messages is not None else [{"role": "user", "content": user_prompt}]
    trace = []
    def finish(status):
        return {"status": status, "steps": len(trace), "trace": trace, "messages": history,
                "result": trace[-1].get("result") if trace else None,
                "session_id": getattr(env, "session_id", None),
                "pending_cancellation": getattr(env, "pending_cancellation", None)}
    for step in range(max_steps):
        output = generate(history, env.tool_definitions())
        action = parse_action(output)
        entry = {"step": step, "model_output": output, "action": action}
        trace.append(entry)
        if not action:
            history.extend([{"role": "assistant", "content": output},
                            {"role": "user", "content": "No valid tool call was parsed. Return a valid JSON tool call."}])
            continue
        result = env.execute(action["name"], action["arguments"])
        entry["result"] = result
        history.extend([
            {"role": "assistant", "tool_calls": [{"type": "function", "function": action}]},
            {"role": "tool", "name": action["name"], "content": json.dumps(result, ensure_ascii=False)},
        ])
        error = result.get("error")
        if error == "confirmation_required":
            return finish("awaiting_confirmation")
        if error in {"already_shipped", "not_cancellable"}:
            return finish("terminal_error")
        if result.get("ok") and result.get("done", True):
            return finish("completed")
        # Recoverable errors and empty searches consume the same bounded budget.
    return finish("max_steps")

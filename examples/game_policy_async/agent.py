# SPDX-License-Identifier: Apache-2.0
"""One candidate: short-context refinement and a raw final-policy score."""

import asyncio
import json
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .agentick_service.policy_worker import BUILTIN_NAMES, METHOD_NAMES
from .client import ChatClient, PolicyEvaluator


def extract_policy(output: str) -> str:
    """Accept plain Python or exactly one fenced Python code block."""
    output = output.strip()
    if "```" in output:
        match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", output, re.DOTALL)
        if match is None:
            raise ValueError("Return plain Python or one Python block, without prose")
        output = match[1].strip()
    if not output or len(output.encode()) > 32_768:
        raise ValueError("Empty or oversized policy")
    return output


Record = Callable[[dict], Awaitable[None]]
PROTOCOL = """Write a Python policy for the Agentick task described by the user.
Return a complete replacement policy: plain Python or exactly one Python fenced
block, with no prose in the final answer. If you use reasoning, put it in one
leading <think>...</think> block, then give only the policy or STOP after it.
Alternatively, output exactly STOP to retain the current
policy and finish. STOP is allowed only after a policy has been produced.
You have a limited number of LLM calls, including this call and any STOP call.
Only your final policy is scored on fresh game instances; previous feedback is
for development. A broken replacement does not revert to an older policy.

Define act(obs, memory) -> (integer_action, next_memory). act runs once per game
step, without LLM calls. memory starts as None for every game seed; next_memory
must be JSON serializable, at most 64 KiB. Observation is a JSON-compatible dict:
  grid: height, width, and terrain/objects/agents/metadata integer matrices [y][x].
        Unknown fogged cells have -1 in every layer; do not treat them as empty.
  legend: terrain/objects/agents dictionaries mapping official enum names to IDs.
  agent: position [x,y], orientation string, inventory list, energy, health.
  annotations: official semantic annotations (e.g. key_colors, door_states,
               task clues/meters). Position keys use the string "x,y".
  task_state: task-specific PUBLIC fields supplied by Agentick; keys vary by task.
  actions: action-name to integer mapping for this environment. Use these IDs.
  step, max_steps: completed game steps and episode limit.
x increases right; y increases down. Standard action names include noop, move_up,
move_down, move_left, move_right, interact. Do not assume unavailable actions exist.
Use grid layers, metadata, annotations, and public task_state for observations;
no environment objects or hidden task state are accessible.

Only function definitions at module level. No imports, classes, private names,
decorators, exceptions, I/O, or external access. Helpers, loops, comprehensions,
and basic containers are supported. Each act call has a 1-second deadline;
the policy process has 30 CPU seconds per game and 256 MiB memory on Linux.
Code is limited to 32 KiB. Available builtins: {builtins}.
Allowed methods: {methods}.
""".format(builtins=", ".join(BUILTIN_NAMES), methods=", ".join(sorted(METHOD_NAMES)))


def policy_answer(output: str) -> str:
    """Remove one leading Qwen3 reasoning block, not tags inside Python code.

    Already-separated Chat Completions content is passed through. This changes
    only the execution/feedback view: the recorded response and AReaL's original
    generation tokens remain untouched. Incomplete reasoning is not executable.
    """
    answer = output.strip()
    if not answer.startswith("<think>"):
        return answer
    reasoning, closed, answer = answer[len("<think>") :].partition("</think>")
    if not closed:
        raise ValueError("Unfinished <think> block; no final policy was produced")
    if "<think>" in reasoning or answer.lstrip().startswith(("<think>", "</think>")):
        raise ValueError(
            "Expected one leading <think> block followed by a final answer"
        )
    if not answer.strip():
        raise ValueError("Reasoning ended without a final policy or STOP")
    return answer.strip()


def build_messages(
    row: dict, previous: str | None, feedback: dict | None, remaining: int
) -> list[dict[str, str]]:
    # Rebuild a short state at every call. Never include scoring seeds/results,
    # older code, older conversations, or session credentials in messages.
    state: dict[str, Any] = {
        "task": row["task"],
        "difficulty": row["difficulty"],
        "instruction": row["instruction"],
        "remaining_calls_including_this": remaining,
    }
    if previous is not None:
        state.update(previous_policy=previous, previous_feedback=feedback)
    return [
        {"role": "system", "content": PROTOCOL},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
    ]


def feedback_view(evaluation: dict, max_bytes: int = 24_576) -> dict:
    """Keep all trial summaries and a bounded public example, never seed IDs."""
    trials = evaluation.get("trials", [])
    result = {
        "success_rate": evaluation["reward"],
        "trial_count": len(trials),
        "trials": [
            {k: t[k] for k in ("success", "steps", "error", "episode_return") if k in t}
            for t in trials
        ],
    }
    if "format_error" in evaluation:
        result["format_error"] = evaluation["format_error"]
    while len(json.dumps(result).encode()) > max_bytes and result["trials"]:
        result["trials"].pop()
        result["summaries_truncated"] = True
    if trials:
        example = next((t for t in trials if not t["success"]), trials[0])
        # Prefer the final observation for diagnosing failures, then initial/trace.
        result["example"] = {}
        for key in ("final", "initial", "trace"):
            if key in example:
                result["example"][key] = example[key]
                if len(json.dumps(result).encode()) > max_bytes:
                    del result["example"][key]
                    result["example_truncated"] = True
    return result


async def refine(
    client: ChatClient,
    row: dict,
    record: Record,
    max_rounds: int = 4,
    *,
    evaluate: Callable[..., Awaitable[dict]],
    **generation: Any,
) -> dict:
    """One session, serial calls, final policy only. No intermediate rewards."""
    if max_rounds < 1:
        raise ValueError("max-rounds must be positive")
    code = previous = None
    feedback = None
    error = "No policy was generated"
    interaction_ids: list[str] = []
    stop_reason = "budget"
    for index in range(max_rounds):
        interaction_id, output = await client.generate(
            build_messages(row, previous, feedback, max_rounds - index),
            **generation,
        )
        if interaction_id in interaction_ids:
            raise RuntimeError("Repeated completion ID within one refinement session")
        interaction_ids.append(interaction_id)
        await record(
            {
                "status": "generated",
                "round": index + 1,
                "interaction_id": interaction_id,
                "output": output,
            }
        )
        # A malformed replacement is the latest candidate, not a best-so-far fallback.
        # Do not echo an unfinished reasoning block as the previous policy.
        previous = ""
        try:
            answer = policy_answer(output)
            if answer == "STOP" and code is not None:
                stop_reason = "stop"
                break
            previous = answer.encode()[:32_768].decode(errors="ignore")
            if answer == "STOP":
                raise ValueError(
                    "STOP requires an existing policy; generate code first"
                )
            code = extract_policy(answer)
            error = ""
        except ValueError as exc:
            code = None
            error = str(exc)
        if index + 1 < max_rounds:
            evaluation = (
                await evaluate(
                    code, row["feedback_seeds"], row["task"], row["difficulty"]
                )
                if code is not None
                else {"reward": 0.0, "trials": [], "format_error": error}
            )
            feedback = feedback_view(evaluation)
            await record(
                {"status": "feedback", "round": index + 1, "evaluation": evaluation}
            )
    evaluation = (
        await evaluate(code, row["score_seeds"], row["task"], row["difficulty"])
        if code is not None
        else {"reward": 0.0, "trials": [], "format_error": error}
    )
    result = {
        "policy": code,
        "evaluation": evaluation,
        "interaction_ids": interaction_ids,
        "calls": len(interaction_ids),
        "stop_reason": stop_reason,
    }
    await record({"status": "scored", **result})
    return result


def append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


class AgentickAgent:
    """AReaL inline agent: own all LLM calls, refinement and terminal scoring."""

    def __init__(
        self,
        service_url: str,
        max_rounds: int = 4,
        timeout: float = 7500,
        evaluation_timeout: float = 600,
        log_dir: str | None = None,
        **generation,
    ):
        self.service_url = service_url
        self.max_rounds = max_rounds
        self.timeout = timeout
        self.evaluation_timeout = evaluation_timeout
        self.log_dir = Path(log_dir) if log_dir else None
        self.generation = generation

    async def run(self, data: dict, **extra_kwargs) -> float:
        http = extra_kwargs["http_client"]
        # Session credentials are used only for inference inside AReaL. The game
        # service receives four execution fields and its own service auth header.
        client = ChatClient(http, extra_kwargs["base_url"], extra_kwargs["api_key"])
        evaluator = PolicyEvaluator(
            http,
            self.service_url,
            os.environ["AGENTICK_SERVICE_KEY"],
            self.evaluation_timeout,
        )
        path = None
        if self.log_dir is not None:
            await asyncio.to_thread(self.log_dir.mkdir, parents=True, exist_ok=True)
            path = self.log_dir / f"{uuid.uuid4().hex}.jsonl"

        async def record(event):
            if path is not None:
                await asyncio.to_thread(append_event, path, event)

        try:
            async with asyncio.timeout(self.timeout):
                await record(
                    {
                        "status": "started",
                        "session_id": extra_kwargs["session_id"],
                        "row": data,
                    }
                )
                result = await refine(
                    client,
                    data,
                    record,
                    self.max_rounds,
                    evaluate=evaluator.evaluate,
                    **self.generation,
                )
                return float(result["evaluation"]["reward"])
        except Exception as exc:
            await record({"status": "failed", "error_type": type(exc).__name__})
            raise

    def classify_proxy_failure(self, error: Exception, **kwargs) -> str:
        return "system_failure_reject"

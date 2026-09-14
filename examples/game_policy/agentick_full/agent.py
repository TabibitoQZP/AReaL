# SPDX-License-Identifier: Apache-2.0
"""Standalone multi-round code refinement, grouped training, and frozen evaluation."""

import argparse
import asyncio
import json
import logging
import math
import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from evaluator import evaluate_policy, extract_policy
from policy_worker import BUILTIN_NAMES, METHOD_NAMES
from rl_client import FrozenClient, RLClient, Session, read_sessions
from tasks import AGENTICK_REVISION, load_rows

logger = logging.getLogger("GamePolicy")
Record = Callable[[dict], Awaitable[None]]
PROTOCOL = """Write a Python policy for the Agentick task described by the user.
Return a complete replacement policy: plain Python or exactly one Python fenced
block, with no prose. Alternatively, output exactly STOP to retain the current
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


def normalize_rewards(rewards: list[float]) -> list[float]:
    if len(rewards) < 2 or any(
        not math.isfinite(r) or not 0 <= r <= 1 for r in rewards
    ):
        raise ValueError("Need at least two finite success rewards in [0, 1]")
    if min(rewards) == max(rewards):
        return [0.0] * len(rewards)
    mean = math.fsum(rewards) / len(rewards)
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + 1e-8) for r in rewards]


async def refine(
    client: RLClient | FrozenClient,
    session: Session | None,
    row: dict,
    record: Record,
    max_rounds: int = 4,
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
            session,
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
        if output.strip() == "STOP" and code is not None:
            stop_reason = "stop"
            break
        # A malformed replacement is the latest candidate, not a best-so-far fallback.
        previous = output.encode()[:32_768].decode(errors="ignore")
        try:
            if output.strip() == "STOP":
                raise ValueError(
                    "STOP requires an existing policy; generate code first"
                )
            code = extract_policy(output)
            error = ""
        except ValueError as exc:
            code = None
            error = str(exc)
        if index + 1 < max_rounds:
            evaluation = (
                await evaluate_policy(
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
        await evaluate_policy(code, row["score_seeds"], row["task"], row["difficulty"])
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


async def run_group(
    client: RLClient,
    sessions: list[Session],
    row: dict,
    record: Record,
    max_rounds: int = 4,
    **generation: Any,
) -> list[dict]:
    if (
        len(sessions) < 2
        or len({s.session_id for s in sessions}) != len(sessions)
        or len({s.session_api_key for s in sessions}) != len(sessions)
    ):
        raise ValueError("Each group requires at least two unique sessions")

    async def run(index: int, session: Session) -> dict:
        async def candidate_record(event: dict) -> None:
            await record(
                {"candidate": index, "session_id": session.session_id, **event}
            )

        return await refine(
            client, session, row, candidate_record, max_rounds, **generation
        )

    pending = [asyncio.create_task(run(i, s)) for i, s in enumerate(sessions)]
    try:
        results = await asyncio.gather(*pending)
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    advantages = normalize_rewards([r["evaluation"]["reward"] for r in results])
    for session, result, advantage in zip(sessions, results, advantages, strict=True):
        result.update(session_id=session.session_id, normalized_reward=advantage)
    # Record every raw score and normalized reward before any request can finalize
    # a session. One terminal reward propagates to all calls with turn_discount=1.
    await record({"status": "normalized", "candidates": results})
    for result, session in zip(results, sessions, strict=True):
        acknowledgement = await client.set_reward(
            session, result["interaction_ids"][-1], result["normalized_reward"]
        )
        await record(
            {
                "status": "rewarded",
                "session_id": session.session_id,
                "interaction_id": result["interaction_ids"][-1],
                "trajectory_id": acknowledgement.get("trajectory_id"),
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    """Macro-average tasks/levels so the two held-out planning tasks don't dominate."""
    cells: dict[tuple, list[float]] = defaultdict(list)
    calls = []
    for result in results:
        cells[result["category"], result["task"], result["difficulty"]].append(
            result["reward"]
        )
        calls.append(result["calls"])
    cell_means = {key: math.fsum(values) / len(values) for key, values in cells.items()}
    categories: dict[str, list[float]] = defaultdict(list)
    levels: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (category, _, difficulty), value in cell_means.items():
        categories[category].append(value)
        levels[difficulty][category].append(value)

    def mean(values: list[float]) -> float:
        return math.fsum(values) / len(values)

    by_category = {key: mean(values) for key, values in categories.items()}
    return {
        "candidates": len(results),
        "mean_calls": mean(calls),
        "category_macro_success": mean(list(by_category.values())),
        "by_category": by_category,
        "by_difficulty": {
            level: mean([mean(v) for v in cats.values()])
            for level, cats in levels.items()
        },
        "by_task_difficulty": [
            {"category": c, "task": t, "difficulty": d, "success_rate": r}
            for (c, t, d), r in cell_means.items()
        ],
    }


async def drive(args: argparse.Namespace) -> None:
    if (
        args.max_rounds <= 0
        or args.max_tokens <= 0
        or not math.isfinite(args.temperature)
        or args.temperature < 0
    ):
        raise ValueError("Use positive budgets and finite nonnegative temperature")
    training = args.mode == "train"
    rows = await asyncio.to_thread(
        load_rows, args.data, "train" if training else "test"
    )
    if (
        args.offset < 0
        or args.offset >= len(rows)
        or (args.limit is not None and args.limit <= 0)
    ):
        raise ValueError("Use an in-range offset and a positive limit")
    if args.limit is not None and args.offset + args.limit > len(rows):
        raise ValueError("Requested row range exceeds the dataset")
    rows = rows[args.offset : None if args.limit is None else args.offset + args.limit]
    sessions = None
    admin_key = None
    if training:
        if args.group_size < 2 or args.epochs <= 0:
            raise ValueError("Require group-size >= 2 and positive epochs")
        if args.sessions:
            sessions = await asyncio.to_thread(read_sessions, args.sessions)
            if len(sessions) < len(rows) * args.epochs * args.group_size:
                raise ValueError("Not enough unique pre-issued sessions for this run")
        else:
            admin_key = os.environ.get(args.admin_key_env)
            if not admin_key:
                raise ValueError("Set the chosen admin-key-env or supply sessions")
    elif args.samples <= 0:
        raise ValueError("samples must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        lock = asyncio.Lock()

        async def record(event: dict) -> None:
            async with lock:
                await asyncio.to_thread(
                    stream.write, json.dumps(event, ensure_ascii=False) + "\n"
                )
                await asyncio.to_thread(stream.flush)

        await record(
            {
                "status": "run",
                "mode": args.mode,
                "revision": AGENTICK_REVISION,
                "max_rounds": args.max_rounds,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "model": args.model,
                "rows": len(rows),
                "offset": args.offset,
                "group_size": args.group_size if training else None,
                "epochs": args.epochs if training else None,
                "samples": None if training else args.samples,
            }
        )
        async with httpx.AsyncClient(timeout=300, trust_env=False) as http:
            client = (
                RLClient(args.gateway, http)
                if training
                else FrozenClient(
                    args.base_url, os.environ.get(args.api_key_env, ""), http
                )
            )
            group_index = 0
            summaries = []
            try:
                for epoch in range(args.epochs if training else 1):
                    for row in rows:
                        context = {
                            k: row[k] for k in ("id", "task", "category", "difficulty")
                        }
                        context.update(group=group_index, epoch=epoch)

                        async def row_record(event: dict) -> None:
                            await record({**context, **event})

                        await row_record({"status": "sample", "data": row})
                        generation = {
                            "model": args.model,
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                        }
                        if training:
                            if sessions is None:
                                group = [
                                    await client.start_session(
                                        f"{row['id']}-{epoch}-{i}", admin_key
                                    )
                                    for i in range(args.group_size)
                                ]
                            else:
                                offset = group_index * args.group_size
                                group = sessions[offset : offset + args.group_size]
                            await row_record(
                                {
                                    "status": "started",
                                    "session_ids": [s.session_id for s in group],
                                }
                            )
                            results = await run_group(
                                client,
                                group,
                                row,
                                row_record,
                                args.max_rounds,
                                **generation,
                            )
                        else:
                            results = []
                            for index in range(args.samples):

                                async def candidate_record(event: dict) -> None:
                                    await row_record({"candidate": index, **event})

                                results.append(
                                    await refine(
                                        client,
                                        None,
                                        row,
                                        candidate_record,
                                        args.max_rounds,
                                        **generation,
                                    )
                                )
                        for result in results:
                            summaries.append(
                                {
                                    **context,
                                    "reward": result["evaluation"]["reward"],
                                    "calls": result["calls"],
                                }
                            )
                        logger.info(
                            "%s %s: final scores=%s",
                            row["task"],
                            row["difficulty"],
                            [r["evaluation"]["reward"] for r in results],
                        )
                        group_index += 1
            except BaseException as exc:
                await record(
                    {
                        "status": "incomplete",
                        "group": group_index,
                        "error_type": type(exc).__name__,
                    }
                )
                raise
            summary = summarize(summaries)
            await record({"status": "summary", **summary})
            logger.info(
                "Completed %d candidates; macro success=%.4f",
                summary["candidates"],
                summary["category_macro_success"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("train", "eval"):
        sub = modes.add_parser(mode)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument(
            "--offset", type=int, default=0, help="First dataset row in this run"
        )
        sub.add_argument(
            "--limit",
            type=int,
            help="Number of rows; use disjoint chunks for pre-issued sessions",
        )
        sub.add_argument("--max-rounds", type=int, default=4)
        sub.add_argument("--max-tokens", type=int, default=4096)
        sub.add_argument("--temperature", type=float, default=1.0)
        sub.add_argument("--model", default="default")
        if mode == "train":
            sub.add_argument(
                "--gateway", required=True, help="Training gateway origin, without /v1"
            )
            sub.add_argument("--group-size", type=int, default=4)
            sub.add_argument("--epochs", type=int, default=1)
            credentials = sub.add_mutually_exclusive_group(required=True)
            credentials.add_argument("--sessions", type=Path)
            credentials.add_argument("--admin-key-env")
        else:
            sub.add_argument(
                "--base-url", required=True, help="Frozen model API base, including /v1"
            )
            sub.add_argument("--api-key-env", default="OPENAI_API_KEY")
            sub.add_argument(
                "--samples",
                type=int,
                default=1,
                help="Independent candidates per row, averaged; never best-of-N",
            )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(drive(parser.parse_args()))


if __name__ == "__main__":
    main()

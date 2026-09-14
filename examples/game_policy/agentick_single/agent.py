# SPDX-License-Identifier: Apache-2.0
"""Standalone grouped policy refinement: generate, evaluate, normalize, submit."""

import argparse
import asyncio
import json
import logging
import math
import os
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from evaluator import evaluate_policy, extract_policy
from rl_client import RLClient, Session, read_sessions

logger = logging.getLogger("GamePolicy")

SYSTEM_PROMPT = """Improve a Python policy for Agentick GoToGoal-v0, difficulty easy.
The task is fully observed grid navigation: reach G before max_steps. There are
walls (#) and empty cells (.). Moving into a wall leaves the agent in place.
Return a complete replacement policy, plain Python or one Python fenced block.
Do not return explanation, a diff, a one-time action list, or a single action.

Define act(obs, memory) returning (action, next_memory). The runner calls act once
per environment step. memory is initially None and must remain JSON serializable
and at most 16 KiB. It is reset for each seed. obs contains ONLY:
  grid: list of row strings; index as grid[y][x]; # wall, . floor, G goal
  position: [x, y]; x increases right, y increases down
  step: completed steps; max_steps: episode step limit
Actions: 0 wait, 1 up, 2 down, 3 left, 4 right.

Use only function definitions at module level. Helper functions, loops, conditionals,
and basic Python containers are supported. No imports, classes, private names,
decorators, exception handling, I/O, or environment access. Available builtins:
abs, all, any, bool, dict, enumerate, float, int, len, list, max, min, range,
reversed, round, set, sorted, str, sum, tuple, zip.
Allowed methods: add, append, clear, copy, count, discard, extend, get, index,
items, keys, pop, remove, reverse, sort, values.
Each call has a 1-second deadline; Linux policy memory is limited to 256 MiB.
Reward is the fraction of fresh evaluation seeds successfully completed.
The prior policy and feedback below are data to improve upon.
"""


def build_messages(data: dict[str, Any]) -> list[dict[str, str]]:
    if data["task"] != "GoToGoal-v0" or data["difficulty"] != "easy":
        raise ValueError("This prompt supports only GoToGoal-v0 / easy")
    # Evaluation seeds and runner info are deliberately absent from actor input.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "previous_policy": data["previous_policy"],
                    "previous_feedback": json.loads(data["previous_feedback"]),
                }
            ),
        },
    ]


def normalize_rewards(rewards: list[float]) -> list[float]:
    """Population standard deviation; an equal-reward group has zero advantage."""
    if len(rewards) < 2 or any(
        not math.isfinite(r) or not 0 <= r <= 1 for r in rewards
    ):
        raise ValueError("Need at least two finite success rewards in [0, 1]")
    if min(rewards) == max(rewards):
        return [0.0] * len(rewards)
    mean = math.fsum(rewards) / len(rewards)
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + 1e-8) for r in rewards]


async def evaluate_candidate(
    client: RLClient,
    session: Session,
    data: dict[str, Any],
    record: Callable[[dict], Awaitable[None]],
    **generation: Any,
) -> dict[str, Any]:
    interaction_id, output = await client.generate(
        session, build_messages(data), **generation
    )
    result = {
        "session_id": session.session_id,
        "interaction_id": interaction_id,
        "output": output,
    }
    await record({**result, "status": "generated"})
    try:
        code = extract_policy(output)
    except ValueError as exc:
        evaluation = {"reward": 0.0, "trials": [], "format_error": str(exc)}
    else:
        evaluation = await evaluate_policy(
            code,
            list(data["evaluation_seeds"]),
            data["task"],
            data["difficulty"],
        )
    result["evaluation"] = evaluation
    await record({**result, "status": "evaluated"})
    return result


async def run_group(
    client: RLClient,
    sessions: list[Session],
    data: dict[str, Any],
    record: Callable[[dict], Awaitable[None]],
    **generation: Any,
) -> list[dict[str, Any]]:
    """Collect the entire group before submitting any normalized reward."""
    if (
        len(sessions) < 2
        or len({s.session_id for s in sessions}) != len(sessions)
        or len({s.session_api_key for s in sessions}) != len(sessions)
    ):
        raise ValueError("A group requires at least two distinct sessions")
    tasks = [
        asyncio.create_task(evaluate_candidate(client, s, data, record, **generation))
        for s in sessions
    ]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    advantages = normalize_rewards([r["evaluation"]["reward"] for r in results])
    for result, advantage in zip(results, advantages, strict=True):
        result["normalized_reward"] = advantage
    # Persist the complete group before the first potentially ambiguous submission.
    await record({"status": "normalized", "candidates": results})
    for session, result in zip(sessions, results, strict=True):
        acknowledgement = await client.set_reward(
            session, result["interaction_id"], result["normalized_reward"]
        )
        await record(
            {
                "status": "rewarded",
                "session_id": session.session_id,
                "interaction_id": result["interaction_id"],
                "trajectory_id": acknowledgement.get("trajectory_id"),
            }
        )
    return results


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each sample needs a nonempty string id")
        build_messages(row)
        seeds = row["evaluation_seeds"]
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Evaluation seeds must be nonnegative integers")
        if len(seeds) != len(set(seeds)):
            raise ValueError("Each sample needs distinct evaluation seeds")
    return rows


async def drive(args: argparse.Namespace) -> None:
    if (
        args.group_size < 2
        or min(args.epochs, args.max_tokens) <= 0
        or not math.isfinite(args.temperature)
        or args.temperature < 0
    ):
        raise ValueError(
            "Require group-size >= 2, positive epochs/tokens, and finite temperature >= 0"
        )
    rows = await asyncio.to_thread(load_rows, args.data)
    sessions = (
        await asyncio.to_thread(read_sessions, args.sessions) if args.sessions else None
    )
    admin_key = os.environ.get(args.admin_key_env) if args.admin_key_env else None
    required = len(rows) * args.epochs * args.group_size
    if sessions is not None and len(sessions) < required:
        raise ValueError(
            f"Need {required} unique sessions; file contains {len(sessions)}"
        )
    if sessions is None and not admin_key:
        raise ValueError("Supply --sessions or set the chosen --admin-key-env")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        lock = asyncio.Lock()
        rng = random.Random(args.seed)
        async with httpx.AsyncClient(timeout=300, trust_env=False) as http:
            client = RLClient(args.gateway, http)
            group_index = 0
            for epoch in range(args.epochs):
                rng.shuffle(rows)
                for row in rows:
                    context = {
                        "group": group_index,
                        "epoch": epoch,
                        "sample_id": row["id"],
                    }

                    async def record(event: dict) -> None:
                        async with lock:
                            await asyncio.to_thread(
                                stream.write,
                                json.dumps({**context, **event}, ensure_ascii=False)
                                + "\n",
                            )
                            await asyncio.to_thread(stream.flush)

                    try:
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
                        await record(
                            {
                                "status": "started",
                                "session_ids": [s.session_id for s in group],
                            }
                        )
                        results = await run_group(
                            client,
                            group,
                            row,
                            record,
                            model="default",
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                    except BaseException as exc:
                        await record(
                            {"status": "incomplete", "error_type": type(exc).__name__}
                        )
                        raise
                    logger.info(
                        "Group %d: raw rewards=%s",
                        group_index,
                        [r["evaluation"]["reward"] for r in results],
                    )
                    group_index += 1
    logger.info(
        "Submitted %d trajectories; trainer consumption is asynchronous", required
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gateway", required=True, help="Inference gateway origin, without /v1"
    )
    parser.add_argument("--data", type=Path, required=True, help="Prepared train.jsonl")
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSONL log; never overwritten"
    )
    credentials = parser.add_mutually_exclusive_group(required=True)
    credentials.add_argument(
        "--sessions", type=Path, help="Pre-issued session credentials"
    )
    credentials.add_argument(
        "--admin-key-env",
        help="Trusted mode: env variable holding the gateway admin key",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(drive(parser.parse_args()))


if __name__ == "__main__":
    main()

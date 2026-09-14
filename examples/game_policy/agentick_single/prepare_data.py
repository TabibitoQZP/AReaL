# SPDX-License-Identifier: Apache-2.0
"""Build fixed previous-policy/feedback examples using real Agentick runs."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from evaluator import evaluate_policy

# Deliberately incomplete policies. Their feedback is measured, never fabricated.
INITIAL_POLICIES = (
    "def act(obs, memory):\n    return 0, None\n",
    "def act(obs, memory):\n    return 4, None\n",
    "def act(obs, memory):\n    return 2, None\n",
    "def act(obs, memory):\n    return [1, 4, 2, 3][obs['step'] % 4], None\n",
)

# Sanity check of the adapter for the obstacle-free easy task, never an actor label.
SMOKE_POLICY = """def act(obs, memory):
    x, y = obs['position']
    for gy, row in enumerate(obs['grid']):
        for gx, cell in enumerate(row):
            if cell == 'G':
                if x < gx:
                    return 4, None
                if x > gx:
                    return 3, None
                if y < gy:
                    return 2, None
                if y > gy:
                    return 1, None
    return 0, None
"""


def seed_groups(
    row_index: int,
    seeds_per_policy: int,
    start_seed: int,
) -> tuple[list[int], list[int]]:
    start = start_seed + row_index * 2 * seeds_per_policy
    return (
        list(range(start, start + seeds_per_policy)),
        list(range(start + seeds_per_policy, start + 2 * seeds_per_policy)),
    )


async def build_rows(
    count: int,
    offset: int,
    seeds_per_policy: int,
    start_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(offset, offset + count):
        feedback_seeds, evaluation_seeds = seed_groups(
            index, seeds_per_policy, start_seed
        )
        previous_policy = INITIAL_POLICIES[index % len(INITIAL_POLICIES)]
        feedback = await evaluate_policy(previous_policy, feedback_seeds)
        rows.append(
            {
                "id": f"goto-goal-easy-{index}",
                "task": "GoToGoal-v0",
                "difficulty": "easy",
                "previous_policy": previous_policy,
                # Keep feedback serialized, matching the actor prompt input.
                "previous_feedback": json.dumps(feedback),
                "feedback_seeds": feedback_seeds,
                "evaluation_seeds": evaluation_seeds,
            }
        )
    return rows


async def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    if min(args.train_size, args.valid_size, args.seeds_per_policy) <= 0:
        raise ValueError("Split sizes and seeds-per-policy must be positive")
    last_seed = (
        args.start_seed
        + 2 * (args.train_size + args.valid_size) * args.seeds_per_policy
    )
    if args.start_seed < 0 or last_seed + args.seeds_per_policy > 2**31:
        raise ValueError("Dataset seeds must fit in [0, 2**31)")

    smoke_seeds = list(range(last_seed, last_seed + args.seeds_per_policy))
    smoke = await evaluate_policy(SMOKE_POLICY, smoke_seeds)
    if smoke["reward"] != 1.0:
        raise RuntimeError(f"Navigation sanity check failed: {smoke}")
    train = await build_rows(args.train_size, 0, args.seeds_per_policy, args.start_seed)
    valid = await build_rows(
        args.valid_size,
        args.train_size,
        args.seeds_per_policy,
        args.start_seed,
    )
    output.mkdir(parents=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (output / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "task": "GoToGoal-v0",
        "difficulty": "easy",
        "train_size": len(train),
        "valid_size": len(valid),
        "seeds_per_policy": args.seeds_per_policy,
        "start_seed": args.start_seed,
        "smoke_check": smoke,
    }
    (output / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/goto_goal")
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--valid-size", type=int, default=8)
    parser.add_argument("--seeds-per-policy", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=10_000)
    asyncio.run(prepare(parser.parse_args()))


if __name__ == "__main__":
    main()

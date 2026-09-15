# SPDX-License-Identifier: Apache-2.0
"""Snapshot official instructions and generate reproducible train/test JSONL."""

import argparse
import json
from pathlib import Path

from .tasks import (
    AGENTICK_REVISION,
    DIFFICULTIES,
    TASK_SPLIT,
    TASKS,
    official_instructions,
    training_schedule,
)


def prepare(
    output: Path,
    train_groups: int = 384,
    test_repeats: int = 4,
    feedback_count: int = 4,
    score_count: int = 8,
    seed: int = 1,
) -> None:
    if seed < 0 or test_repeats <= 0 or not 1 <= min(feedback_count, score_count):
        raise ValueError("Use nonnegative seed and positive counts")
    if max(feedback_count, score_count) > 512:
        raise ValueError("At most 512 seeds per role")
    train = training_schedule(train_groups, seed)
    test = [
        (task, level)
        for task, (_, split) in TASKS.items()
        if split == "test"
        for level in DIFFICULTIES
        for _ in range(test_repeats)
    ]
    if max(len(train), len(test)) >= 10**9:
        raise ValueError("Too many rows for the seed allocation")
    instructions = official_instructions()
    # Exclusive directory creation avoids mixing manifests from different runs.
    output.mkdir(parents=True, exist_ok=False)
    for split_index, (split, schedule) in enumerate((("train", train), ("test", test))):
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for index, (task, difficulty) in enumerate(schedule):
                base = ((seed * 2 + split_index) * 10**9 + index) * 1024
                row = {
                    "id": f"{split}-{index:06d}",
                    "split": split,
                    "revision": AGENTICK_REVISION,
                    "task": task,
                    "category": TASKS[task][0],
                    "difficulty": difficulty,
                    "instruction": instructions[task, difficulty],
                    "feedback_seeds": list(range(base, base + feedback_count)),
                    "score_seeds": list(range(base + 512, base + 512 + score_count)),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "revision": AGENTICK_REVISION,
        "split": TASK_SPLIT,
        "seed": seed,
        "train_groups": len(train),
        "test_rows": len(test),
        "feedback_count": feedback_count,
        "score_count": score_count,
        "sampling": "equal category/difficulty; shuffled task cycles within each",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-groups", type=int, default=384)
    parser.add_argument("--test-repeats", type=int, default=4)
    parser.add_argument("--feedback-count", type=int, default=4)
    parser.add_argument("--score-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Execute every task/difficulty with a no-op policy; no LLM or training required."""

import argparse
import asyncio
import json
from pathlib import Path

from evaluator import evaluate_policy
from tasks import DIFFICULTIES, TASKS, official_instructions

POLICY = 'def act(obs, memory):\n    return obs["actions"]["noop"], memory\n'


async def smoke(output: Path, seed: int = 0) -> None:
    official_instructions()
    output.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(4)
    failures = []
    with output.open("x", encoding="utf-8") as stream:

        async def check(task: str, difficulty: str) -> None:
            async with semaphore:
                try:
                    evaluation = await evaluate_policy(POLICY, [seed], task, difficulty)
                    trial = evaluation["trials"][0]
                    if trial["error"]:
                        raise RuntimeError(trial["error"])
                    result = {
                        "task": task,
                        "difficulty": difficulty,
                        "status": "passed",
                        "steps": trial["steps"],
                        "success": trial["success"],
                    }
                except Exception as exc:
                    result = {
                        "task": task,
                        "difficulty": difficulty,
                        "status": "failed",
                        "error": str(exc),
                    }
                    failures.append(result)
                stream.write(json.dumps(result) + "\n")
                stream.flush()

        await asyncio.gather(*(check(t, d) for t in TASKS for d in DIFFICULTIES))
    if failures:
        raise RuntimeError(f"{len(failures)} combinations failed; see {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(smoke(args.output, args.seed))


if __name__ == "__main__":
    main()

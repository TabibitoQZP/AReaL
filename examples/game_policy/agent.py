# SPDX-License-Identifier: Apache-2.0
"""One actor completion: previous policy + feedback -> replacement policy."""

import json
from typing import Any

from .evaluator import evaluate_policy, extract_policy

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


class AgentickPolicyAgent:
    def __init__(self, **generation_kwargs: Any):
        self.generation_kwargs = generation_kwargs

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        from openai import AsyncOpenAI

        # Require AReaL's session client; never fall back to a hosted model.
        client = AsyncOpenAI(
            base_url=extra_kwargs["base_url"],
            api_key=extra_kwargs["api_key"],
            http_client=extra_kwargs["http_client"],
            max_retries=0,
        )
        completion = await client.chat.completions.create(
            model="default",
            messages=build_messages(data),
            **self.generation_kwargs,
        )
        try:
            code = extract_policy(completion.choices[0].message.content or "")
        except ValueError:
            return 0.0
        result = await evaluate_policy(
            code,
            list(data["evaluation_seeds"]),
            data["task"],
            data["difficulty"],
        )
        return float(result["reward"])

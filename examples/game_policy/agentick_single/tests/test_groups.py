# SPDX-License-Identifier: Apache-2.0
"""Group advantage calculation and all-candidates-before-reward behavior."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import agent
import pytest
from rl_client import Session


def test_normalization_centers_and_scales_population_rewards() -> None:
    advantages = agent.normalize_rewards([0, 0.25, 0.75, 1])
    assert sum(advantages) == pytest.approx(0)
    assert sum(a * a for a in advantages) / 4 == pytest.approx(1)
    assert advantages[0] < -1 and advantages[-1] > 1


@pytest.mark.parametrize("reward", [0.0, 0.1, 1.0])
def test_equal_rewards_produce_exact_zero_advantages(reward) -> None:
    assert agent.normalize_rewards([reward] * 4) == [0.0] * 4


@pytest.mark.parametrize(
    "rewards", [[], [1], [0, float("nan")], [0, float("inf")], [0, -0.1], [0, 1.1]]
)
def test_invalid_raw_scores_are_not_normalized(rewards) -> None:
    with pytest.raises(ValueError):
        agent.normalize_rewards(rewards)


def test_failed_candidate_blocks_entire_group_and_cancels_peers(monkeypatch) -> None:
    cancelled = []
    client = SimpleNamespace(
        generate=AsyncMock(side_effect=[("c0", "bad"), ("c1", "waiting")]),
        set_reward=AsyncMock(),
    )

    async def evaluate(code, *args):
        if code == "bad":
            await asyncio.sleep(0.01)
            raise RuntimeError("environment unavailable")
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    data = {
        "id": "task",
        "task": "GoToGoal-v0",
        "difficulty": "easy",
        "previous_policy": "old",
        "previous_feedback": "{}",
        "evaluation_seeds": [1],
    }
    with pytest.raises(RuntimeError, match="environment unavailable"):
        asyncio.run(
            agent.run_group(
                client, [Session("s0", "k0"), Session("s1", "k1")], data, AsyncMock()
            )
        )
    client.set_reward.assert_not_awaited()
    assert cancelled == [True]

# SPDX-License-Identifier: Apache-2.0
"""Real CPU integration using the pinned optional Agentick dependency."""

import asyncio
import json

import pytest

from examples.game_policy_async.agentick_service.env_runner import policy_observation
from examples.game_policy_async.agentick_service.evaluator import run_policy
from examples.game_policy_async.tasks import official_instructions

agentick = pytest.importorskip("agentick")


def test_all_official_descriptions_have_four_difficulties():
    instructions = official_instructions()
    assert len(instructions) == 148
    from agentick.tasks.descriptions import get_task_description_structured

    desc = get_task_description_structured("SequenceMemory-v0")
    assert desc.summary in instructions["SequenceMemory-v0", "hard"]
    assert desc.actions in instructions["SequenceMemory-v0", "hard"]


@pytest.mark.parametrize(
    "task,hidden",
    [
        ("FogOfWarExploration-v0", ("goal_positions",)),
        ("SequenceMemory-v0", ("sequence", "distractors", "goal_positions")),
        ("FewShotAdaptation-v0", ("true_goal", "rule_name", "target_type", "trials")),
    ],
)
def test_real_observation_excludes_hidden_task_fields(task, hidden):
    env = agentick.make(task, difficulty="easy", render_mode="state_dict")
    try:
        obs, _ = env.reset(seed=0)
        public = policy_observation(obs, 0, env.max_steps, {}, {})
        assert not set(hidden) & public["task_state"].keys()
        assert "entities" not in public
        json.dumps(public, allow_nan=False)
        if task.startswith("FogOfWar"):
            grid = public["grid"]
            fog = [
                (x, y)
                for y, r in enumerate(grid["metadata"])
                for x, value in enumerate(r)
                if value == -1
            ]
            assert fog
            for x, y in fog:
                assert all(
                    grid[k][y][x] == -1 for k in ("terrain", "objects", "agents")
                )
    finally:
        env.close()


SOLVER = """def act(obs, memory):
    x, y = obs["agent"]["position"]
    grid = obs["grid"]["objects"]
    for gy, row in enumerate(grid):
        for gx, value in enumerate(row):
            if value == obs["legend"]["objects"]["GOAL"]:
                if gx > x:
                    return obs["actions"]["move_right"], None
                if gx < x:
                    return obs["actions"]["move_left"], None
                if gy > y:
                    return obs["actions"]["move_down"], None
                return obs["actions"]["move_up"], None
    return obs["actions"]["noop"], None
"""


def test_real_policy_observation_supports_goal_solver():
    for seed in (0, 1):
        result = asyncio.run(run_policy(SOLVER, seed, "GoToGoal-v0", "easy"))
        assert result["success"] is True


@pytest.mark.parametrize(
    "code",
    [
        "def act(obs, memory):\n    return 100, None",
        "def act(obs, memory):\n    while True:\n        pass",
        "not valid python",
    ],
)
def test_real_policy_error_becomes_feedback(code):
    result = asyncio.run(run_policy(code, 0, "GoToGoal-v0", "easy"))
    assert result["success"] is False and result["error"]

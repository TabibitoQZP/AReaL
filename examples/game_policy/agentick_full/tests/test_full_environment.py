# SPDX-License-Identifier: Apache-2.0
"""Real CPU integration; skipped when optional Agentick is not installed."""

import asyncio
import json
from collections import Counter

import httpx
import pytest

pytest.importorskip("agentick")

import agentick
from agent import run_group
from env_runner import policy_observation
from evaluator import evaluate_policy
from rl_client import RLClient, Session
from tasks import official_instructions


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
    result = asyncio.run(evaluate_policy(SOLVER, [0, 1], "GoToGoal-v0", "easy"))
    assert result["reward"] == 1.0


@pytest.mark.parametrize(
    "code",
    [
        "def act(obs, memory):\n    return 100, None",
        "def act(obs, memory):\n    while True:\n        pass",
        "not valid python",
    ],
)
def test_real_policy_error_becomes_feedback(code):
    result = asyncio.run(evaluate_policy(code, [0], "GoToGoal-v0", "easy"))
    assert result["reward"] == 0 and result["trials"][0]["error"]


def test_real_game_feedback_to_refinement_to_terminal_http_reward():
    calls = Counter()
    rewards = []
    bad_policy = "def act(obs, memory):\n    return 100, None"
    sample = {
        "task": "GoToGoal-v0",
        "difficulty": "easy",
        "instruction": official_instructions()["GoToGoal-v0", "easy"],
        "feedback_seeds": [0],
        "score_seeds": [1],
    }

    def respond(request):
        key = request.headers["Authorization"].removeprefix("Bearer ")
        body = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            calls[key] += 1
            if calls[key] == 1:
                output = bad_policy
            else:
                state = json.loads(body["messages"][-1]["content"])
                assert state["previous_feedback"]["success_rate"] == 0
                assert state["previous_feedback"]["trials"][0]["error"]
                output = SOLVER if key == "s0" else bad_policy
            return httpx.Response(
                200,
                json={
                    "id": f"{key}-{calls[key]}",
                    "choices": [{"message": {"content": output}}],
                },
            )
        assert request.url.path == "/rl/set_reward"
        rewards.append(body)
        return httpx.Response(
            200, json={"session_id": key, "trajectory_ready": True, "trajectory_id": 0}
        )

    async def run():
        async def record(event):
            pass

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            return await run_group(
                RLClient("http://gateway.test", http),
                [Session("s0", "s0"), Session("s1", "s1")],
                sample,
                record,
                max_rounds=2,
            )

    results = asyncio.run(run())
    assert [r["evaluation"]["reward"] for r in results] == [1, 0]
    assert [r["reward"] for r in rewards] == pytest.approx([1, -1])
    assert [r["interaction_id"] for r in rewards] == ["s0-2", "s1-2"]

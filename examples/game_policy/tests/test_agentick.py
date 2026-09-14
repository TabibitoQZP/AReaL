# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests; no Agentick, AReaL, or model installation required."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.game_policy import agent, env_runner, evaluator
from examples.game_policy.env_runner import (
    PolicyError,
    PolicyProcess,
    policy_observation,
)
from examples.game_policy.policy_worker import compile_policy, validate_result
from examples.game_policy.prepare_data import (
    INITIAL_POLICIES,
    SMOKE_POLICY,
    seed_groups,
)


def sample() -> dict:
    return {
        "task": "GoToGoal-v0",
        "difficulty": "easy",
        "previous_policy": INITIAL_POLICIES[0],
        "previous_feedback": '{"reward": 0.0, "trials": []}',
        "evaluation_seeds": [87654321, 87654322],
    }


def test_policy_memory_persists_and_resets_between_trials() -> None:
    """The policy child retains only per-trial state across action requests."""
    code = "def act(obs, memory):\n    memory = (memory or 0) + 1\n    return memory, memory"
    with PolicyProcess(code, timeout=1.0) as process:
        assert process.request({}) == {"action": 1}
        assert process.request({}) == {"action": 2}
    assert process.process.poll() is not None
    with PolicyProcess(code, timeout=1.0) as restarted:
        assert restarted.request({}) == {"action": 1}


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef act(obs, memory):\n    return 0, None",
        "def act(obs, memory):\n    return obs.__class__, None",
        "def act(obs, memory):\n    return open('/etc/passwd').read(), None",
        "def act(obs, memory):\n    return 9, None",
        "def act(obs, memory):\n    return True, None",
        "def act(obs, memory):\n    return 1 / 0, None",
        "def broken(",
    ],
)
def test_bad_policy_becomes_policy_error(code: str) -> None:
    """Syntax, restricted operations, runtime errors and actions are failures."""
    with pytest.raises(PolicyError):
        with PolicyProcess(code, timeout=1.0) as process:
            process.request({})


def test_infinite_policy_is_killed() -> None:
    """An infinite action cannot occupy the reward worker indefinitely."""
    code = "def act(obs, memory):\n    while True:\n        pass"
    with pytest.raises(PolicyError, match="timed out"):
        with PolicyProcess(code, timeout=1.0) as process:
            process.request({})
    assert process.process.poll() is not None


def test_oversized_memory_is_rejected() -> None:
    """Candidate memory has a serialized size bound."""
    with pytest.raises(ValueError, match="Memory exceeds"):
        validate_result((1, "x" * 16_384))


def test_smoke_policy_reaches_goal_with_documented_coordinates() -> None:
    """Exercise the Python API and x/y action mapping without Agentick."""
    policy = compile_policy(SMOKE_POLICY)
    obs = {"grid": ["#####", "#..G#", "#...#", "#...#", "#####"], "position": [1, 3]}
    deltas = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    for _ in range(4):
        action, _ = validate_result(policy(obs, None))
        dx, dy = deltas[action]
        obs["position"] = [obs["position"][0] + dx, obs["position"][1] + dy]
    assert obs["position"] == [3, 1]


def test_observation_excludes_task_info_and_oracles(monkeypatch) -> None:
    """Only the documented public fields cross into generated code."""
    monkeypatch.setitem(
        sys.modules,
        "agentick.core.types",
        SimpleNamespace(
            CellType=SimpleNamespace(WALL=1),
            ObjectType=SimpleNamespace(GOAL=1),
        ),
    )
    raw = {
        "grid": {"terrain": [[1, 0]], "objects": [[0, 1]], "metadata": "private"},
        "agent": {"position": (1, 0)},
        "info": {"_optimal_path": [1, 2, 3]},
    }
    assert policy_observation(raw, 0, 20) == {
        "grid": ["#G"],
        "position": [1, 0],
        "step": 0,
        "max_steps": 20,
    }


def test_dataset_seed_sets_are_disjoint() -> None:
    """No seed is reused across feedback, scoring or train/test rows."""
    seeds = []
    for index in range(40):
        feedback, evaluation = seed_groups(index, 4, 10_000)
        seeds.extend(feedback + evaluation)
    assert len(seeds) == 320
    assert len(set(seeds)) == len(seeds)
    assert "87654321" not in json.dumps(agent.build_messages(sample()))


@pytest.mark.parametrize("success", [True, False])
def test_runner_uses_success_flag_instead_of_raw_reward(monkeypatch, success) -> None:
    """Termination and a positive environment reward alone do not mean success."""
    closed = []
    fake_env = SimpleNamespace(
        max_steps=20,
        reset=lambda **kwargs: ({}, {}),
        step=lambda action: ({}, 999.0, True, False, {"success": success}),
        close=lambda: closed.append(True),
    )
    monkeypatch.setitem(
        sys.modules, "agentick", SimpleNamespace(make=lambda *a, **kw: fake_env)
    )
    monkeypatch.setattr(env_runner, "policy_observation", lambda *a: {})
    result = env_runner.run_trial(
        {
            "task": "GoToGoal-v0",
            "difficulty": "easy",
            "seed": 1,
            "code": INITIAL_POLICIES[0],
            "action_timeout": 1.0,
        }
    )
    assert result["success"] is success
    assert result["steps"] == 1
    assert closed == [True]


def test_runner_policy_failure_does_not_execute_environment_step(monkeypatch) -> None:
    """A failed action produces zero success and records completed steps only."""

    def unexpected_step(action):
        raise AssertionError("Invalid policy must never reach env.step")

    fake_env = SimpleNamespace(
        max_steps=20,
        reset=lambda **kw: ({}, {}),
        step=unexpected_step,
        close=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules, "agentick", SimpleNamespace(make=lambda *a, **kw: fake_env)
    )
    monkeypatch.setattr(env_runner, "policy_observation", lambda *a: {})
    result = env_runner.run_trial(
        {
            "task": "GoToGoal-v0",
            "difficulty": "easy",
            "seed": 1,
            "code": "def act(obs, memory):\n    return 9, None",
            "action_timeout": 1.0,
        }
    )
    assert result["success"] is False
    assert result["steps"] == 0
    assert result["error"]


def test_extract_policy_requires_single_complete_program() -> None:
    """Prose and multiple fenced programs do not silently change the candidate."""
    code = INITIAL_POLICIES[0].strip()
    assert evaluator.extract_policy(f"```python\n{code}\n```") == code
    with pytest.raises(ValueError):
        evaluator.extract_policy(f"Here is code:\n```python\n{code}\n```")
    with pytest.raises(ValueError):
        evaluator.extract_policy("")


@pytest.mark.parametrize("returncode, expected", [(0, 0.5), (1, None)])
def test_evaluator_aggregates_success_and_propagates_runner_errors(
    monkeypatch, returncode, expected
) -> None:
    """Runner failures never masquerade as unsuccessful game episodes."""
    results = iter([True, False])

    async def launch(*args, **kwargs):
        payload = json.dumps(
            {"success": next(results), "steps": 1, "error": ""}
        ).encode()
        return SimpleNamespace(
            returncode=returncode,
            communicate=AsyncMock(return_value=(payload, b"missing dependency")),
        )

    monkeypatch.setattr(evaluator.asyncio, "create_subprocess_exec", launch)
    if expected is None:
        with pytest.raises(RuntimeError, match="missing dependency"):
            asyncio.run(evaluator.evaluate_policy(INITIAL_POLICIES[0], [1, 2]))
    else:
        result = asyncio.run(evaluator.evaluate_policy(INITIAL_POLICIES[0], [1, 2]))
        assert result["reward"] == expected


def test_outer_timeout_kills_environment_process(monkeypatch) -> None:
    """The outer runner deadline raises instead of producing training reward."""
    original_launch = asyncio.create_subprocess_exec
    processes = []

    async def launch(*args, **kwargs):
        process = await original_launch(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(evaluator.asyncio, "create_subprocess_exec", launch)
    with pytest.raises(TimeoutError):
        asyncio.run(
            evaluator.evaluate_policy("", [1], action_timeout=0.01, trial_timeout=0.1)
        )
    assert processes[0].returncode is not None


@pytest.mark.parametrize("output, expected", [(INITIAL_POLICIES[0], 0.75), ("", 0.0)])
def test_agent_calls_actor_once_and_returns_scalar(
    monkeypatch, output, expected
) -> None:
    """Exactly one captured actor completion generates one candidate reward."""
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=output)),
            ]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    received = {}

    def make_client(**kwargs):
        received.update(kwargs)
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=make_client))
    evaluate = AsyncMock(return_value={"reward": 0.75})
    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    result = asyncio.run(
        agent.AgentickPolicyAgent(max_completion_tokens=2048).run(
            sample(),
            base_url="http://test.invalid",
            api_key="test-session",
            http_client=object(),
        )
    )
    assert type(result) is float and result == expected
    create.assert_awaited_once()
    assert received["api_key"] == "test-session"
    assert received["max_retries"] == 0
    assert evaluate.await_count == bool(output)

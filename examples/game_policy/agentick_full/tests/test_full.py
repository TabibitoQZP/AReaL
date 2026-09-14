# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import agent
import httpx
import prepare_data
import pytest
from env_runner import PolicyProcess, policy_observation
from policy_worker import compile_policy, validate_result
from rl_client import FrozenClient, ProtocolError, RLClient, Session
from tasks import (
    AGENTICK_REVISION,
    DIFFICULTIES,
    TASK_SPLIT,
    TASKS,
    load_rows,
    training_schedule,
)

CODE1 = "def act(obs, memory):\n    return 0, memory"
CODE2 = "def act(obs, memory):\n    return 5, memory"


def row(split="train"):
    task = "GoToGoal-v0" if split == "train" else "RecursiveRooms-v0"
    return {
        "id": "case",
        "split": split,
        "revision": AGENTICK_REVISION,
        "task": task,
        "category": "navigation",
        "difficulty": "easy",
        "instruction": "Official fixture instruction",
        "feedback_seeds": [10, 11],
        "score_seeds": [20, 21],
    }


async def ignore(event):
    pass


def test_fixed_split_and_balanced_schedule():
    assert len(TASKS) == 37
    assert Counter(split for _, split in TASKS.values()) == {"train": 30, "test": 7}
    assert sum(len(a) + len(b) for a, b in TASK_SPLIT.values()) == 37
    schedule = training_schedule(384, 7)
    assert schedule == training_schedule(384, 7)
    assert schedule != training_schedule(384, 8)
    counts = Counter((TASKS[t][0], d) for t, d in schedule)
    assert len(counts) == 24 and set(counts.values()) == {16}
    for category, (names, _) in TASK_SPLIT.items():
        for difficulty in DIFFICULTIES:
            task_counts = Counter(
                t for t, d in schedule if d == difficulty and TASKS[t][0] == category
            )
            assert set(task_counts) == set(names)
            assert max(task_counts.values()) - min(task_counts.values()) <= 1


def test_data_generation_partitions_and_seed_roles(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prepare_data,
        "official_instructions",
        lambda: {(t, d): f"{t}: {d}" for t in TASKS for d in DIFFICULTIES},
    )
    output = tmp_path / "data"
    prepare_data.prepare(output, train_groups=24, test_repeats=1)
    train, test = [load_rows(output / f"{s}.jsonl", s) for s in ("train", "test")]
    assert len(train) == 24 and len(test) == 28
    sets = [
        {seed for r in rows for seed in r[field]}
        for rows in (train, test)
        for field in ("feedback_seeds", "score_seeds")
    ]
    for i, seeds in enumerate(sets):
        assert all(not seeds & other for other in sets[i + 1 :])
    with pytest.raises(FileExistsError):
        prepare_data.prepare(output, train_groups=24)


@pytest.mark.parametrize(
    "change",
    [
        {"task": "RecursiveRooms-v0"},
        {"split": "test"},
        {"difficulty": "invalid"},
        {"revision": "other"},
        {"score_seeds": [10]},
        {"score_seeds": [True]},
    ],
)
def test_loader_rejects_split_leakage_and_invalid_data(tmp_path, change):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({**row(), **change}) + "\n")
    with pytest.raises(ValueError):
        load_rows(data, "train")


def test_stop_reuses_latest_code_and_scores_only_fresh_seeds(monkeypatch):
    client = SimpleNamespace(
        generate=AsyncMock(side_effect=[("c1", CODE1), ("c2", "STOP")])
    )
    evaluate = AsyncMock(
        side_effect=[{"reward": 0.5, "trials": []}, {"reward": 1.0, "trials": []}]
    )
    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    session = Session("s", "secret")
    result = asyncio.run(agent.refine(client, session, row(), ignore, max_rounds=4))
    assert result["policy"] == CODE1 and result["calls"] == 2
    assert result["interaction_ids"][-1] == "c2" and result["stop_reason"] == "stop"
    assert [call.args[1] for call in evaluate.await_args_list] == [[10, 11], [20, 21]]
    assert all(call.args[0] is session for call in client.generate.await_args_list)
    messages = client.generate.await_args_list[-1].args[1]
    state = json.loads(messages[-1]["content"])
    assert len(messages) == 2 and state["previous_policy"] == CODE1
    assert state["remaining_calls_including_this"] == 3
    assert "score_seeds" not in state and "secret" not in json.dumps(messages)


def test_budget_uses_final_code_and_rebuilds_short_context(monkeypatch):
    third = "def act(obs, memory):\n    return 2, memory"
    client = SimpleNamespace(
        generate=AsyncMock(side_effect=[("c1", CODE1), ("c2", CODE2), ("c3", third)])
    )
    evaluate = AsyncMock(return_value={"reward": 0.0, "trials": []})
    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    result = asyncio.run(agent.refine(client, None, row(), ignore, 3))
    messages = client.generate.await_args_list[-1].args[1]
    assert CODE1 not in json.loads(messages[-1]["content"])["previous_policy"]
    assert json.loads(messages[-1]["content"])["previous_policy"] == CODE2
    assert result["policy"] == third and result["calls"] == 3
    assert evaluate.await_args_list[-1].args[:2] == (third, [20, 21])


def test_malformed_final_replacement_does_not_fall_back(monkeypatch):
    client = SimpleNamespace(
        generate=AsyncMock(side_effect=[("c1", CODE1), ("c2", "")])
    )
    evaluate = AsyncMock(return_value={"reward": 1.0, "trials": []})
    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    result = asyncio.run(agent.refine(client, None, row(), ignore, 2))
    assert result["policy"] is None and result["evaluation"]["reward"] == 0
    assert evaluate.await_count == 1


def test_initial_stop_consumes_budget_and_requests_code(monkeypatch):
    client = SimpleNamespace(
        generate=AsyncMock(side_effect=[("c1", "STOP"), ("c2", CODE1)])
    )
    monkeypatch.setattr(
        agent, "evaluate_policy", AsyncMock(return_value={"reward": 1.0})
    )
    result = asyncio.run(agent.refine(client, None, row(), ignore, 2))
    state = json.loads(client.generate.await_args_list[-1].args[1][-1]["content"])
    assert "STOP requires" in state["previous_feedback"]["format_error"]
    assert result["policy"] == CODE1


def test_feedback_omits_seeds_and_bounds_observation_detail():
    feedback = agent.feedback_view(
        {
            "reward": 0.0,
            "trials": [
                {
                    "seed": 123,
                    "success": False,
                    "steps": 2,
                    "error": "",
                    "final": "x" * 50000,
                }
            ],
        }
    )
    assert "seed" not in json.dumps(feedback)
    assert feedback["example_truncated"] is True
    assert len(json.dumps(feedback)) < 24576


@pytest.mark.parametrize("rewards", [[0, 0, 1, 1], [0, 0, 0, 1], [0.2, 0.3, 0.9]])
def test_group_normalization_centers_final_scores(rewards):
    values = agent.normalize_rewards(rewards)
    assert math.fsum(values) == pytest.approx(0, abs=1e-7)
    assert math.fsum(x * x for x in values) / len(values) == pytest.approx(1, abs=1e-6)


def test_constant_groups_and_invalid_rewards():
    assert agent.normalize_rewards([0, 0]) == [0, 0]
    for scores in ([0], [math.nan, 0], [0, math.inf], [-1, 1]):
        with pytest.raises(ValueError):
            agent.normalize_rewards(scores)


def test_http_group_uses_same_session_each_round_and_terminal_reward_only(monkeypatch):
    requests, events = [], []
    counts = Counter()

    def handle(request):
        body = json.loads(request.content)
        key = request.headers["Authorization"].removeprefix("Bearer ")
        requests.append((request.url.path, key, body))
        if request.url.path == "/v1/chat/completions":
            counts[key] += 1
            output = (CODE1 if key == "k0" else CODE2) if counts[key] == 1 else "STOP"
            return httpx.Response(
                200,
                json={
                    "id": f"{key}-{counts[key]}",
                    "choices": [{"message": {"content": output}}],
                },
            )
        assert request.url.path == "/rl/set_reward"
        assert counts == {"k0": 2, "k1": 2}
        return httpx.Response(
            200,
            json={
                "session_id": "s" + key[-1],
                "trajectory_ready": True,
                "trajectory_id": 0,
            },
        )

    async def evaluate(code, seeds, *args):
        # Feedback scores deliberately disagree with final scores.
        score = int(code == CODE2) if seeds == [20, 21] else int(code == CODE1)
        return {"reward": score, "trials": []}

    async def record(event):
        events.append(event)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            return await agent.run_group(
                RLClient("http://gateway.test", http),
                [Session("s0", "k0"), Session("s1", "k1")],
                row(),
                record,
            )

    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    results = asyncio.run(run())
    posts = [(key, body) for path, key, body in requests if path == "/rl/set_reward"]
    assert len(posts) == 2
    assert [body["interaction_id"] for _, body in posts] == ["k0-2", "k1-2"]
    assert [body["reward"] for _, body in posts] == pytest.approx([-1, 1])
    assert [r["calls"] for r in results] == [2, 2]
    normalized = next(
        i for i, event in enumerate(events) if event["status"] == "normalized"
    )
    assert all(
        i > normalized
        for i, event in enumerate(events)
        if event["status"] == "rewarded"
    )
    assert '"k0"' not in json.dumps(events)


def test_group_infrastructure_failure_cancels_peers_without_reward(monkeypatch):
    cancelled = []
    started = asyncio.Event()

    async def refine(client, session, *args, **kwargs):
        if session.session_id == "s0":
            await started.wait()
            raise RuntimeError("missing environment")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    client = SimpleNamespace(set_reward=AsyncMock())
    monkeypatch.setattr(agent, "refine", refine)
    with pytest.raises(RuntimeError, match="missing environment"):
        asyncio.run(
            agent.run_group(
                client, [Session("s0", "k0"), Session("s1", "k1")], row(), ignore
            )
        )
    assert cancelled == [True] and client.set_reward.await_count == 0


def test_partial_reward_failure_preserves_entire_normalized_group(monkeypatch):
    monkeypatch.setattr(
        agent,
        "refine",
        AsyncMock(
            side_effect=[
                {"evaluation": {"reward": 0.0}, "interaction_ids": ["c0"]},
                {"evaluation": {"reward": 1.0}, "interaction_ids": ["c1"]},
            ]
        ),
    )
    client = SimpleNamespace(
        set_reward=AsyncMock(
            side_effect=[{"trajectory_id": 0}, ProtocolError("unknown outcome")]
        )
    )
    events = []

    async def record(event):
        events.append(event)

    with pytest.raises(ProtocolError):
        asyncio.run(
            agent.run_group(
                client, [Session("s0", "k0"), Session("s1", "k1")], row(), record
            )
        )
    assert events[0]["status"] == "normalized" and len(events[0]["candidates"]) == 2
    assert len([e for e in events if e["status"] == "rewarded"]) == 1


def test_frozen_evaluation_never_calls_rl_and_averages_candidates(
    monkeypatch, tmp_path
):
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps(row("test")) + "\n")
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": f"c{len(requests)}",
                "choices": [{"message": {"content": CODE1}}],
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        agent.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    )
    monkeypatch.setattr(
        agent,
        "evaluate_policy",
        AsyncMock(side_effect=[{"reward": 0.0}, {"reward": 1.0}]),
    )
    args = argparse.Namespace(
        mode="eval",
        data=data,
        output=tmp_path / "log.jsonl",
        max_rounds=1,
        max_tokens=100,
        temperature=1.0,
        samples=2,
        offset=0,
        limit=None,
        base_url="http://frozen.test/v1",
        api_key_env="UNSET_FULL_TEST_KEY",
        model="frozen",
    )
    asyncio.run(agent.drive(args))
    events = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert events[-1]["category_macro_success"] == 0.5
    assert len(requests) == 2
    assert not any(e["status"] in ("normalized", "rewarded") for e in events)
    assert not hasattr(FrozenClient, "set_reward")


def test_observation_uses_public_fields_and_masks_fog():
    obs = {
        "grid": {
            "height": 1,
            "width": 2,
            "terrain": [[0, 1]],
            "objects": [[0, 7]],
            "agents": [[1, 2]],
            "metadata": [[0, -1]],
        },
        "agent": {"position": [0, 0]},
        "annotations": {"fog_cells": ["1,0"]},
        "entities": [{"secret_position": [1, 0]}],
        "info": {"task_config": {"visible_clue": "up"}, "oracle": "hidden"},
    }
    public = policy_observation(obs, 0, 10, {"interact": 5}, {})
    assert public["grid"]["objects"] == [[0, -1]]
    assert obs["grid"]["objects"] == [[0, 7]]
    assert "entities" not in public and "info" not in public
    assert "hidden" not in json.dumps(public)
    assert public["task_state"] == {"visible_clue": "up"}


def test_policy_process_supports_interact_and_preserves_memory():
    policy = "def act(obs, memory):\n    return (5 if memory is None else 4), (memory or 0) + 1"
    fn = compile_policy(policy)
    assert validate_result(fn({}, None)) == (5, 1)
    with PolicyProcess(policy, timeout=2) as process:
        assert process.request({}) == {"action": 5}
        assert process.request({}) == {"action": 4}
    with pytest.raises(ValueError):
        validate_result((9, None))


def test_copied_directory_imports_without_areal_or_agentick(tmp_path):
    source = Path(__file__).resolve().parents[1]
    dest = tmp_path / "agentick_full"
    dest.mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, dest / path.name)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agent, prepare_data, sys; "
            "assert not {'areal', 'torch', 'agentick', 'datasets', 'openai'} & sys.modules.keys()",
        ],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_train_driver_uses_preissued_sessions_and_selected_chunk(monkeypatch, tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps({**row(), "id": f"row{i}"}) for i in range(3)))
    credentials = tmp_path / "sessions.jsonl"
    credentials.write_text(
        "\n".join(
            json.dumps({"session_id": f"s{i}", "session_api_key": f"private-key-{i}"})
            for i in range(2)
        )
    )
    requests = []

    def handle(request):
        requests.append(request)
        key = request.headers["Authorization"].removeprefix("Bearer ")
        index = int(key[-1])
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={"id": f"c{index}", "choices": [{"message": {"content": CODE1}}]},
            )
        assert request.url.path == "/rl/set_reward"
        return httpx.Response(
            200,
            json={
                "session_id": f"s{index}",
                "trajectory_ready": True,
                "trajectory_id": 0,
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        agent.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    )
    monkeypatch.setattr(
        agent, "evaluate_policy", AsyncMock(return_value={"reward": 1.0})
    )
    args = argparse.Namespace(
        mode="train",
        data=data,
        output=tmp_path / "log.jsonl",
        max_rounds=1,
        max_tokens=100,
        temperature=1.0,
        group_size=2,
        epochs=1,
        gateway="http://gateway.test",
        sessions=credentials,
        admin_key_env=None,
        model="default",
        offset=1,
        limit=1,
    )
    asyncio.run(agent.drive(args))
    events = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert [e["id"] for e in events if e["status"] == "sample"] == ["row1"]
    assert events[-1]["candidates"] == 2
    assert len(requests) == 4
    assert "private-key" not in args.output.read_text()
    assert not any("start_session" in str(r.url) for r in requests)


def test_macro_average_weights_categories_equally():
    summaries = [
        {
            "category": "navigation",
            "task": "RecursiveRooms-v0",
            "difficulty": "easy",
            "calls": 1,
            "reward": 1.0,
        },
        {
            "category": "planning",
            "task": "ToolUse-v0",
            "difficulty": "easy",
            "calls": 2,
            "reward": 0.0,
        },
        {
            "category": "planning",
            "task": "PackingPuzzle-v0",
            "difficulty": "easy",
            "calls": 3,
            "reward": 1.0,
        },
    ]
    summary = agent.summarize(summaries)
    assert summary["category_macro_success"] == 0.75
    assert summary["by_difficulty"]["easy"] == 0.75
    assert summary["mean_calls"] == 2


def test_signed_reward_and_transport_failure_are_not_retried():
    calls = []

    def fail(request):
        calls.append(request)
        assert json.loads(request.content)["reward"] == -1.5
        raise httpx.ReadTimeout("unknown outcome", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
            await RLClient("http://gateway.test", http).set_reward(
                Session("s", "key"), "c", -1.5
            )

    with pytest.raises(ProtocolError, match="outcome unknown"):
        asyncio.run(run())
    assert len(calls) == 1

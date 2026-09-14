# SPDX-License-Identifier: Apache-2.0
"""HTTP lifecycle and standalone deployment tests without a training runtime."""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import agent
import httpx
import prepare_data
import pytest
import rl_client
from rl_client import ProtocolError, RLClient, Session


class FakeGateway:
    """Model the native one-completion, reward-finalized session contract."""

    def __init__(self):
        self.requests = []
        self.sessions = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, body = request.url.path, json.loads(request.content)
        key = request.headers["Authorization"].removeprefix("Bearer ")
        self.requests.append((path, key, body))
        if path == "/rl/start_session":
            assert key == "admin-secret"
            assert body["group_size"] == 1
            sid = f"s{len(self.requests)}"
            self.sessions[f"key-{sid}"] = sid
            return httpx.Response(
                201,
                json={
                    "group_id": sid,
                    "sessions": [
                        {
                            "session_id": sid,
                            "session_api_key": f"key-{sid}",
                        }
                    ],
                },
            )
        assert key in self.sessions
        sid = self.sessions[key]
        if path == "/v1/chat/completions":
            assert body["n"] == 1 and body["stream"] is False
            return httpx.Response(
                200,
                json={
                    "id": f"cmpl-{sid}",
                    "choices": [
                        {
                            "message": {"content": prepare_data.INITIAL_POLICIES[0]},
                        }
                    ],
                },
            )
        if path == "/rl/set_reward":
            assert body["interaction_id"] == f"cmpl-{sid}"
            del self.sessions[key]  # Trainer may consume and remove immediately.
            return httpx.Response(
                200,
                json={
                    "session_id": sid,
                    "trajectory_ready": True,
                    "trajectory_id": 0,
                    "ready_transition": True,
                },
            )
        raise AssertionError(f"Unexpected endpoint {path}")


def row() -> dict:
    return {
        "id": "sample",
        "task": "GoToGoal-v0",
        "difficulty": "easy",
        "previous_policy": prepare_data.INITIAL_POLICIES[0],
        "previous_feedback": '{"reward": 0, "trials": []}',
        "evaluation_seeds": [1, 2],
    }


def test_http_group_lifecycle_uses_session_auth_and_relative_rewards(
    monkeypatch,
) -> None:
    """All four completions precede reward submission; signed values survive HTTP."""
    gateway = FakeGateway()
    monkeypatch.setattr(
        agent,
        "evaluate_policy",
        AsyncMock(
            side_effect=[{"reward": r, "trials": []} for r in (0.0, 0.0, 1.0, 1.0)]
        ),
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(gateway.handle)
        ) as http:
            client = RLClient("http://gateway.test", http)
            sessions = [
                await client.start_session("sample", "admin-secret") for _ in range(4)
            ]
            return await agent.run_group(client, sessions, row(), AsyncMock())

    results = asyncio.run(run())
    assert [r["normalized_reward"] for r in results] == pytest.approx([-1, -1, 1, 1])
    assert [r[0] for r in gateway.requests] == ["/rl/start_session"] * 4 + [
        "/v1/chat/completions"
    ] * 4 + ["/rl/set_reward"] * 4
    assert [r[1] for r in gateway.requests[4:8]] == [r[1] for r in gateway.requests[8:]]
    assert [r[2]["reward"] for r in gateway.requests[8:]] == pytest.approx(
        [-1, -1, 1, 1]
    )
    assert not gateway.sessions


@pytest.mark.parametrize(
    "path", ["/rl/start_session", "/v1/chat/completions", "/rl/set_reward"]
)
def test_unknown_http_outcome_is_not_replayed(path) -> None:
    """Replaying a mutation can create orphan sessions or double-submit data."""
    requests = []

    def fail(request):
        requests.append(request)
        raise httpx.ReadTimeout("read timed out", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
            await RLClient("http://gateway.test", http)._post(path, "secret", {})

    with pytest.raises(ProtocolError, match="outcome unknown"):
        asyncio.run(run())
    assert len(requests) == 1


def test_pending_reward_is_not_reported_as_completed() -> None:
    """A delayed trajectory boundary cannot silently mix adjacent candidates."""

    async def run():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "session_id": "s",
                    "trajectory_ready": False,
                    "trajectory_id": None,
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as http:
            await RLClient("http://gateway.test", http).set_reward(
                Session("s", "key"), "cmpl", 0.5
            )

    with pytest.raises(ProtocolError, match="ready trajectory"):
        asyncio.run(run())


def test_reward_failure_keeps_normalized_group_for_review(monkeypatch) -> None:
    """A partial submission leaves the entire group and its computed values logged."""
    events = []
    client = argparse.Namespace(
        generate=AsyncMock(return_value=("cmpl", prepare_data.INITIAL_POLICIES[0])),
        set_reward=AsyncMock(
            side_effect=[{"trajectory_id": 0}, ProtocolError("outcome unknown")]
        ),
    )
    monkeypatch.setattr(
        agent,
        "evaluate_policy",
        AsyncMock(side_effect=[{"reward": 0.0}, {"reward": 1.0}]),
    )

    async def record(event):
        events.append(event)

    with pytest.raises(ProtocolError, match="outcome unknown"):
        asyncio.run(
            agent.run_group(
                client, [Session("s0", "k0"), Session("s1", "k1")], row(), record
            )
        )
    normalized = next(e for e in events if e["status"] == "normalized")
    assert [r["normalized_reward"] for r in normalized["candidates"]] == pytest.approx(
        [-1, 1]
    )
    assert len([e for e in events if e["status"] == "rewarded"]) == 1
    assert client.set_reward.await_count == 2


def test_provision_writes_private_session_file(monkeypatch, tmp_path) -> None:
    """Provisioning does not print credentials or permit overwriting a key file."""
    monkeypatch.setenv("TEST_ADMIN", "admin-secret")
    start = AsyncMock(return_value=Session("s", "session-secret"))
    monkeypatch.setattr(RLClient, "start_session", start)
    output = tmp_path / "sessions.jsonl"
    args = argparse.Namespace(
        admin_key_env="TEST_ADMIN",
        count=1,
        output=output,
        timeout=1,
        gateway="http://gateway.test",
        task_prefix="task",
    )
    asyncio.run(rl_client.provision(args))
    assert output.stat().st_mode & 0o777 == 0o600
    assert rl_client.read_sessions(output) == [Session("s", "session-secret")]
    assert "session-secret" not in repr(Session("s", "session-secret"))
    with pytest.raises(FileExistsError):
        asyncio.run(rl_client.provision(args))
    assert start.await_count == 1


def test_driver_uses_preissued_keys_without_admin_and_logs_no_credentials(
    monkeypatch, tmp_path
) -> None:
    """The low-privilege deployment can submit all jobs without requesting sessions."""
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(row()) + "\n")
    keys = tmp_path / "sessions.jsonl"
    keys.write_text(
        "\n".join(
            json.dumps({"session_id": f"s{i}", "session_api_key": f"key-s{i}"})
            for i in range(4)
        )
    )
    gateway = FakeGateway()
    gateway.sessions = {f"key-s{i}": f"s{i}" for i in range(4)}
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        agent.httpx,
        "AsyncClient",
        lambda **kw: real_client(
            transport=httpx.MockTransport(gateway.handle),
            **kw,
        ),
    )
    monkeypatch.setattr(
        agent, "evaluate_policy", AsyncMock(return_value={"reward": 0.5, "trials": []})
    )
    args = argparse.Namespace(
        data=dataset,
        output=tmp_path / "run.jsonl",
        sessions=keys,
        admin_key_env=None,
        epochs=2,
        group_size=2,
        max_tokens=2048,
        temperature=1.0,
        seed=1,
        gateway="http://gateway.test",
    )
    asyncio.run(agent.drive(args))
    events = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len([e for e in events if e["status"] == "rewarded"]) == 4
    assert "key-s" not in args.output.read_text()
    assert "session_api_key" not in args.output.read_text()
    assert all(path != "/rl/start_session" for path, _, _ in gateway.requests)
    with pytest.raises(FileExistsError):
        asyncio.run(agent.drive(args))


def test_duplicate_session_keys_fail_before_generation(tmp_path) -> None:
    """Candidates must not share a session, even when sample IDs are identical."""
    path = tmp_path / "sessions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"session_id": f"s{i}", "session_api_key": "same"})
            for i in range(2)
        )
    )
    with pytest.raises(ValueError, match="unique session"):
        rl_client.read_sessions(path)


def test_prepare_data_writes_standalone_jsonl(monkeypatch, tmp_path) -> None:
    """Prepared data needs neither Arrow nor an AReaL dataset loader."""
    monkeypatch.setattr(
        prepare_data,
        "evaluate_policy",
        AsyncMock(return_value={"reward": 1.0, "trials": []}),
    )
    output = tmp_path / "dataset"
    asyncio.run(
        prepare_data.prepare(
            argparse.Namespace(
                output=output,
                train_size=2,
                valid_size=1,
                seeds_per_policy=2,
                start_seed=10,
            )
        )
    )
    assert len(agent.load_rows(output / "train.jsonl")) == 2
    assert len(agent.load_rows(output / "valid.jsonl")) == 1
    assert (output / "preparation.json").exists()


def test_client_runs_outside_areal_checkout(tmp_path) -> None:
    """Copy just client modules and execute them with no repo on sys.path."""
    package = tmp_path / "agentick"
    package.mkdir()
    source = Path(agent.__file__).parent
    for name in (
        "agent.py",
        "rl_client.py",
        "evaluator.py",
        "prepare_data.py",
        "env_runner.py",
        "policy_worker.py",
    ):
        shutil.copy(source / name, package / name)
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    script = "import sys; import agent, prepare_data; assert not any(m in sys.modules for m in ('areal', 'torch', 'datasets', 'openai'))"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

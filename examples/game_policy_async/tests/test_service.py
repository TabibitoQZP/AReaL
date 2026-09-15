# SPDX-License-Identifier: Apache-2.0
"""Test the game-only HTTP boundary and refinement inside the inline agent."""

import asyncio
import json
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from aiohttp import web

from examples.game_policy_async import agent
from examples.game_policy_async.agentick_service import server
from examples.game_policy_async.client import PolicyEvaluator
from examples.game_policy_async.evaluate import evaluate as frozen_evaluate
from examples.game_policy_async.evaluate import summarize
from examples.game_policy_async.tasks import AGENTICK_REVISION

CODE = "def act(obs, memory):\n    return 0, memory"


def row():
    return {
        "id": "train-0",
        "split": "train",
        "revision": AGENTICK_REVISION,
        "task": "GoToGoal-v0",
        "category": "navigation",
        "difficulty": "easy",
        "instruction": "Reach the goal",
        "feedback_seeds": [10],
        "score_seeds": [20],
    }


def payload(seed=0):
    return {"task": "GoToGoal-v0", "difficulty": "easy", "seed": seed, "code": CODE}


@asynccontextmanager
async def serve(app):
    runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_inline_agent_owns_model_calls_and_only_sends_execution_fields(
    tmp_path, monkeypatch
):
    """Session credentials, instructions and refinement state never cross the game API."""
    generations, games = [], []

    async def model(request):
        generations.append((request.headers["Authorization"], await request.json()))
        answer = CODE if len(generations) == 1 else "STOP"
        return web.json_response(
            {
                "id": f"c{len(generations)}",
                "choices": [
                    {
                        "message": {
                            "content": answer,
                            "reasoning_content": "private-plan",
                        }
                    }
                ],
            }
        )

    async def game(request):
        body = await request.json()
        games.append(body)
        assert set(body) == {"task", "difficulty", "seed", "code"}
        assert request.headers["Authorization"] == "Bearer service-key"
        return web.json_response(
            {
                "task": body["task"],
                "difficulty": body["difficulty"],
                "seed": body["seed"],
                "success": body["seed"] != 21,
                "steps": 1,
            }
        )

    model_app, game_app = web.Application(), web.Application()
    model_app.router.add_post("/v1/chat/completions", model)
    game_app.router.add_post("/evaluate", game)
    monkeypatch.setenv("AGENTICK_SERVICE_KEY", "service-key")
    async with (
        serve(model_app) as model_url,
        serve(game_app) as game_url,
        httpx.AsyncClient() as http,
    ):
        runner = agent.AgentickAgent(
            game_url, model="default", max_completion_tokens=100, log_dir=str(tmp_path)
        )
        reward = await runner.run(
            {**row(), "score_seeds": [20, 21]},
            base_url=f"{model_url}/v1",
            api_key="session-key",
            session_id="session-id",
            http_client=http,
            proxy_gateway_api_key="control-key",
            worker_runtime=object(),
        )
    assert reward == 0.5 and type(reward) is float
    assert [g["seed"] for g in games] == [10, 20, 21]
    assert all(key == "Bearer session-key" for key, _ in generations)
    state = json.loads(generations[1][1]["messages"][-1]["content"])
    assert (
        state["previous_policy"] == CODE
        and state["remaining_calls_including_this"] == 3
    )
    assert all(len(body["messages"]) == 2 and body["n"] == 1 for _, body in generations)
    logs = "".join(p.read_text() for p in tmp_path.glob("*.jsonl"))
    for secret in ("session-key", "control-key", "service-key", "private-plan"):
        assert secret not in logs and secret not in json.dumps(games)
    assert "session-id" not in json.dumps(games) and "instruction" not in json.dumps(
        games
    )
    assert '"status": "scored"' in logs


@pytest.mark.asyncio
async def test_service_runs_requests_concurrently_with_bounded_cpu(monkeypatch):
    running = peak = 0
    entered = asyncio.Event()

    async def run_policy(code, seed, task, difficulty):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running == 2:
            entered.set()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await asyncio.sleep(0.02)
            return {"success": True, "steps": 1}
        finally:
            running -= 1

    monkeypatch.setattr(server, "run_policy", run_policy)
    async with (
        serve(server.create_app("key", evaluations=2)) as url,
        httpx.AsyncClient() as http,
    ):
        responses = await asyncio.gather(
            *[
                http.post(
                    f"{url}/evaluate",
                    json=payload(i),
                    headers={"Authorization": "Bearer key"},
                )
                for i in range(4)
            ]
        )
    assert peak == 2 and all(r.status_code == 200 for r in responses)
    assert [r.json()["seed"] for r in responses] == list(range(4))
    assert all("reward" not in r.json() for r in responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_disconnect_or_deadline_cancels_game(monkeypatch, timeout):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def run_policy(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(server, "run_policy", run_policy)
    async with (
        serve(server.create_app("key", timeout=0.3 if timeout else 30)) as url,
        httpx.AsyncClient() as http,
    ):
        pending = asyncio.create_task(
            http.post(
                f"{url}/evaluate",
                json=payload(),
                headers={"Authorization": "Bearer key"},
            )
        )
        await asyncio.wait_for(entered.wait(), 3)
        if timeout:
            assert (await pending).status_code == 504
        else:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await asyncio.wait_for(cancelled.wait(), 3)


@pytest.mark.asyncio
async def test_service_rejects_agent_payload_and_reports_infrastructure_failure(
    monkeypatch,
):
    run = AsyncMock(side_effect=RuntimeError("private-error-body"))
    monkeypatch.setattr(server, "run_policy", run)
    async with serve(server.create_app("key")) as url, httpx.AsyncClient() as http:
        assert (await http.post(f"{url}/evaluate", json=payload())).status_code == 401
        for invalid in (
            {},
            {**payload(), "api_key": "session-secret"},
            {**payload(), "seed": True},
        ):
            assert (
                await http.post(
                    f"{url}/evaluate",
                    json=invalid,
                    headers={"Authorization": "Bearer key"},
                )
            ).status_code == 400
        response = await http.post(
            f"{url}/evaluate", json=payload(), headers={"Authorization": "Bearer key"}
        )
        assert response.status_code == 502 and "private-error-body" not in response.text
    assert run.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("last", ["", "<think>unfinished", "```python\nbad\n``` prose"])
async def test_invalid_replacement_never_reuses_old_policy(last):
    client = SimpleNamespace(
        generate=AsyncMock(side_effect=[("c1", CODE), ("c2", last)])
    )
    evaluate = AsyncMock(return_value={"reward": 1.0, "trials": []})
    result = await agent.refine(client, row(), AsyncMock(), 2, evaluate=evaluate)
    assert result["evaluation"]["reward"] == 0.0 and result["policy"] is None
    assert evaluate.await_count == 1


@pytest.mark.asyncio
async def test_budget_rebuilds_short_context_and_scores_latest_policy():
    codes = [
        CODE,
        CODE.replace("return 0", "return 1"),
        CODE.replace("return 0", "return 2"),
    ]
    client = SimpleNamespace(
        generate=AsyncMock(
            side_effect=[(f"c{i}", code) for i, code in enumerate(codes)]
        )
    )
    evaluate = AsyncMock(return_value={"reward": 0.5, "trials": []})
    result = await agent.refine(client, row(), AsyncMock(), 3, evaluate=evaluate)
    assert result["policy"] == codes[-1] and result["calls"] == 3
    assert evaluate.await_args.args[:2] == (codes[-1], [20])
    messages = client.generate.await_args.args[0]
    state = json.loads(messages[-1]["content"])
    assert len(messages) == 2 and state["previous_policy"] == codes[1]
    assert state["remaining_calls_including_this"] == 1 and "score_seeds" not in state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [{"seed": 1}, {"success": 1}, {"task": "other"}, {"difficulty": "hard"}]
)
async def test_evaluator_rejects_mismatched_game_result(change):
    result = {
        "task": "GoToGoal-v0",
        "difficulty": "easy",
        "seed": 0,
        "success": True,
        **change,
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=result))
    ) as http:
        with pytest.raises(RuntimeError, match="Invalid"):
            await PolicyEvaluator(http, "http://game.test", "key").evaluate(
                CODE, [0], "GoToGoal-v0", "easy"
            )


@pytest.mark.asyncio
async def test_real_game_failure_feedback_can_be_refined_to_success(
    tmp_path, monkeypatch
):
    from test_environment import SOLVER

    calls = 0

    async def model(request):
        nonlocal calls
        calls += 1
        body = await request.json()
        answer = "def act(obs, memory):\n    return 100, None"
        if calls == 2:
            feedback = json.loads(body["messages"][-1]["content"])["previous_feedback"]
            assert feedback["success_rate"] == 0 and feedback["trials"][0]["error"]
            answer = SOLVER
        return web.json_response(
            {"id": f"c{calls}", "choices": [{"message": {"content": answer}}]}
        )

    model_app = web.Application()
    model_app.router.add_post("/chat/completions", model)
    monkeypatch.setenv("AGENTICK_SERVICE_KEY", "key")
    async with (
        serve(model_app) as model_url,
        serve(server.create_app("key")) as game_url,
        httpx.AsyncClient() as http,
    ):
        runner = agent.AgentickAgent(
            game_url, max_rounds=2, model="default", max_completion_tokens=100
        )
        result = await runner.run(
            {**row(), "feedback_seeds": [0], "score_seeds": [1]},
            base_url=model_url,
            api_key="session",
            session_id="sid",
            http_client=http,
        )
        assert result == 1.0 and calls == 2


def test_frozen_summary_weights_categories_equally():
    rows = [
        {"category": c, "task": t, "difficulty": "easy"}
        for c, t in [("planning", "a"), ("planning", "b"), ("memory", "c")]
    ]
    assert (
        summarize(rows, [{"reward": r} for r in (0, 0, 1)])["category_macro_success"]
        == 0.5
    )


@pytest.mark.asyncio
async def test_frozen_driver_runs_agent_locally(tmp_path, monkeypatch):
    data = tmp_path / "test.jsonl"
    data.write_text(
        json.dumps({**row(), "task": "RecursiveRooms-v0", "split": "test"}) + "\n"
    )
    output = tmp_path / "results.jsonl"
    seen = []

    async def run(self, data, **kwargs):
        seen.append((data, kwargs))
        return 0.75

    monkeypatch.setattr(agent.AgentickAgent, "run", run)
    monkeypatch.setenv("AGENTICK_SERVICE_URL", "http://game.test")
    monkeypatch.setenv("AGENTICK_SERVICE_KEY", "service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "frozen-key")
    result = await frozen_evaluate(
        SimpleNamespace(
            data=data,
            output=output,
            base_url="http://frozen.test/v1",
            model="frozen",
            concurrency=2,
            max_rounds=4,
            max_tokens=100,
            timeout=30,
            evaluation_timeout=30,
            temperature=0.6,
        )
    )
    assert result["category_macro_success"] == 0.75
    assert seen[0][1]["api_key"] == "frozen-key" and seen[0][0]["split"] == "test"
    assert "frozen-key" not in output.read_text()


def test_execution_package_can_be_deployed_without_parent_agent(tmp_path):
    """Copy only the execution package and run a real policy in that directory."""
    source = Path(server.__file__).parent
    shutil.copytree(
        source,
        tmp_path / "agentick_service",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    script = (
        "import asyncio, json, sys\n"
        "from agentick_service.evaluator import run_policy\n"
        "r = asyncio.run(run_policy(sys.stdin.read(), 0))\n"
        "print(json.dumps({'success': r['success'], 'error': r.get('error')}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input="invalid python",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        timeout=20,
        check=True,
    )
    assert json.loads(result.stdout)["success"] is False
    assert json.loads(result.stdout)["error"]


@pytest.mark.asyncio
async def test_inline_candidates_can_generate_concurrently(monkeypatch):
    """Moving the agent locally must not serialize model calls across candidates."""
    entered = 0
    ready = asyncio.Event()

    async def model(request):
        nonlocal entered
        entered += 1
        if entered == 4:
            ready.set()
        await asyncio.wait_for(ready.wait(), 3)
        return web.json_response(
            {
                "id": request.headers["Authorization"],
                "choices": [{"message": {"content": CODE}}],
            }
        )

    model_app = web.Application()
    model_app.router.add_post("/chat/completions", model)
    monkeypatch.setenv("AGENTICK_SERVICE_KEY", "key")
    monkeypatch.setattr(server, "run_policy", AsyncMock(return_value={"success": True}))
    async with (
        serve(model_app) as model_url,
        serve(server.create_app("key")) as game_url,
        httpx.AsyncClient() as http,
    ):
        runner = agent.AgentickAgent(
            game_url, max_rounds=1, model="default", max_completion_tokens=100
        )
        rewards = await asyncio.gather(
            *[
                runner.run(
                    row(),
                    base_url=model_url,
                    api_key=f"s{i}",
                    session_id=f"s{i}",
                    http_client=http,
                )
                for i in range(4)
            ]
        )
    assert rewards == [1.0] * 4 and entered == 4

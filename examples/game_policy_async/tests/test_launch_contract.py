# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for forked workers and the model HTTP contract."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf

from examples.game_policy_async import evaluate as frozen
from examples.game_policy_async.client import ChatClient
from examples.game_policy_async.tasks import AGENTICK_REVISION
from examples.game_policy_async.train import (
    proxy_generation_kwargs,
    validate_colocation,
)


@pytest.fixture
def config(monkeypatch, tmp_path):
    """Resolve the shipped recipe without loading any GPU libraries."""
    for key in (
        "FILEROOT",
        "AGENTICK_TRAIN_DATA",
        "AREAL_PROXY_ADMIN_KEY",
        "AGENTICK_SERVICE_KEY",
    ):
        monkeypatch.setenv(key, str(tmp_path / key))
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config.yaml")
    # Filled by GenerationHyperparameters when the real PPOConfig is loaded.
    cfg.gconfig.setdefault("top_p", 1.0)
    return cfg


def test_colocation_eight_single_gpu_workers_pass(config):
    """Inference DP8 has eight workers, matching the actor's eight ranks."""
    validate_colocation(
        config,
        SimpleNamespace(world_size=8),
        SimpleNamespace(dp_size=8, tp_size=1, pp_size=1),
    )


@pytest.mark.parametrize(
    "dp,tp,pp,error",
    [
        (4, 2, 1, "worker count mismatch"),
        (8, 2, 1, "TP1/PP1"),
        (8, 1, 2, "TP1/PP1"),
    ],
)
def test_colocation_incompatible_inference_workers_rejected(config, dp, tp, pp, error):
    """Equal GPU totals alone do not satisfy forked-worker placement."""
    with pytest.raises(ValueError, match=error):
        validate_colocation(
            config,
            SimpleNamespace(world_size=8),
            SimpleNamespace(dp_size=dp, tp_size=tp, pp_size=pp),
        )


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("rollout.scheduling_strategy.fork", False, "forked colocation"),
        ("rollout.scheduling_strategy.target", "ref", "forked colocation"),
        ("actor.scheduling_spec.0.gpu", 2, "gpu=1"),
        ("actor.scheduling_spec.0.port_count", 1, "port_count"),
        ("cluster.n_nodes", 2, "one node"),
        ("cluster.n_gpus_per_node", 4, "one node"),
    ],
)
def test_colocation_invalid_resources_rejected(config, key, value, error):
    """Reject unsupported forks and local GPU oversubscription before launch."""
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError, match=error):
        validate_colocation(
            config,
            SimpleNamespace(world_size=8),
            SimpleNamespace(dp_size=8, tp_size=1, pp_size=1),
        )


@pytest.mark.asyncio
async def test_training_http_preserves_nested_thinking_options(config):
    """The exact raw HTTP body retains extra_body for AReaL create()."""
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "choices": [{"message": {"content": "def act(obs, memory): pass"}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = ChatClient(http, "http://model.test/v1", "test-key")
        await client.generate(
            [{"role": "user", "content": "Write a policy"}],
            **proxy_generation_kwargs(config.gconfig),
        )
    body = requests[0]
    assert "chat_template_kwargs" not in body
    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert body["max_completion_tokens"] == config.gconfig.max_new_tokens
    assert body["n"] == 1 and body["stream"] is False


@pytest.mark.asyncio
async def test_frozen_evaluation_keeps_sglang_top_level_options(tmp_path, monkeypatch):
    """The frozen-model path must not inherit AReaL-specific HTTP nesting."""
    row = {
        "id": "test-0",
        "split": "test",
        "revision": AGENTICK_REVISION,
        "task": "GoToGoal-v0",
        "category": "navigation",
        "difficulty": "easy",
        "instruction": "Reach the goal",
        "feedback_seeds": [10],
        "score_seeds": [20],
    }
    monkeypatch.setattr(frozen, "load_rows", lambda *args: [row])
    monkeypatch.setenv("AGENTICK_SERVICE_URL", "http://game.test")
    monkeypatch.setenv("AGENTICK_SERVICE_KEY", "test-game-key")
    generations = []

    def respond(request):
        body = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            generations.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "completion-1",
                    "choices": [
                        {"message": {"content": "def act(obs, memory): return 0, None"}}
                    ],
                },
            )
        assert request.url.path == "/evaluate"
        return httpx.Response(
            200,
            json={
                "task": body["task"],
                "difficulty": body["difficulty"],
                "seed": body["seed"],
                "success": True,
            },
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        frozen.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(respond), **kwargs
        ),
    )
    result = await frozen.evaluate(
        argparse.Namespace(
            data=tmp_path / "unused.jsonl",
            output=tmp_path / "evaluation.jsonl",
            base_url="http://model.test/v1",
            concurrency=1,
            max_rounds=1,
            max_tokens=128,
            timeout=30,
            evaluation_timeout=10,
            model="test-model",
            temperature=0.6,
        )
    )
    assert result["category_macro_success"] == 1.0
    assert len(generations) == 1
    assert generations[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "extra_body" not in generations[0]

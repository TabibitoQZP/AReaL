# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.trajectory import EncodedPrompt, TrajectoryConverter
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.rpc.serialization import deserialize_value
from areal.v2.inference_service.data_proxy.app import create_app
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import SessionStore


class _RecordingAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, interaction):
        self.calls += 1
        return EncodedPrompt(input_ids=[901, 902])


@pytest.mark.asyncio
async def test_export_trajectories_uses_injected_converter(monkeypatch):
    """The v2 export endpoint converts raw interactions through its adapter."""
    adapter = _RecordingAdapter()
    converter = TrajectoryConverter(adapter)
    config = DataProxyConfig(backend_addr="", tokenizer_path="")
    app = create_app(config, trajectory_converter=converter)

    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    session_id, _ = store.start_session("converter")
    session = store.get_session(session_id)
    assert session is not None
    interaction = InteractionWithTokenLogpReward(
        messages=[{"role": "user", "content": "hello"}],
        output_message_list=[{"role": "assistant", "content": "answer"}],
        model_response=ModelResponse(
            input_tokens=[1],
            output_tokens=[41],
            output_logprobs=[-0.2],
            output_versions=[3],
        ),
    )
    interaction.interaction_id = "interaction-0"
    session.active_completions["interaction-0"] = interaction
    session.set_reward("interaction-0", 2.0)

    app.state.session_store = store
    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        lambda obj, node_addr: obj,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/export_trajectories",
            json={"session_ids": [session_id], "style": "individual"},
            headers={"Authorization": f"Bearer {config.admin_api_key}"},
        )

    assert response.status_code == 200
    trajectory = deserialize_value(response.json()["traj"])
    assert adapter.calls == 1
    torch.testing.assert_close(
        trajectory["input_ids"], torch.tensor([[901, 902, 41]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        trajectory["rewards"], torch.tensor([2.0]), rtol=0, atol=0
    )

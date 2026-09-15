# SPDX-License-Identifier: Apache-2.0
"""CPU checks for coupled resource, sequence-budget and reward settings."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def config(monkeypatch, tmp_path):
    for key in (
        "FILEROOT",
        "AGENTICK_TRAIN_DATA",
        "AREAL_PROXY_ADMIN_KEY",
        "AGENTICK_SERVICE_KEY",
    ):
        monkeypatch.setenv(key, str(tmp_path / key))
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_colocation_and_awex_environment(config):
    """Both DP4 x TP2 roles fit the same eight cards and allow memory remapping."""
    cfg = OmegaConf.to_container(config, resolve=True)
    assert cfg["cluster"]["n_nodes"] == 1 and cfg["cluster"]["n_gpus_per_node"] == 8
    assert cfg["actor"]["backend"] == "megatron:d4p1t2c1e1"
    assert cfg["rollout"]["backend"] == "sglang:d4p1t2"
    assert cfg["rollout"]["scheduling_strategy"] == {
        "type": "colocation",
        "target": "actor",
        "fork": True,
    }
    assert cfg["actor"]["weight_update_mode"] == "awex"
    assert cfg["sglang"]["enable_memory_saver"]
    for spec in cfg["rollout"]["scheduling_spec"]:
        for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            assert spec["env_vars"][key] == "expandable_segments:False"


def test_reward_group_and_turn_contract(config):
    """Normalize once across candidates, then share the outcome across turns."""
    assert config.actor._version == config.rollout._version == "v1"
    assert config.rollout.agent.mode == "inline"
    assert config.gconfig.n_samples == 4
    assert config.gconfig.reward_normalization and config.gconfig.drop_incomplete_group
    assert config.actor.reward_norm is None and config.actor.adv_norm is None
    assert config.actor.reward_scaling == 1 and config.actor.reward_bias == 0
    assert (
        config.rollout.agent.export_style == "individual"
        and config.rollout.agent.turn_discount == 1
    )
    assert config.train_dataset.batch_size == 8 and config.total_train_epochs == 1
    assert (
        config.train_dataset.batch_size * (config.rollout.max_head_offpolicyness + 1)
        == config.rollout.max_concurrent_rollouts
    )


def test_sequence_limits_leave_prompt_space_and_sglang_margin(config):
    """Short-context refinement still needs room for previous code and feedback."""
    assert config.gconfig.max_new_tokens < config.gconfig.max_tokens
    assert config.gconfig.max_tokens < config.sglang.context_length
    assert config.actor.mb_spec.max_tokens_per_mb >= config.gconfig.max_tokens
    assert config.rollout.agent.engine_max_tokens == config.gconfig.max_tokens
    assert config.rollout.agent.reasoning_parser == "qwen3"
    assert config.actor.dtype == config.sglang.dtype == "bfloat16"

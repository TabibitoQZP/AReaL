# SPDX-License-Identifier: Apache-2.0
"""Server-side regressions for the standalone game-policy example defaults."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def game_policy_config(monkeypatch):
    monkeypatch.setenv("AREAL_ROLLOUT_ADMIN_KEY", "test-rollout-key")
    monkeypatch.setenv("AREAL_ACTOR_ADMIN_KEY", "test-actor-key")
    path = Path(__file__).resolve().parents[1] / "examples/game_policy/config.yaml"
    return OmegaConf.load(path)


def test_full_sequence_budget_is_aligned(game_policy_config):
    """Prompt plus completion must fit the actor and both inference limits."""
    cfg = game_policy_config
    assert cfg.gconfig.max_tokens == 32768
    assert cfg.actor.mb_spec.max_tokens_per_mb == cfg.gconfig.max_tokens
    assert cfg.rollout.agent.engine_max_tokens == cfg.gconfig.max_tokens
    assert cfg.sglang.context_length == cfg.gconfig.max_tokens
    # Full-example requests allow 4096 output tokens, plus nonempty input.
    assert cfg.actor.mb_spec.max_tokens_per_mb > 4096


def test_context_override_updates_all_sequence_limits(game_policy_config):
    """A context override must not leave the actor packing limit behind."""
    cfg = game_policy_config
    cfg.gconfig.max_tokens = 16384
    assert cfg.actor.mb_spec.max_tokens_per_mb == 16384
    assert cfg.rollout.agent.engine_max_tokens == 16384
    assert cfg.sglang.context_length == 16384


def test_independent_sampling_flags_are_enabled(game_policy_config):
    """Both proxy seed generation and SGLang seed consumption are necessary."""
    cfg = game_policy_config
    assert cfg.rollout.deterministic_sampling is True
    assert cfg.sglang.enable_deterministic_inference is True
    assert cfg.gconfig.greedy is False
    assert cfg.gconfig.temperature > 0


def test_client_reward_contract_is_unchanged(game_policy_config):
    """Length and sampling fixes must preserve terminal reward propagation."""
    cfg = game_policy_config
    assert cfg.rollout.agent.mode == "online"
    assert cfg.rollout.agent.export_style == "individual"
    assert cfg.rollout.agent.turn_discount == 1
    assert cfg.rollout.agent.set_reward_finish_timeout == 0
    assert cfg.gconfig.n_samples == 1
    assert cfg.gconfig.reward_normalization is False
    assert cfg.actor.reward_norm is None
    assert cfg.actor.adv_norm is None

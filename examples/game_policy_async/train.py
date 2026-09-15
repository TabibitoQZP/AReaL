# SPDX-License-Identifier: Apache-2.0
"""v1 inline GRPO with an external Agentick service and colocated AWEX."""

import math
import os
import sys
from pathlib import Path

from examples.game_policy_async.tasks import load_rows


def validate_colocation(config, actor, rollout) -> None:
    """Check the example's one-GPU-per-worker fork contract before launch."""
    strategy = config.rollout.scheduling_strategy
    if strategy.type != "colocation":
        return
    if strategy.target != "actor" or not strategy.fork:
        raise ValueError("This example requires forked colocation with actor")
    if actor.world_size != rollout.dp_size:
        raise ValueError(
            "Colocation worker count mismatch: actor creates "
            f"{actor.world_size} workers but rollout creates {rollout.dp_size}; "
            "use one TP1 rollout replica per actor rank"
        )
    if rollout.tp_size * rollout.pp_size != 1:
        raise ValueError("Forked single-GPU actor workers require TP1/PP1 rollout")
    if any(
        spec.gpu != 1 or spec.port_count < 2 for spec in config.actor.scheduling_spec
    ):
        raise ValueError("Forked actor workers require gpu=1 and port_count>=2")
    if config.scheduler.type == "local" and (
        config.cluster.n_nodes != 1 or actor.world_size > config.cluster.n_gpus_per_node
    ):
        raise ValueError("Local colocation must fit on one node without GPU reuse")


def validate_prefill(config) -> None:
    """Reject chunk budgets that stall deterministic FlashInfer prefill."""
    sglang = config.sglang
    if (
        not sglang.enable_deterministic_inference
        or sglang.attention_backend != "flashinfer"
    ):
        return
    alignment = int(os.environ.get("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", "4096"))
    if alignment <= 0:
        raise ValueError("FlashInfer prefill split tile size must be positive")
    chunk = sglang.chunked_prefill_size
    if chunk < alignment or chunk % alignment:
        raise ValueError(
            "Deterministic FlashInfer requires chunked_prefill_size to be a "
            f"positive multiple of the prefill split tile size ({alignment}); "
            "smaller chunks can stall long requests, and unchunked prefill "
            "can overflow the attention workspace"
        )
    if sglang.max_prefill_tokens < alignment:
        raise ValueError(
            f"Deterministic FlashInfer requires max_prefill_tokens >= {alignment}"
        )


def proxy_generation_kwargs(gconfig) -> dict:
    """Parameters for raw HTTP to AReaL, not an OpenAI SDK call."""
    return {
        "model": "default",
        "temperature": gconfig.temperature,
        "top_p": gconfig.top_p,
        "max_completion_tokens": gconfig.max_new_tokens,
        # AReaL filters top-level parameters by its create() signature. The
        # template options must reach create(extra_body=...) intact.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def main(args: list[str]) -> None:
    from areal import PPOTrainer
    from areal.api.alloc_mode import ModelAllocation
    from areal.api.cli_args import PPOConfig, load_expr_config

    config, _ = load_expr_config(args, PPOConfig)
    validate_prefill(config)
    validate_colocation(
        config,
        ModelAllocation.from_str(config.actor.backend).parallel,
        ModelAllocation.from_str(config.rollout.backend).parallel,
    )
    if (
        config.actor._version != "v1"
        or config.rollout._version != "v1"
        or config.rollout.agent.mode != "inline"
        or config.gconfig.n_samples < 2
        or not config.gconfig.reward_normalization
        or not config.gconfig.drop_incomplete_group
        or config.actor.reward_norm is not None
        or config.actor.adv_norm is not None
        or config.rollout.agent.export_style != "individual"
        or config.rollout.agent.turn_discount != 1.0
    ):
        raise ValueError(
            "Require v1 inline, complete GRPO groups and one rollout normalization"
        )
    service_url = os.environ["AGENTICK_SERVICE_URL"]
    if not os.environ.get("AGENTICK_SERVICE_KEY"):
        raise ValueError("Set AGENTICK_SERVICE_KEY on trainer and rollout workers")
    train_data = load_rows(Path(config.train_dataset.path), "train")
    valid_data = (
        load_rows(Path(config.valid_dataset.path), "test")
        if config.valid_dataset is not None
        else None
    )
    if len(train_data) % config.train_dataset.batch_size:
        raise ValueError("Training row count must be divisible by the group batch size")
    workflow_kwargs = {
        "service_url": service_url,
        "max_rounds": int(os.environ.get("AGENTICK_MAX_ROUNDS", "4")),
        "timeout": float(os.environ.get("AGENTICK_REQUEST_TIMEOUT", "7500")),
        "log_dir": str(
            Path(config.cluster.fileroot)
            / config.experiment_name
            / config.trial_name
            / "agentick"
        ),
        "evaluation_timeout": float(
            os.environ.get("AGENTICK_EVALUATION_TIMEOUT", "600")
        ),
        **proxy_generation_kwargs(config.gconfig),
    }
    if not 1 <= workflow_kwargs["max_rounds"] <= 16:
        raise ValueError("Require 1 <= AGENTICK_MAX_ROUNDS <= 16")
    timeout = workflow_kwargs["timeout"]
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("AGENTICK_REQUEST_TIMEOUT must be finite and positive")
    with PPOTrainer(
        config, train_dataset=train_data, valid_dataset=valid_data
    ) as trainer:
        trainer.train(
            workflow="examples.game_policy_async.agent.AgentickAgent",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.game_policy_async.agent.AgentickAgent",
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])

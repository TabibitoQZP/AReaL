# SPDX-License-Identifier: Apache-2.0
"""v1 inline GRPO with an external Agentick service and colocated AWEX."""

import math
import os
import sys
from pathlib import Path

from examples.game_policy_async.tasks import load_rows

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, PPOConfig)
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
        "model": "default",
        "temperature": config.gconfig.temperature,
        "top_p": config.gconfig.top_p,
        "max_completion_tokens": config.gconfig.max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
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

# SPDX-License-Identifier: Apache-2.0
"""Thin training/inference service; all task logic runs in the HTTP client."""

import sys

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, PPOConfig)
    if config.rollout.agent.mode != "online" or config.gconfig.n_samples != 1:
        raise ValueError("This service requires online mode and gconfig.n_samples=1")
    with PPOTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import allocate_balanced_mbs
from areal.workflow.openai.vision_geometry3k_agent import _fill_image_urls

ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "examples/vlm/qwen3_5_4b_geometry3k_grpo_v2.yaml"


def test_qwen3_5_example_uses_singleton_training_microbatches():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    actor = config["actor"]

    assert config["total_train_epochs"] == 1
    assert config["cluster"]["n_gpus_per_node"] == 8
    assert config["rollout"]["backend"] == "sglang:d4p1t1"
    assert actor["backend"] == "fsdp:d4p1t1"

    actor_allocation = ModelAllocation.from_str(actor["backend"])
    actor_dp = actor_allocation.parallel.data_parallel_size
    global_batch_size = (
        config["train_dataset"]["batch_size"] * config["gconfig"]["n_samples"]
    )
    local_batch_size = global_batch_size // actor_dp
    ppo_minibatch_size = local_batch_size // actor["ppo_n_minibatches"]

    assert global_batch_size % actor_dp == 0
    assert local_batch_size % actor["ppo_n_minibatches"] == 0
    assert actor["mb_spec"]["n_mbs"] == ppo_minibatch_size

    groups = allocate_balanced_mbs(
        MicroBatchSpec(**actor["mb_spec"]), [1024] * ppo_minibatch_size
    )

    assert len(groups) == ppo_minibatch_size
    assert all(len(group) == 1 for group in groups)


def test_fill_image_urls_preserves_messages_and_fills_placeholder():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Solve the problem."},
                {"type": "image_url", "image_url": {"url": ""}},
            ],
        }
    ]

    filled = _fill_image_urls(messages, ["encoded-image"])

    assert messages[0]["content"][1]["image_url"]["url"] == ""
    assert (
        filled[0]["content"][1]["image_url"]["url"]
        == "data:image/png;base64,encoded-image"
    )

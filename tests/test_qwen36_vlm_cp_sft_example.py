# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from examples.vlm.qwen3_6_35b_a3b_megatron_vlm_cp_sft import (
    validate_single_sequence_microbatches,
)


def test_validate_single_sequence_microbatches_accepts_one_sequence_per_forward():
    config = SimpleNamespace(
        actor=SimpleNamespace(mb_spec=SimpleNamespace(n_mbs=8)),
        train_dataset=SimpleNamespace(batch_size=8),
    )

    validate_single_sequence_microbatches(config)


def test_validate_single_sequence_microbatches_rejects_packed_forward():
    config = SimpleNamespace(
        actor=SimpleNamespace(mb_spec=SimpleNamespace(n_mbs=4)),
        train_dataset=SimpleNamespace(batch_size=8),
    )

    with pytest.raises(ValueError, match="one sequence"):
        validate_single_sequence_microbatches(config)

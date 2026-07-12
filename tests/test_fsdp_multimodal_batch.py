# SPDX-License-Identifier: Apache-2.0

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import (
    _flatten_padded_seq_values,
    _prepare_multimodal_forward_inputs,
    _prepare_padded_seq_mb_list,
)
from areal.utils.data import split_padded_tensor_dict_into_mb_list


def test_multimodal_forward_inputs_are_not_kept_in_loss_mb():
    pixel_values = [torch.randn(2, 4), torch.randn(3, 4)]
    image_grid_thw = [torch.tensor([[1, 1, 2]]), torch.tensor([[1, 1, 3]])]
    mb = {
        "input_ids": torch.ones(5, dtype=torch.long),
        "multi_modal_input": [
            {
                "pixel_values": pixel_values[0],
                "image_grid_thw": image_grid_thw[0],
            },
            {
                "pixel_values": pixel_values[1],
                "image_grid_thw": image_grid_thw[1],
            },
        ],
    }
    padded_mb = dict(mb)

    _prepare_multimodal_forward_inputs(mb, padded_mb)

    assert "multi_modal_input" not in mb
    assert "multi_modal_input" not in padded_mb
    assert "pixel_values" not in mb
    assert "image_grid_thw" not in mb
    assert torch.equal(padded_mb["pixel_values"], torch.cat(pixel_values, dim=0))
    assert torch.equal(padded_mb["image_grid_thw"], torch.cat(image_grid_thw, dim=0))


def test_multimodal_forward_inputs_fall_back_to_padded_mb():
    pixel_values = [torch.randn(2, 4)]
    mb = {"input_ids": torch.ones(2, dtype=torch.long)}
    padded_mb = {
        "input_ids": torch.ones(2, dtype=torch.long),
        "multi_modal_input": [{"pixel_values": pixel_values[0]}],
    }

    _prepare_multimodal_forward_inputs(mb, padded_mb)

    assert "multi_modal_input" not in mb
    assert "multi_modal_input" not in padded_mb
    assert "pixel_values" not in mb
    assert torch.equal(padded_mb["pixel_values"], torch.cat(pixel_values, dim=0))


def test_padded_sequence_microbatches_preserve_batch_dimension():
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ]
    )
    input_ids = torch.arange(20).reshape(4, 5)
    data = {
        "input_ids": input_ids,
        "position_ids": torch.arange(5).expand(4, -1),
        "attention_mask": attention_mask,
        "loss_mask": attention_mask.clone(),
    }
    mb_list = split_padded_tensor_dict_into_mb_list(
        data,
        MicroBatchSpec(n_mbs=1, max_tokens_per_mb=8),
    )

    mb_list = _prepare_padded_seq_mb_list(mb_list)

    assert len(mb_list) == 2
    assert mb_list.padding_lengths == [0, 0]
    for packed_mb, padded_mb in zip(mb_list.mbs, mb_list.padded_mbs, strict=True):
        assert padded_mb["input_ids"].ndim == 2
        assert padded_mb["input_ids"].shape[0] == 2
        assert padded_mb["attention_mask"].shape == padded_mb["input_ids"].shape
        expected_ids = padded_mb["input_ids"][padded_mb["attention_mask"].bool()]
        torch.testing.assert_close(packed_mb["input_ids"], expected_ids, rtol=0, atol=0)


def test_flatten_padded_sequence_values_uses_attention_mask_order():
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    flattened = _flatten_padded_seq_values(values, attention_mask)

    torch.testing.assert_close(flattened, torch.tensor([1.0, 2.0, 4.0]), rtol=0, atol=0)

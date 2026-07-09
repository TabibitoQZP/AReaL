# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.engine.fsdp_engine import (
    _get_qwen_vl_position_model,
    _prepare_multimodal_forward_inputs,
)


class _Wrapper(torch.nn.Module):
    def __init__(
        self,
        *,
        model: torch.nn.Module | None = None,
        base_model: torch.nn.Module | None = None,
    ):
        super().__init__()
        if model is not None:
            self.model = model
        if base_model is not None:
            self.base_model = base_model


class _QwenVlCore(torch.nn.Module):
    def compute_3d_position_ids(self):
        return None


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


def test_get_qwen_vl_position_model_unwraps_nested_model():
    core = _QwenVlCore()
    wrapped = _Wrapper(model=_Wrapper(base_model=_Wrapper(model=core)))

    assert _get_qwen_vl_position_model(wrapped) is core


def test_get_qwen_vl_position_model_raises_for_missing_helper():
    wrapped = _Wrapper(model=_Wrapper())

    with pytest.raises(AttributeError, match="compute_3d_position_ids"):
        _get_qwen_vl_position_model(wrapped)

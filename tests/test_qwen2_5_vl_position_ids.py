# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.models.transformers.qwen2_vl import (
    correct_qwen2_5_vl_image_position_ids,
)


def test_correct_image_position_ids_with_scaled_temporal_origin_fixes_image():
    """Affected Transformers positions should use the unscaled image origin."""
    position_ids = torch.tensor(
        [
            [[0, 1, 4, 4, 4, 4, 4]],
            [[0, 1, 2, 2, 3, 3, 4]],
            [[0, 1, 2, 3, 2, 3, 4]],
        ]
    )
    token_types = torch.tensor([[0, 0, 1, 1, 1, 1, 0]])
    grid_thw = torch.tensor([[1, 4, 4]])

    corrected = correct_qwen2_5_vl_image_position_ids(
        position_ids,
        token_types,
        grid_thw,
        spatial_merge_size=2,
    )

    torch.testing.assert_close(
        corrected[0],
        torch.tensor([[0, 1, 2, 2, 2, 2, 4]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        corrected[1:],
        position_ids[1:],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        position_ids[0],
        torch.tensor([[0, 1, 4, 4, 4, 4, 4]]),
        rtol=0,
        atol=0,
    )


def test_correct_image_position_ids_with_fixed_layout_is_noop():
    """The compatibility fix should be inert after upgrading Transformers."""
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 4]],
            [[0, 1, 2, 2, 3, 3, 4]],
            [[0, 1, 2, 3, 2, 3, 4]],
        ]
    )
    token_types = torch.tensor([[0, 0, 1, 1, 1, 1, 0]])

    corrected = correct_qwen2_5_vl_image_position_ids(
        position_ids,
        token_types,
        torch.tensor([[1, 4, 4]]),
        spatial_merge_size=2,
    )

    assert corrected is position_ids


def test_correct_image_position_ids_with_mismatched_grid_raises():
    """Malformed multimodal metadata should fail before model forward."""
    position_ids = torch.zeros(3, 1, 5, dtype=torch.long)
    token_types = torch.tensor([[0, 1, 1, 1, 0]])

    with pytest.raises(ValueError, match="does not match image_grid_thw"):
        correct_qwen2_5_vl_image_position_ids(
            position_ids,
            token_types,
            torch.tensor([[1, 4, 4]]),
            spatial_merge_size=2,
        )

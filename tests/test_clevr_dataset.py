# SPDX-License-Identifier: Apache-2.0

import torch
from PIL import Image

from areal.dataset.clevr_count_70k import (
    _build_completion_loss_mask,
    _decode_images,
    _get_mm_token_type_ids,
    _limit_dataset,
    convert_image,
)


def test_mm_token_type_ids_prefers_qwen_multimodal_key():
    processed_input = {
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
        "token_type_ids": torch.tensor([[9, 9, 9, 9]]),
    }

    actual = _get_mm_token_type_ids(processed_input)

    torch.testing.assert_close(actual, torch.tensor([0, 1, 1, 0]))


def test_mm_token_type_ids_supports_legacy_key_and_absence():
    legacy = _get_mm_token_type_ids({"token_type_ids": torch.tensor([[0, 1]])})

    torch.testing.assert_close(legacy, torch.tensor([0, 1]))
    assert _get_mm_token_type_ids({}) is None


def test_decode_images_restores_rgb_pil_image():
    source = Image.new("L", (2, 3), color=127)
    decoded = _decode_images([convert_image(source, max_pixels=None)])

    assert len(decoded) == 1
    assert decoded[0].mode == "RGB"
    assert decoded[0].size == (2, 3)


def test_decode_images_accepts_dataset_decoded_pil_image():
    decoded = _decode_images([Image.new("L", (2, 3), color=127)])

    assert decoded[0].mode == "RGB"
    assert decoded[0].size == (2, 3)


def test_completion_mask_includes_eos_token():
    class Tokenizer:
        eos_token = "<eos>"

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            assert text == "42<eos>"
            return [1, 2, 3]

    assert _build_completion_loss_mask(8, "42", Tokenizer()) == [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
    ]


def test_limit_dataset_selects_before_preprocessing():
    class Dataset:
        def __init__(self, size):
            self.size = size
            self.selected = None

        def __len__(self):
            return self.size

        def select(self, indices):
            self.selected = list(indices)
            return self

    dataset = Dataset(size=10)

    assert _limit_dataset(dataset, 3).selected == [0, 1, 2]
    assert _limit_dataset(dataset, 0) is dataset

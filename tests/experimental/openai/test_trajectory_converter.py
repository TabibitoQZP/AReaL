# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
from typing import Any

import pytest
import torch
from PIL import Image

from areal.api import ModelResponse
from areal.experimental.openai.trajectory import (
    DEFAULT_TRAJECTORY_CONVERTER,
    LLMInputAdapter,
    PlaceholderVLMInputAdapter,
    TrajectoryConverter,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils.data import concat_padded_tensors


def _image_data_uri() -> str:
    image = Image.new("RGB", (2, 2), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _interaction(
    *,
    messages: list[dict[str, Any]] | None = None,
    input_tokens: list[int] | None = None,
    output_tokens: list[int] | None = None,
) -> InteractionWithTokenLogpReward:
    output_tokens = output_tokens or [31, 32]
    return InteractionWithTokenLogpReward(
        messages=messages or [{"role": "user", "content": "hello"}],
        model_response=ModelResponse(
            input_tokens=input_tokens or [11, 12, 13],
            output_tokens=output_tokens,
            output_logprobs=[-0.25, -0.5],
            output_versions=[7, 8],
        ),
        reward=1.5,
    )


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        self.calls.append(
            {"messages": messages, "tokenize": tokenize, "kwargs": kwargs}
        )
        assert tokenize is False
        return "rendered prompt"


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "images" in kwargs:
            return {
                "input_ids": torch.tensor([[101, 102, 103]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
                "pixel_values": torch.ones((4, 8), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        return {
            "input_ids": torch.tensor([[201, 202]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }


def test_default_converter_preserves_llm_trajectory_schema():
    """Default conversion keeps the established LLM token alignment and dtypes."""
    interaction = _interaction()

    result = interaction.to_tensor_dict()

    expected = {
        "input_ids": torch.tensor([[11, 12, 13, 31, 32]]),
        "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
        "logprobs": torch.tensor([[0.0, 0.0, 0.0, -0.25, -0.5]]),
        "versions": torch.tensor([[-1, -1, -1, 7, 8]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        "rewards": torch.tensor([1.5]),
    }
    assert result.keys() == expected.keys()
    assert interaction._cache is result
    for key, expected_value in expected.items():
        torch.testing.assert_close(result[key], expected_value, rtol=0, atol=0)


def test_llm_adapter_rejects_image_messages():
    """Image interactions cannot silently fall back to rollout prompt IDs."""
    interaction = _interaction(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_uri()},
                    },
                ],
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Multimodal interaction"):
        LLMInputAdapter().encode(interaction)


def test_vlm_converter_exports_processor_inputs_and_request_context():
    """VLM conversion rebuilds the prompt and preserves processor model inputs."""
    processor = _FakeProcessor()
    converter = TrajectoryConverter(
        PlaceholderVLMInputAdapter(processor),
        allow_concat=False,
        include_model_inputs=True,
    )
    interaction = _interaction(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_uri()},
                    },
                ],
            }
        ]
    )
    interaction.tools = [{"type": "function", "function": {"name": "look"}}]
    interaction.chat_template_kwargs = {"enable_thinking": False}

    result = converter.convert(interaction)

    torch.testing.assert_close(
        result["input_ids"], torch.tensor([[101, 102, 103, 31, 32]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        result["mm_token_type_ids"],
        torch.tensor([[0, 1, 0, 0, 0]]),
        rtol=0,
        atol=0,
    )
    assert len(result["multi_modal_input"]) == 1
    model_inputs = result["multi_modal_input"][0]
    assert set(model_inputs) == {"pixel_values", "image_grid_thw"}
    assert "images" in processor.calls[0]
    assert processor.tokenizer.calls[0]["messages"][0]["content"][1] == {
        "type": "image"
    }
    assert processor.tokenizer.calls[0]["kwargs"]["tools"] == interaction.tools
    assert processor.tokenizer.calls[0]["kwargs"]["enable_thinking"] is False


def test_vlm_converter_keeps_text_only_samples_batch_compatible():
    """Text-only VLM samples retain the same keys as image samples."""
    processor = _FakeProcessor()
    converter = TrajectoryConverter(
        PlaceholderVLMInputAdapter(processor),
        allow_concat=False,
        include_model_inputs=True,
    )
    image_interaction = _interaction(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_uri()},
                    },
                ],
            }
        ]
    )
    text_interaction = _interaction()

    image_result = converter.convert(image_interaction)
    text_result = converter.convert(text_interaction)
    batched = concat_padded_tensors([image_result, text_result])

    assert image_result.keys() == text_result.keys()
    assert text_result["multi_modal_input"] == [{}]
    assert len(batched["multi_modal_input"]) == 2
    assert batched["multi_modal_input"][1] == {}
    torch.testing.assert_close(
        text_result["mm_token_type_ids"],
        torch.zeros((1, 4), dtype=torch.long),
        rtol=0,
        atol=0,
    )


def test_vlm_converter_rejects_parent_concat():
    """VLM concat fails closed until parent token provenance is explicit."""
    converter = TrajectoryConverter(
        PlaceholderVLMInputAdapter(_FakeProcessor()),
        allow_concat=False,
        include_model_inputs=True,
    )
    parent = _interaction()
    child = _interaction(input_tokens=[11, 12, 13, 31, 32, 41])
    child.chat_template_type = "concat"
    child.parent = parent

    with pytest.raises(RuntimeError, match="Multimodal concat export"):
        converter.convert(child)


def test_converter_rejects_mismatched_rollout_fields():
    """Malformed rollout metadata is rejected before tensor packing."""
    interaction = _interaction()
    assert interaction.model_response is not None
    interaction.model_response.output_versions = [7]

    with pytest.raises(RuntimeError, match="equal lengths"):
        DEFAULT_TRAJECTORY_CONVERTER.convert(interaction)

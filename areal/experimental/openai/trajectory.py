# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch

from areal.experimental.openai.message_utils import (
    extract_images_from_messages,
    iter_image_urls,
    parse_tool_call_arguments,
)
from areal.utils import logging
from areal.utils.hf_utils import apply_chat_template

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

logger = logging.getLogger("OpenAIClient")

_PROCESSOR_SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "mm_token_type_ids",
    "position_ids",
}


@dataclass
class EncodedPrompt:
    """Model-specific prompt encoding before generated tokens are appended."""

    input_ids: list[int]
    sequence_inputs: dict[str, list[int]] = field(default_factory=dict)
    model_inputs: dict[str, Any] = field(default_factory=dict)


class TrajectoryInputAdapter(Protocol):
    """Convert an interaction prompt into trainer-side model inputs."""

    def encode(self, interaction: InteractionWithTokenLogpReward) -> EncodedPrompt: ...


def _first_sequence(processed: dict[str, Any], key: str) -> list[int] | None:
    value = processed.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim == 2:
            value = value[0]
        if value.ndim != 1:
            raise ValueError(f"Processor field {key!r} must be a 1D sequence.")
        return [int(x) for x in value.tolist()]
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(x) for x in value]
    raise TypeError(f"Unsupported processor field type for {key!r}: {type(value)}")


class LLMInputAdapter:
    """Use the exact prompt token IDs recorded by the rollout engine."""

    def encode(self, interaction: InteractionWithTokenLogpReward) -> EncodedPrompt:
        if iter_image_urls(interaction.messages):
            raise RuntimeError(
                "Multimodal interaction requires a configured trajectory input adapter."
            )
        response = interaction.model_response
        assert response is not None, "Model response is not set."
        return EncodedPrompt(input_ids=list(response.input_tokens))


class PlaceholderVLMInputAdapter:
    """Build placeholder-token VLM inputs with a Hugging Face processor."""

    def __init__(self, processor: Any, image_timeout: float | None = None):
        self.processor = processor
        self.image_timeout = image_timeout

    def encode(self, interaction: InteractionWithTokenLogpReward) -> EncodedPrompt:
        from transformers.image_utils import load_image

        _, tokenizer_messages, _ = extract_images_from_messages(interaction.messages)
        tokenizer_messages = parse_tool_call_arguments(tokenizer_messages)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("VLM trajectory adapter requires processor.tokenizer.")

        prompt = apply_chat_template(
            tokenizer,
            tokenizer_messages,
            tools=interaction.tools,
            add_generation_prompt=True,
            tokenize=False,
            **interaction.chat_template_kwargs,
        )

        image_urls = iter_image_urls(interaction.messages)
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "padding": False,
            "return_tensors": "pt",
        }
        if image_urls:
            processor_kwargs["images"] = [
                load_image(url, timeout=self.image_timeout) for url in image_urls
            ]

        processed = self.processor(**processor_kwargs)
        input_ids = _first_sequence(processed, "input_ids")
        if input_ids is None:
            raise RuntimeError("VLM processor output must contain input_ids.")

        sequence_inputs: dict[str, list[int]] = {}
        mm_token_type_ids = _first_sequence(processed, "mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = _first_sequence(processed, "token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = [0] * len(input_ids)
        if len(mm_token_type_ids) != len(input_ids):
            raise RuntimeError(
                "Processor token type IDs do not match processor input length."
            )
        sequence_inputs["mm_token_type_ids"] = mm_token_type_ids

        model_inputs = {
            key: value
            for key, value in processed.items()
            if key not in _PROCESSOR_SEQUENCE_KEYS
        }
        if image_urls and "pixel_values" not in model_inputs:
            raise RuntimeError(
                "VLM processor output must contain pixel_values for image inputs."
            )

        return EncodedPrompt(
            input_ids=input_ids,
            sequence_inputs=sequence_inputs,
            model_inputs=model_inputs,
        )


class TrajectoryConverter:
    """Pack model-specific prompt inputs into the common RL trajectory schema."""

    def __init__(
        self,
        adapter: TrajectoryInputAdapter,
        *,
        allow_concat: bool = True,
        include_model_inputs: bool = False,
    ):
        self.adapter = adapter
        self.allow_concat = allow_concat
        self.include_model_inputs = include_model_inputs

    def convert(self, interaction: InteractionWithTokenLogpReward) -> dict[str, Any]:
        response = interaction.model_response
        assert response is not None, "Model response is not set."
        if not (
            response.output_len
            == len(response.output_logprobs)
            == len(response.output_versions)
        ):
            raise RuntimeError(
                "Output token IDs, logprobs, and versions must have equal lengths."
            )

        if (
            interaction.chat_template_type == "concat"
            and interaction.parent is not None
            and not self.allow_concat
        ):
            raise RuntimeError(
                "Multimodal concat export is not supported yet because re-encoding "
                "the full prompt cannot safely recover parent token provenance."
            )

        encoded = self.adapter.encode(interaction)
        prompt_ids = encoded.input_ids
        prompt_len = len(prompt_ids)
        sequence = prompt_ids + response.output_tokens
        interaction.seq_tokens = sequence

        if (
            interaction.chat_template_type == "concat"
            and interaction.parent is not None
        ):
            parent_result = self.convert(interaction.parent)
            parent_ids = parent_result["input_ids"].squeeze(0).tolist()
            parent_logprobs = parent_result["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_result["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_result["versions"].squeeze(0).tolist()
            parent_len = len(parent_ids)

            if prompt_len > parent_len:
                logprobs = (
                    parent_logprobs
                    + [0.0] * (prompt_len - parent_len)
                    + response.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (prompt_len - parent_len)
                    + [1] * response.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (prompt_len - parent_len)
                    + response.output_versions
                )
            else:
                self._log_invalid_parent(prompt_ids, parent_len)
                logprobs, loss_mask, versions = self._individual_fields(
                    prompt_len, response
                )
        else:
            logprobs, loss_mask, versions = self._individual_fields(
                prompt_len, response
            )

        reward = interaction.reward if interaction.reward is not None else 0.0
        result: dict[str, Any] = {
            "input_ids": torch.tensor(sequence, dtype=torch.long).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones(len(sequence), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([float(reward)], dtype=torch.float32),
        }

        for key, values in encoded.sequence_inputs.items():
            if len(values) != prompt_len:
                raise RuntimeError(
                    f"Sequence input {key!r} does not match prompt length."
                )
            result[key] = torch.tensor(
                values + [0] * response.output_len,
                dtype=torch.long,
            ).unsqueeze(0)

        if self.include_model_inputs:
            result["multi_modal_input"] = [encoded.model_inputs]

        return result

    @staticmethod
    def _individual_fields(prompt_len: int, response: Any):
        return (
            [0.0] * prompt_len + response.output_logprobs,
            [0] * prompt_len + [1] * response.output_len,
            [-1] * prompt_len + response.output_versions,
        )

    @staticmethod
    def _log_invalid_parent(prompt_ids: list[int], parent_len: int) -> None:
        logger.warning(
            "The child prompt length (%d) is less than or equal to the parent "
            "trajectory length (%d); masking the parent as prompt context. "
            "Child prompt token IDs: %s",
            len(prompt_ids),
            parent_len,
            prompt_ids,
        )


DEFAULT_TRAJECTORY_CONVERTER = TrajectoryConverter(LLMInputAdapter())


def build_trajectory_converter(model_path: str) -> TrajectoryConverter:
    """Build the default converter for a local Hugging Face model path."""
    from transformers import AutoProcessor

    if not model_path:
        return DEFAULT_TRAJECTORY_CONVERTER

    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception:
        logger.warning(
            "Failed to load a multimodal processor from %s; using the LLM "
            "trajectory converter.",
            model_path,
            exc_info=True,
        )
        return DEFAULT_TRAJECTORY_CONVERTER

    if getattr(processor, "image_processor", None) is None:
        return DEFAULT_TRAJECTORY_CONVERTER

    return TrajectoryConverter(
        PlaceholderVLMInputAdapter(processor),
        allow_concat=False,
        include_model_inputs=True,
    )

# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.utils import logging
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

logger = logging.getLogger("Dataset")


def validate_single_sequence_microbatches(config: SFTConfig) -> None:
    """Keep this CP example on the one-sequence-per-forward validation path."""
    batch_size = config.train_dataset.batch_size
    n_mbs = config.actor.mb_spec.n_mbs
    if n_mbs != batch_size:
        raise ValueError(
            "This Qwen3.6 VLM CP SFT example requires "
            "actor.mb_spec.n_mbs to equal train_dataset.batch_size so each "
            "Megatron micro-batch contains exactly one sequence; got "
            f"n_mbs={n_mbs}, batch_size={batch_size}."
        )


class SyntheticLongVLMDataset(Dataset):
    """Deterministic image-text SFT data for CP consistency validation."""

    _COLORS = (
        ("red", (210, 50, 50)),
        ("green", (45, 165, 85)),
        ("blue", (55, 95, 210)),
        ("purple", (145, 70, 190)),
    )
    _SHAPES = ("square", "circle", "triangle", "cross")

    def __init__(
        self,
        processor: Any,
        tokenizer: Any,
        *,
        num_samples: int,
        context_repeats: int,
        answer_repeats: int,
        image_size: int,
        include_image: bool,
        distinct_samples: bool,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if min(context_repeats, answer_repeats, image_size) <= 0:
            raise ValueError(
                "context_repeats, answer_repeats, and image_size must be positive"
            )

        image_processor_type = processor.image_processor.image_processor_type.lower()
        if include_image and "qwen" not in image_processor_type:
            raise ValueError(
                "This consistency dataset currently supports Qwen image processors, "
                f"got {processor.image_processor.image_processor_type!r}."
            )

        self.samples = [
            self._build_sample(
                processor,
                tokenizer,
                index=index if distinct_samples else 0,
                context_repeats=context_repeats,
                answer_repeats=answer_repeats,
                image_size=image_size,
                include_image=include_image,
            )
            for index in range(num_samples)
        ]

        lengths = [int(sample["input_ids"].shape[0]) for sample in self.samples]
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_tokens = [
            int((sample["input_ids"] == image_token_id).count_nonzero().item())
            for sample in self.samples
        ]
        logger.info(
            "Prepared %d deterministic VLM samples: sequence length=%d..%d, "
            "image tokens=%d..%d, image size=%dx%d.",
            num_samples,
            min(lengths),
            max(lengths),
            min(image_tokens),
            max(image_tokens),
            image_size,
            image_size,
        )

    @classmethod
    def _build_image(cls, index: int, image_size: int) -> Image.Image:
        _, color = cls._COLORS[index % len(cls._COLORS)]
        shape = cls._SHAPES[(index // len(cls._COLORS)) % len(cls._SHAPES)]
        image = Image.new("RGB", (image_size, image_size), color=(225, 225, 225))
        draw = ImageDraw.Draw(image)
        margin = image_size // 4
        box = (margin, margin, image_size - margin, image_size - margin)

        if shape == "square":
            draw.rectangle(box, fill=color)
        elif shape == "circle":
            draw.ellipse(box, fill=color)
        elif shape == "triangle":
            draw.polygon(
                (
                    (image_size // 2, margin),
                    (image_size - margin, image_size - margin),
                    (margin, image_size - margin),
                ),
                fill=color,
            )
        else:
            width = max(8, image_size // 10)
            center = image_size // 2
            draw.rectangle(
                (center - width, margin, center + width, image_size - margin),
                fill=color,
            )
            draw.rectangle(
                (margin, center - width, image_size - margin, center + width),
                fill=color,
            )
        return image

    @classmethod
    def _build_sample(
        cls,
        processor: Any,
        tokenizer: Any,
        *,
        index: int,
        context_repeats: int,
        answer_repeats: int,
        image_size: int,
        include_image: bool,
    ) -> dict[str, Any]:
        color_name, _ = cls._COLORS[index % len(cls._COLORS)]
        shape_name = cls._SHAPES[(index // len(cls._COLORS)) % len(cls._SHAPES)]
        image = cls._build_image(index, image_size) if include_image else None

        context_unit = (
            "This evidence record belongs to a controlled visual audit. "
            "Keep every numbered statement in context, but use the image as the "
            "only authoritative source for its central shape and color. "
        )
        context = "".join(
            f"Evidence {record:03d}: {context_unit}"
            for record in range(context_repeats)
        )
        image_prefix = (
            "<|vision_start|><|image_pad|><|vision_end|>\n" if include_image else ""
        )
        prefix = (
            f"{image_prefix}Inspect the synthetic record and produce the requested "
            f"verification log.\n{context}\n"
        )
        answer_line = (
            f"The central object is a {color_name} {shape_name}; this conclusion "
            "comes from the image rather than the repeated context. "
        )
        target = "Answer:\n" + "".join(
            f"Verification {record:03d}: {answer_line}"
            for record in range(answer_repeats)
        )
        target += tokenizer.eos_token

        processor_kwargs: dict[str, Any] = {
            "text": [prefix + target],
            "padding": False,
            "return_tensors": "pt",
            "return_attention_mask": False,
        }
        if image is not None:
            processor_kwargs["images"] = [image]
        processed = processor(**processor_kwargs)
        input_ids = processed["input_ids"].squeeze(0).to(torch.long)
        mm_token_type_ids = processed.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0).to(torch.long)
            if mm_token_type_ids.shape != input_ids.shape:
                raise ValueError(
                    "mm_token_type_ids must have the same shape as input_ids"
                )
        target_tokens = tokenizer.encode(target, add_special_tokens=False)
        if not target_tokens or len(target_tokens) >= input_ids.shape[0]:
            raise ValueError(
                "Synthetic target must occupy a non-empty suffix of input_ids"
            )
        target_ids = torch.tensor(target_tokens, dtype=torch.long)
        if not torch.equal(input_ids[-len(target_tokens) :].cpu(), target_ids):
            raise ValueError(
                "Synthetic target tokenization is not an exact input_ids suffix"
            )

        loss_mask = torch.zeros(input_ids.shape[0], dtype=torch.long)
        loss_mask[-len(target_tokens) :] = 1
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_token_count = int((input_ids == image_token_id).count_nonzero().item())
        if include_image and image_token_count == 0:
            raise ValueError(
                "Processor output does not contain image placeholder tokens"
            )
        if include_image and mm_token_type_ids is None:
            raise ValueError("Processor output does not contain mm_token_type_ids")
        if (
            include_image
            and int((mm_token_type_ids == 1).count_nonzero()) != image_token_count
        ):
            raise ValueError(
                "Image token count does not match multimodal token type IDs"
            )

        sample: dict[str, Any] = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }
        if mm_token_type_ids is not None:
            sample["mm_token_type_ids"] = mm_token_type_ids
        if include_image:
            sample["multi_modal_input"] = [
                {
                    "pixel_values": processed["pixel_values"],
                    "image_grid_thw": processed["image_grid_thw"],
                }
            ]
        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, SFTConfig)
    validate_single_sequence_microbatches(config)
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = SyntheticLongVLMDataset(
        processor,
        tokenizer,
        num_samples=int(os.getenv("VLM_CP_NUM_SAMPLES", "64")),
        context_repeats=int(os.getenv("VLM_CP_CONTEXT_REPEATS", "40")),
        answer_repeats=int(os.getenv("VLM_CP_ANSWER_REPEATS", "24")),
        image_size=int(os.getenv("VLM_CP_IMAGE_SIZE", "336")),
        include_image=os.getenv("VLM_CP_INCLUDE_IMAGE", "1") == "1",
        distinct_samples=os.getenv("VLM_CP_DISTINCT_SAMPLES", "0") == "1",
    )
    if len(train_dataset) < config.train_dataset.batch_size:
        raise ValueError(
            "Synthetic dataset must contain at least one global batch: "
            f"samples={len(train_dataset)}, batch_size={config.train_dataset.batch_size}."
        )

    with SFTTrainer(config, train_dataset=train_dataset) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])

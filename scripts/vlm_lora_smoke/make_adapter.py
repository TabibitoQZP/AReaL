# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model


def _load_model(model_path: str):
    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a random PEFT LoRA adapter for the VLM LoRA smoke test."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated PEFT LoRA target modules.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = output_dir / "adapter_config.json"
    adapter_weights = output_dir / "adapter_model.safetensors"
    if adapter_config.exists() and adapter_weights.exists():
        print(f"adapter already exists at {output_dir}; skipping")
        return

    target_modules = [
        module.strip() for module in args.target_modules.split(",") if module.strip()
    ]
    if not target_modules:
        raise ValueError("--target-modules must not be empty")

    print("loading base model on CPU to create PEFT adapter...")
    model = _load_model(args.model_path)
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"saved adapter to {output_dir}")


if __name__ == "__main__":
    main()

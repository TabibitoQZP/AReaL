# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for AReaL's embedded SGLang server."""

from __future__ import annotations

from collections.abc import Iterable
from types import MethodType
from typing import Any

_TEXT_CONFIG_ATTRS = (
    "vocab_size",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "tie_word_embeddings",
)


def _has_attr(obj: Any, name: str) -> bool:
    try:
        getattr(obj, name)
    except AttributeError:
        return False
    return True


def promote_text_config_attrs(
    hf_config: Any,
    attrs: Iterable[str] = _TEXT_CONFIG_ATTRS,
) -> Any:
    """Expose text-model config attrs expected by SGLang LoRA on composite VLMs.

    Some SGLang paths expect an LLM-like config where LoRA metadata such as
    ``vocab_size`` and ``num_hidden_layers`` live at the top level. Newer VLM
    configs can keep those fields under ``text_config`` instead. Promote only
    missing top-level attrs so existing SGLang-normalized configs remain
    untouched.
    """

    text_config = getattr(hf_config, "text_config", None)
    if text_config is None:
        return hf_config

    for attr in attrs:
        if _has_attr(hf_config, attr) or not _has_attr(text_config, attr):
            continue
        setattr(hf_config, attr, getattr(text_config, attr))
    return hf_config


_LORA_MODULE_ATTR_PATHS = {
    "qkv_proj": (("qkv_proj",), ("self_attn", "qkv_proj")),
    "o_proj": (("o_proj",), ("self_attn", "o_proj")),
    "gate_up_proj": (("gate_up_proj",), ("mlp", "gate_up_proj")),
    "down_proj": (("down_proj",), ("mlp", "down_proj")),
}


def _get_text_config(config: Any) -> Any:
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return getattr(config, "text_config", config)


def _is_qwen3_5_model(model: Any) -> bool:
    config = getattr(model, "config", None)
    text_config = _get_text_config(config)
    model_types = {
        str(getattr(config, "model_type", "")),
        str(getattr(text_config, "model_type", "")),
        model.__class__.__name__,
    }
    return any("qwen3_5" in model_type.lower() for model_type in model_types)


def _get_text_model(model: Any) -> Any | None:
    seen: set[int] = set()
    queue: list[tuple[Any, int]] = [(model, 0)]

    while queue:
        current, depth = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))

        if _has_attr(current, "layers"):
            return current
        if depth >= 3:
            continue

        for attr in ("model", "language_model", "transformer"):
            child = getattr(current, attr, None)
            if child is not None:
                queue.append((child, depth + 1))
    return None


def _get_path(obj: Any, path: tuple[str, ...]) -> Any | None:
    current = obj
    for attr in path:
        if not _has_attr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def _unwrap_lora_layer(module: Any) -> Any:
    return getattr(module, "base_layer", module)


def _linear_hidden_dim(module: Any) -> tuple[int, int] | None:
    module = _unwrap_lora_layer(module)
    input_size = getattr(module, "input_size", None)
    output_sizes = getattr(module, "output_sizes", None)
    if output_sizes is not None:
        output_size = sum(int(x) for x in output_sizes)
    else:
        output_size = getattr(module, "output_size", None)
    if input_size is None or output_size is None:
        return None
    return int(input_size), int(output_size)


def _iter_layers_for_hidden_dim(model: Any, layer_idx: int) -> Iterable[Any]:
    text_model = _get_text_model(model)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        return

    yielded_ids: set[int] = set()

    try:
        layer = layers[layer_idx]
    except Exception:
        layer = None
    if layer is not None:
        yielded_ids.add(id(layer))
        yield layer

    try:
        layer_count = len(layers)
    except TypeError:
        return

    for idx in range(layer_count):
        try:
            layer = layers[idx]
        except Exception:
            continue
        if id(layer) in yielded_ids:
            continue
        yielded_ids.add(id(layer))
        yield layer


def _module_hidden_dim_from_layers(
    model: Any,
    module_name: str,
    layer_idx: int,
) -> tuple[int, int] | None:
    attr_paths = _LORA_MODULE_ATTR_PATHS.get(module_name)
    if not attr_paths:
        return None

    for layer in _iter_layers_for_hidden_dim(model, layer_idx):
        for path in attr_paths:
            module = _get_path(layer, path)
            if module is None:
                continue
            hidden_dim = _linear_hidden_dim(module)
            if hidden_dim is not None:
                return hidden_dim
    return None


def _qwen3_5_config_hidden_dim(
    model: Any,
    module_name: str,
    layer_idx: int,
) -> tuple[int, int] | None:
    del layer_idx
    config = _get_text_config(getattr(model, "config", None))
    if config is None:
        return None

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        return None
    hidden_size = int(hidden_size)

    vocab_size = getattr(config, "vocab_size", None)
    if module_name == "embed_tokens" and vocab_size is not None:
        return int(vocab_size), hidden_size
    if module_name == "lm_head" and vocab_size is not None:
        return hidden_size, int(vocab_size)

    intermediate_size = getattr(config, "intermediate_size", None)
    if module_name == "gate_up_proj" and intermediate_size is not None:
        return hidden_size, int(intermediate_size) * 2
    if module_name == "down_proj" and intermediate_size is not None:
        return int(intermediate_size), hidden_size

    moe_intermediate_size = (
        getattr(config, "moe_intermediate_size", None) or intermediate_size
    )
    if module_name == "gate_up_proj_moe" and moe_intermediate_size is not None:
        return hidden_size, int(moe_intermediate_size) * 2
    if module_name == "down_proj_moe" and moe_intermediate_size is not None:
        return int(moe_intermediate_size), hidden_size

    head_dim = getattr(config, "head_dim", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    if head_dim is None or num_attention_heads is None or num_key_value_heads is None:
        return None
    head_dim = int(head_dim)
    num_attention_heads = int(num_attention_heads)
    num_key_value_heads = int(num_key_value_heads)

    if module_name == "qkv_proj":
        query_multiplier = 2 if getattr(config, "attn_output_gate", True) else 1
        output_dim = head_dim * (
            num_attention_heads * query_multiplier + num_key_value_heads * 2
        )
        return hidden_size, output_dim
    if module_name == "o_proj":
        return head_dim * num_attention_heads, hidden_size
    return None


def attach_lora_hidden_dim_from_modules(model: Any) -> bool:
    """Attach SGLang LoRA hidden-dim inference for Qwen3.5-style models.

    SGLang's generic LoRA fallback assumes ordinary Q/K/V projections. Qwen3.5
    attention can pack an output gate into ``qkv_proj``, so the correct LoRA B
    dimension is the actual fused module output size rather than the standard
    QKV formula.
    """

    if hasattr(model, "get_hidden_dim"):
        return False
    if not _is_qwen3_5_model(model):
        return False

    def get_hidden_dim(self: Any, module_name: str, layer_idx: int) -> tuple[int, int]:
        hidden_dim = _module_hidden_dim_from_layers(self, module_name, layer_idx)
        if hidden_dim is not None:
            return hidden_dim
        hidden_dim = _qwen3_5_config_hidden_dim(self, module_name, layer_idx)
        if hidden_dim is not None:
            return hidden_dim
        raise NotImplementedError("get_hidden_dim not implemented for " + module_name)

    model.get_hidden_dim = MethodType(get_hidden_dim, model)
    return True


def patch_sglang_lora_config_compat() -> None:
    """Patch SGLang ModelRunner to normalize composite configs before LoRA init."""

    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_areal_lora_config_compat_patched", False):
        return

    original_init_lora_manager = ModelRunner.init_lora_manager

    def init_lora_manager_with_config_compat(self):
        promote_text_config_attrs(self.model_config.hf_config)
        attach_lora_hidden_dim_from_modules(self.model)
        return original_init_lora_manager(self)

    ModelRunner.init_lora_manager = init_lora_manager_with_config_compat
    ModelRunner._areal_lora_config_compat_patched = True

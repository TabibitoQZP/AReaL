# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter


@pytest.mark.parametrize(
    ("model_type", "raw_name", "expected"),
    [
        (
            "qwen2_5_vl",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        (
            "qwen3_5",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "qwen2_5_vl",
            "model.model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "qwen3_5",
            "model.visual.blocks.0.attn.qkv.weight",
            "visual.blocks.0.attn.qkv.weight",
        ),
    ],
)
def test_fsdp_qwen_vl_name_matches_hf_checkpoint(
    model_type: str, raw_name: str, expected: str
):
    """Qwen-VL training names should use the checkpoint's canonical layout."""
    adapter = object.__new__(AwexFSDPAdapter)
    adapter._engine = SimpleNamespace(
        is_vision_model=True,
        model_config=SimpleNamespace(model_type=model_type),
    )

    assert adapter._to_hf_name(raw_name) == expected


@pytest.mark.parametrize(("suffix", "shape"), [("weight", (6, 2)), ("bias", (6,))])
def test_sglang_qwen_vl_visual_qkv_matches_hf_checkpoint(
    suffix: str, shape: tuple[int, ...]
):
    """Visual QKV stays fused because the HF checkpoint also stores it fused."""
    adapter = object.__new__(AwexSGLangAdapter)
    tensor = torch.arange(torch.Size(shape).numel()).reshape(shape)

    result = adapter._unfuse_params(f"visual.blocks.0.attn.qkv_proj.{suffix}", tensor)

    assert len(result) == 1
    assert result[0][0] == f"visual.blocks.0.attn.qkv.{suffix}"
    assert result[0][1] is tensor


def test_sglang_language_qkv_still_splits_gqa_projections():
    """The visual special case must not change language-model QKV handling."""
    adapter = object.__new__(AwexSGLangAdapter)
    adapter._scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    config=SimpleNamespace(
                        num_attention_heads=4,
                        num_key_value_heads=2,
                    )
                )
            )
        )
    )
    tensor = torch.arange(16).reshape(8, 2)

    result = adapter._unfuse_params("model.layers.0.self_attn.qkv_proj.weight", tensor)

    assert [name for name, _ in result] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    assert [value.shape for _, value in result] == [(4, 2), (2, 2), (2, 2)]


def test_sglang_qwen_vl_visual_mlp_removes_runtime_padding():
    """SGLang's aligned visual MLP width should match the HF checkpoint width."""
    adapter = object.__new__(AwexSGLangAdapter)
    adapter._scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    config=SimpleNamespace(
                        vision_config=SimpleNamespace(intermediate_size=5)
                    )
                )
            )
        )
    )
    gate_up = torch.arange(24).reshape(12, 2)
    down = torch.arange(12).reshape(2, 6)

    gate, up = adapter._unfuse_params(
        "visual.blocks.0.mlp.gate_up_proj.weight", gate_up
    )
    down_result = adapter._unfuse_params("visual.blocks.0.mlp.down_proj.weight", down)

    assert gate[0] == "visual.blocks.0.mlp.gate_proj.weight"
    assert up[0] == "visual.blocks.0.mlp.up_proj.weight"
    torch.testing.assert_close(gate[1], gate_up[:5], rtol=0, atol=0)
    torch.testing.assert_close(up[1], gate_up[6:11], rtol=0, atol=0)
    assert down_result[0][0] == "visual.blocks.0.mlp.down_proj.weight"
    torch.testing.assert_close(down_result[0][1], down[:, :5], rtol=0, atol=0)


def test_sglang_qwen3_5_full_attention_restores_hf_layout():
    """Qwen3.5 gated QKV and flat attention names should match HF names."""
    adapter = object.__new__(AwexSGLangAdapter)
    adapter._scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    config=SimpleNamespace(
                        model_type="qwen3_5_text",
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        attn_output_gate=True,
                    )
                )
            )
        )
    )
    qkv = torch.arange(24).reshape(12, 2)
    o_proj = torch.arange(8).reshape(4, 2)

    result = adapter._unfuse_params("model.layers.3.qkv_proj.weight", qkv)
    o_result = adapter._unfuse_params("model.layers.3.o_proj.weight", o_proj)

    assert [name for name, _ in result] == [
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
    ]
    assert [value.shape for _, value in result] == [(8, 2), (2, 2), (2, 2)]
    assert o_result[0][0] == "model.layers.3.self_attn.o_proj.weight"
    assert o_result[0][1] is o_proj


def test_sglang_qwen3_5_gdn_restores_split_checkpoint_layout():
    """Qwen3.5 GDN fused runtime projections should expose checkpoint views."""
    adapter = object.__new__(AwexSGLangAdapter)
    adapter._scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    config=SimpleNamespace(
                        model_type="qwen3_5_text",
                        linear_num_key_heads=2,
                        linear_key_head_dim=2,
                        linear_num_value_heads=3,
                        linear_value_head_dim=2,
                    )
                )
            )
        )
    )
    qkvz = torch.arange(80).reshape(20, 4)
    ba = torch.arange(24).reshape(6, 4)

    qkv_result = adapter._unfuse_params(
        "model.layers.0.linear_attn.in_proj_qkvz.weight", qkvz
    )
    ba_result = adapter._unfuse_params(
        "model.layers.0.linear_attn.in_proj_ba.weight", ba
    )

    assert [name for name, _ in qkv_result] == [
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
    ]
    assert [value.shape for _, value in qkv_result] == [(14, 4), (6, 4)]
    assert [name for name, _ in ba_result] == [
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
    ]
    assert [value.shape for _, value in ba_result] == [(3, 4), (3, 4)]


def test_fsdp_awex_parameters_use_compute_dtype():
    """FP32 master weights must be cast before transfer to BF16 inference."""
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
    adapter = object.__new__(AwexFSDPAdapter)
    adapter._engine = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(tie_word_embeddings=False),
        is_vision_model=False,
        _cast_to_compute_dtype=lambda tensor: tensor.to(torch.bfloat16),
    )

    params = adapter.get_local_shard_parameters()

    assert params["weight"].dtype == torch.bfloat16
    a_log = torch.ones(2, dtype=torch.float32)
    assert adapter._to_transfer_dtype(
        "model.layers.0.linear_attn.A_log", a_log
    ).dtype == (torch.float32)

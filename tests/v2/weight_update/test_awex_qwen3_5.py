# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter


def _sglang_adapter(config) -> AwexSGLangAdapter:
    adapter = object.__new__(AwexSGLangAdapter)
    adapter._scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(config=config))
        )
    )
    return adapter


def test_fsdp_qwen3_5_name_matches_hf_checkpoint():
    adapter = object.__new__(AwexFSDPAdapter)
    adapter._engine = SimpleNamespace(
        is_vision_model=True,
        model_config=SimpleNamespace(model_type="qwen3_5"),
    )

    assert (
        adapter._to_hf_name("model.language_model.layers.0.self_attn.q_proj.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        adapter._to_hf_name("model.visual.blocks.0.attn.qkv.weight")
        == "visual.blocks.0.attn.qkv.weight"
    )


def test_fsdp_qwen_vl_name_mapping_keeps_existing_behavior():
    adapter = object.__new__(AwexFSDPAdapter)
    adapter._engine = SimpleNamespace(
        is_vision_model=True,
        model_config=SimpleNamespace(model_type="qwen2_5_vl"),
    )

    assert (
        adapter._to_hf_name("model.model.layers.0.self_attn.q_proj.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        adapter._to_hf_name("model.language_model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )


def test_fsdp_qwen3_5_preserves_a_log_fp32():
    adapter = object.__new__(AwexFSDPAdapter)
    adapter._engine = SimpleNamespace(
        model_config=SimpleNamespace(model_type="qwen3_5"),
        _cast_to_compute_dtype=lambda tensor: tensor.to(torch.bfloat16),
    )

    tensor = torch.ones(2, dtype=torch.float32)

    assert (
        adapter._to_transfer_dtype("model.layers.0.linear_attn.A_log", tensor).dtype
        == torch.float32
    )
    assert (
        adapter._to_transfer_dtype("model.layers.0.mlp.up_proj.weight", tensor).dtype
        == torch.bfloat16
    )


def test_fsdp_non_qwen3_5_preserves_transfer_dtype():
    adapter = object.__new__(AwexFSDPAdapter)
    adapter._engine = SimpleNamespace(
        model_config=SimpleNamespace(model_type="llama"),
        _cast_to_compute_dtype=lambda tensor: tensor.to(torch.bfloat16),
    )
    tensor = torch.ones(2, dtype=torch.float32)

    assert adapter._to_transfer_dtype("model.layers.0.mlp.weight", tensor) is tensor


def test_sglang_qwen3_5_restores_attention_checkpoint_layout():
    adapter = _sglang_adapter(
        SimpleNamespace(
            model_type="qwen3_5",
            num_attention_heads=2,
            num_key_value_heads=1,
            attn_output_gate=True,
        )
    )
    qkv = torch.arange(24).reshape(12, 2)

    result = adapter._unfuse_params("model.layers.3.qkv_proj.weight", qkv)

    assert [name for name, _ in result] == [
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
    ]
    assert [value.shape for _, value in result] == [(8, 2), (2, 2), (2, 2)]
    torch.testing.assert_close(result[0][1], qkv[:8], rtol=0, atol=0)
    torch.testing.assert_close(result[1][1], qkv[8:10], rtol=0, atol=0)
    torch.testing.assert_close(result[2][1], qkv[10:], rtol=0, atol=0)


def test_sglang_qwen3_5_restores_gdn_checkpoint_layout():
    adapter = _sglang_adapter(
        SimpleNamespace(
            model_type="qwen3_5",
            linear_num_key_heads=2,
            linear_key_head_dim=2,
            linear_num_value_heads=3,
            linear_value_head_dim=2,
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
    torch.testing.assert_close(qkv_result[0][1], qkvz[:14], rtol=0, atol=0)
    torch.testing.assert_close(qkv_result[1][1], qkvz[14:], rtol=0, atol=0)
    assert [name for name, _ in ba_result] == [
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
    ]
    torch.testing.assert_close(ba_result[0][1], ba[:3], rtol=0, atol=0)
    torch.testing.assert_close(ba_result[1][1], ba[3:], rtol=0, atol=0)


def test_sglang_qwen3_5_visual_weights_match_hf_layout():
    adapter = _sglang_adapter(
        SimpleNamespace(
            model_type="qwen3_5",
            vision_config=SimpleNamespace(intermediate_size=5),
        )
    )
    qkv = torch.arange(12).reshape(6, 2)
    gate_up = torch.arange(24).reshape(12, 2)
    down = torch.arange(12).reshape(2, 6)

    qkv_result = adapter._unfuse_params("visual.blocks.0.attn.qkv_proj.weight", qkv)
    gate, up = adapter._unfuse_params(
        "visual.blocks.0.mlp.gate_up_proj.weight", gate_up
    )
    down_result = adapter._unfuse_params("visual.blocks.0.mlp.down_proj.weight", down)

    assert qkv_result[0][0] == "visual.blocks.0.attn.qkv.weight"
    torch.testing.assert_close(gate[1], gate_up[:5], rtol=0, atol=0)
    torch.testing.assert_close(up[1], gate_up[6:11], rtol=0, atol=0)
    torch.testing.assert_close(down_result[0][1], down[:, :5], rtol=0, atol=0)

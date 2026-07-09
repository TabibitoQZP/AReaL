from types import SimpleNamespace

from areal.v2.inference_service.sglang.compat import (
    attach_lora_hidden_dim_from_modules,
    promote_text_config_attrs,
)


def test_promote_text_config_attrs_fills_lora_attrs_for_composite_config():
    text_config = SimpleNamespace(
        vocab_size=248320,
        num_hidden_layers=32,
        hidden_size=2560,
        intermediate_size=9216,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        tie_word_embeddings=True,
    )
    hf_config = SimpleNamespace(model_type="qwen3_5", text_config=text_config)

    promote_text_config_attrs(hf_config)

    assert hf_config.vocab_size == 248320
    assert hf_config.num_hidden_layers == 32
    assert hf_config.hidden_size == 2560
    assert hf_config.intermediate_size == 9216
    assert hf_config.num_attention_heads == 16
    assert hf_config.num_key_value_heads == 4
    assert hf_config.head_dim == 256
    assert hf_config.tie_word_embeddings is True


def test_promote_text_config_attrs_preserves_existing_top_level_attrs():
    text_config = SimpleNamespace(
        vocab_size=248320,
        num_hidden_layers=32,
    )
    hf_config = SimpleNamespace(
        text_config=text_config,
        vocab_size=151936,
        num_hidden_layers=36,
    )

    promote_text_config_attrs(hf_config)

    assert hf_config.vocab_size == 151936
    assert hf_config.num_hidden_layers == 36


def test_attach_lora_hidden_dim_uses_qwen35_fused_projection_shapes():
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=248320,
        hidden_size=2560,
        intermediate_size=9216,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        attn_output_gate=True,
    )
    linear_layer = SimpleNamespace(
        mlp=SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                input_size=2560,
                output_sizes=[9216, 9216],
            ),
            down_proj=SimpleNamespace(input_size=9216, output_size=2560),
        ),
    )
    attention_layer = SimpleNamespace(
        qkv_proj=SimpleNamespace(
            input_size=2560,
            output_sizes=[8192, 1024, 1024],
        ),
        o_proj=SimpleNamespace(input_size=4096, output_size=2560),
        mlp=linear_layer.mlp,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5", text_config=text_config),
        model=SimpleNamespace(layers=[linear_layer, attention_layer]),
    )

    assert attach_lora_hidden_dim_from_modules(model) is True

    assert model.get_hidden_dim("qkv_proj", 0) == (2560, 10240)
    assert model.get_hidden_dim("o_proj", 0) == (4096, 2560)
    assert model.get_hidden_dim("gate_up_proj", 0) == (2560, 18432)
    assert model.get_hidden_dim("down_proj", 0) == (9216, 2560)


def test_attach_lora_hidden_dim_preserves_existing_model_method():
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5"),
        get_hidden_dim=lambda module_name, layer_idx: (1, 2),
    )

    assert attach_lora_hidden_dim_from_modules(model) is False
    assert model.get_hidden_dim("qkv_proj", 0) == (1, 2)

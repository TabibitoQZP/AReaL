# VLM LoRA Smoke Notes

These notes capture issues found while turning the smoke scripts into a future directly
runnable v2 VLM LoRA example.

## SGLang dynamic LoRA startup

When SGLang starts with `--enable-lora` and no initial `--lora-paths`, it requires both
of these flags:

```bash
--max-lora-rank <rank>
--lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

Otherwise startup fails with:

```text
When no initial --lora-paths is provided, you need to specify both
--max-lora-rank and --lora-target-modules for LoRA initialization.
```

## SGLang LoRA capacity

`--max-loaded-loras` must be greater than or equal to `--max-loras-per-batch`. SGLang
defaults `max_loras_per_batch` to `8`, so a smaller `max_loaded_loras` fails at startup.

The smoke defaults are:

```bash
SGLANG_MAX_LORAS_PER_BATCH=8
SGLANG_MAX_LOADED_LORAS=8
```

## Generation length field

`VisionGeometry3KAgent` drops `max_tokens` in `__init__`, so the smoke uses
`max_completion_tokens=64`. Using `max_tokens=64` does not cap the data proxy's
completion budget and can make SGLang reject the request as exceeding context length.

## v2 request-side LoRA wiring

For v2 data proxy generation to select a loaded adapter, the data proxy needs:

```bash
--use-lora
--lora-name <base-lora-name>
```

The generation request then uses the data proxy weight version to request:

```text
<base-lora-name>-v<version>
```

SGLang's default access log records `/load_lora_adapter` and `/generate` status codes,
but it does not print the full generation JSON payload. The request-side `lora_path`
field is covered by the v2 `InfBridge` unit tests.

## v2 service admin keys

v2 training and inference services refuse to bind to a non-loopback host with the
default `areal-admin-key`. A local launcher may appear stuck if the top-level process
keeps waiting after a child service exits with:

```text
Refusing to start server on non-loopback host ... with the default admin API key
```

A runnable example should set non-default `rollout.admin_api_key` and
`actor.admin_api_key` values. `rollout.agent.admin_api_key` is ignored by the v2 rollout
controller.

## Training-side attention implementation

The v2 VLM LoRA example uses `actor.attn_impl: sdpa` by default. The project environment
may contain `flash-attn-4`, while Transformers' Qwen2.5-VL path still imports the
FlashAttention 2 API:

```text
from flash_attn import flash_attn_func, flash_attn_varlen_func
```

With an incompatible or partial flash-attn installation, train-worker initialization
fails before rollout starts. Keeping the smoke on SDPA makes the example runnable
without a custom FlashAttention 2 wheel. FlashAttention can still be documented as an
optional performance dependency later.

## Background process cleanup

The full-example launcher should start the job in a dedicated process group and stop
related child services by experiment name. Killing only the outer runner can leave
data-service or training-service children alive, which makes the next status check
misleading.

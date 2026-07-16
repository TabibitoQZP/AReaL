# VLM LoRA Smoke Test

Development helper only. This directory contains local scripts for a narrow runtime
smoke test:

```text
VisionGeometry3KAgent
  -> v2 inference data proxy with use_lora=True
  -> InfBridge
  -> SGLang with enable_lora=True
  -> versioned LoRA adapter
```

The scripts write logs, PID files, and generated smoke adapters under:

```text
/mnt/data/zipengqiu/areal/vlm_lora_smoke
```

The directly runnable training example lives in:

```text
examples/vlm/geometry3k_grpo_v2_lora.py
examples/vlm/geometry3k_grpo_v2_lora.yaml
```

## Run

From the repo root:

```bash
bash scripts/vlm_lora_smoke/00_make_adapter.sh
bash scripts/vlm_lora_smoke/10_start_sglang.sh
bash scripts/vlm_lora_smoke/20_start_data_proxy.sh
bash scripts/vlm_lora_smoke/30_load_lora.sh
bash scripts/vlm_lora_smoke/40_run_agent_smoke.sh
```

Stop the background services:

```bash
bash scripts/vlm_lora_smoke/90_stop.sh
```

## Defaults

- GPU: `CUDA_VISIBLE_DEVICES=7`
- model: `/mnt/data/zipengqiu/areal/models/Qwen2.5-VL-3B-Instruct`
- SGLang: `http://127.0.0.1:30005`
- data proxy: `http://127.0.0.1:8085`
- LoRA base name: `vlm-smoke-lora`
- LoRA version: `0`
- SGLang LoRA capacity: `SGLANG_MAX_LORAS_PER_BATCH=8`, `SGLANG_MAX_LOADED_LORAS=8`
- adapter path: `/mnt/data/zipengqiu/areal/vlm_lora_smoke/adapters/vlm-smoke-lora-v0`

Override any default inline, for example:

```bash
CUDA_VISIBLE_DEVICES=7 SGLANG_PORT=30007 DATA_PROXY_PORT=8087 \
  bash scripts/vlm_lora_smoke/10_start_sglang.sh
```

Use the same overrides for all later scripts in the same smoke run.

## Notes

`00_make_adapter.sh` creates a random initialized PEFT adapter. It is only for checking
load/request plumbing, not model quality. If you already have a real trained adapter,
skip `00_make_adapter.sh` and set:

```bash
LORA_ADAPTER_DIR=/path/to/adapter LORA_NAME=my-lora LORA_VERSION=0
```

Then run `10_start_sglang.sh`, `20_start_data_proxy.sh`, `30_load_lora.sh`, and
`40_run_agent_smoke.sh`.

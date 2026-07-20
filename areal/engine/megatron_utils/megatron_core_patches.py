# SPDX-License-Identifier: Apache-2.0

"""Runtime patches for Megatron-Core bugs not yet fixed upstream."""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from megatron.core.jit import jit_fuser

import areal.utils.logging as logging
from areal.engine.core.model import is_qwen3_5_model

logger = logging.getLogger("MCorePatches")


def _uses_fp32_a_log_exp(method) -> bool:
    method = getattr(method, "_torchdynamo_orig_callable", method)
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return False
    compact = "".join(source.split())
    return "A_log_local_cp.float().exp()" in compact


def _gdn_decay_and_beta(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the HF Qwen3.5 GDN decay calculation in FP32."""
    g = -A_log.float().exp() * F.softplus(alpha.float() + dt_bias)
    return g, beta.sigmoid()


def apply_qwen3_5_gdn_precision_patch(model_type: str) -> None:
    """Compute Qwen3.5/3.6 GatedDeltaNet ``exp(A_log)`` in FP32.

    Megatron-Core 0.18.0, as well as its current main/dev branches, calls
    ``exp`` directly on the BF16 checkpoint parameter before multiplying by
    the FP32 softplus term. Hugging Face explicitly casts ``A_log`` to FP32.
    For Qwen3.5/3.6 this changes the decay by up to O(1e-1).

    This is applied explicitly after the model type is known, so importing
    AReaL does not mutate Megatron-Core for unrelated models. The source check
    also makes the patch a no-op once upstream performs the FP32 cast itself.
    """
    if not is_qwen3_5_model(model_type):
        return

    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return

    if getattr(GatedDeltaNet, "_areal_fp32_a_log_exp", False):
        return
    if _uses_fp32_a_log_exp(GatedDeltaNet._compute_g_and_beta):
        return

    @jit_fuser
    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        del self
        return _gdn_decay_and_beta(
            A_log_local_cp,
            dt_bias_local_cp,
            alpha,
            beta,
        )

    GatedDeltaNet._compute_g_and_beta = _compute_g_and_beta
    GatedDeltaNet._areal_fp32_a_log_exp = True
    logger.info("Patched Megatron-Core GatedDeltaNet A_log.exp() to run in FP32.")

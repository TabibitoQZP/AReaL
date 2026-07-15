# SPDX-License-Identifier: Apache-2.0

"""Runtime patches for Megatron-Core bugs not yet fixed upstream."""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from megatron.core.jit import jit_fuser

import areal.utils.logging as logging

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


def _patch_gdn_a_log_exp_precision() -> None:
    """Compute GatedDeltaNet ``exp(A_log)`` in FP32.

    Megatron-Core 0.18.0, as well as its current main/dev branches, calls
    ``exp`` directly on the BF16 checkpoint parameter before multiplying by
    the FP32 softplus term. Hugging Face explicitly casts ``A_log`` to FP32.
    For Qwen3.5/3.6 this changes the decay by up to O(1e-1).

    The source check makes this compatibility patch a no-op once upstream
    performs the FP32 cast itself.
    """
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


def _apply_patches_on_import() -> None:
    _patch_gdn_a_log_exp_precision()


_apply_patches_on_import()

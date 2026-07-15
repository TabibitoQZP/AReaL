# SPDX-License-Identifier: Apache-2.0

import torch

from areal.engine.megatron_utils.megatron_core_patches import (
    _gdn_decay_and_beta,
    _uses_fp32_a_log_exp,
)


def test_gdn_decay_computes_a_log_exp_in_fp32():
    A_log = torch.tensor([3.53125, -3.765625], dtype=torch.bfloat16)
    dt_bias = torch.tensor([0.82421875, -2.84375], dtype=torch.bfloat16)
    alpha = torch.tensor([-1.7265625, 0.5859375], dtype=torch.bfloat16)
    beta = torch.tensor([-0.5, 0.75], dtype=torch.bfloat16)

    decay, actual_beta = _gdn_decay_and_beta(A_log, dt_bias, alpha, beta)

    expected_decay = -A_log.float().exp() * torch.nn.functional.softplus(
        alpha.float() + dt_bias
    )
    bf16_exp_decay = -A_log.exp() * torch.nn.functional.softplus(
        alpha.float() + dt_bias
    )
    torch.testing.assert_close(decay, expected_decay, rtol=0, atol=0)
    assert not torch.equal(decay, bf16_exp_decay)
    torch.testing.assert_close(actual_beta, beta.sigmoid(), rtol=0, atol=0)


def test_gdn_precision_is_fixed_by_upstream_or_runtime_patch():
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    assert getattr(GatedDeltaNet, "_areal_fp32_a_log_exp", False) or (
        _uses_fp32_a_log_exp(GatedDeltaNet._compute_g_and_beta)
    )

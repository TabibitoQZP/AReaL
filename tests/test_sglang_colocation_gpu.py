# SPDX-License-Identifier: Apache-2.0
"""Colocated SGLang must address the worker's CUDA-visible namespace."""

from types import SimpleNamespace

import pytest

from areal.engine import sglang_remote
from areal.engine.sglang_remote import SGLangBackend


@pytest.fixture
def launched(monkeypatch):
    """Capture real CLI arguments and environment without creating a server."""
    for key in ("AREAL_SGLANG_FORK", "AWEX_META_SERVER_ADDR", "SLURM_LOCALID"):
        monkeypatch.delenv(key, raising=False)
    calls = []

    def popen(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]))
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(sglang_remote.subprocess, "Popen", popen)
    return calls


@pytest.mark.parametrize("localid", [None, "0", "6"])
def test_isolated_colocation_uses_logical_zero(monkeypatch, launched, localid):
    """Neither a global replica index nor inherited Slurm rank is a CUDA index."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("RANK", "6")
    monkeypatch.setenv("WORLD_SIZE", "8")
    if localid is not None:
        monkeypatch.setenv("SLURM_LOCALID", localid)

    SGLangBackend().launch_server(
        {"model_path": "model", "base_gpu_id": 6, "_awex_gpus_per_server": 1}
    )

    cmd, env = launched[0]
    assert cmd[cmd.index("--base-gpu-id") + 1] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "6"
    assert env["RANK"] == "6" and env["WORLD_SIZE"] == "8"
    assert "--awex-gpus-per-server" not in cmd


@pytest.mark.parametrize(
    "visible,localid,incoming,expected",
    [
        ("0,1,2,3,4,5,6,7", "2", 0, 4),
        ("0,1,2,3,4,5,6,7", None, 4, 4),
        ("7,5,3,1", "1", 0, 2),
        (None, "2", 0, 4),
        (None, None, 4, 4),
    ],
)
def test_shared_colocation_retains_slot_offsets(
    monkeypatch, launched, visible, localid, incoming, expected
):
    """Shared-node placement still needs distinct offsets within visible GPUs."""
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    if localid is not None:
        monkeypatch.setenv("SLURM_LOCALID", localid)

    SGLangBackend().launch_server(
        {
            "model_path": "model",
            "base_gpu_id": incoming,
            "tp_size": 2,
            "_awex_gpus_per_server": 2,
        }
    )

    cmd, env = launched[0]
    assert cmd[cmd.index("--base-gpu-id") + 1] == str(expected)
    assert env.get("CUDA_VISIBLE_DEVICES") == visible


def test_separated_launch_does_not_rewrite_gpu_id(monkeypatch, launched):
    """The new mapping must be gated on the controller's colocation marker."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("SLURM_LOCALID", "6")

    SGLangBackend().launch_server({"model_path": "model", "base_gpu_id": 0})

    cmd, env = launched[0]
    assert cmd[cmd.index("--base-gpu-id") + 1] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "6"


@pytest.mark.parametrize("visible", ["", "-1", "6,", ",6", "6,6"])
def test_invalid_visible_namespace_rejected_before_launch(
    monkeypatch, launched, visible
):
    """Do not mistake an empty, disabled or malformed list for one usable GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        SGLangBackend().launch_server(
            {"model_path": "model", "base_gpu_id": 6, "_awex_gpus_per_server": 1}
        )
    assert launched == []


def test_insufficient_visible_devices_rejected_before_launch(monkeypatch, launched):
    """A two-GPU instance cannot be forked from a single-visible-GPU worker."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    with pytest.raises(ValueError, match="visible GPUs"):
        SGLangBackend().launch_server(
            {"model_path": "model", "base_gpu_id": 0, "_awex_gpus_per_server": 2}
        )
    assert launched == []

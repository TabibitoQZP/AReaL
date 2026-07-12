# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType

from areal.v2.training_service.worker.awex import _create_training_adapter


def test_create_training_adapter_does_not_import_megatron_for_fsdp(monkeypatch):
    """FSDP weight update must not require optional Megatron dependencies."""

    class FakeFSDPEngine:
        pass

    class FakeAwexFSDPAdapter:
        def __init__(self, engine):
            self.engine = engine

    fsdp_engine_module = ModuleType("areal.engine.fsdp_engine")
    fsdp_engine_module.FSDPEngine = FakeFSDPEngine
    fsdp_adapter_module = ModuleType("areal.v2.weight_update.awex.fsdp_adapter")
    fsdp_adapter_module.AwexFSDPAdapter = FakeAwexFSDPAdapter

    monkeypatch.setitem(sys.modules, "areal.engine.fsdp_engine", fsdp_engine_module)
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.fsdp_adapter",
        fsdp_adapter_module,
    )
    monkeypatch.delitem(sys.modules, "areal.engine.megatron_engine", raising=False)

    engine = FakeFSDPEngine()
    adapter = _create_training_adapter(engine)

    assert isinstance(adapter, FakeAwexFSDPAdapter)
    assert adapter.engine is engine
    assert "areal.engine.megatron_engine" not in sys.modules

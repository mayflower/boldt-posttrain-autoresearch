from types import SimpleNamespace

import pytest

from boldt_posttrain.training import validate_device


class FakeCuda:
    def __init__(self, name, vram_gb=48, bf16=True):
        self.properties = SimpleNamespace(name=name, total_memory=vram_gb * 1024**3)
        self.bf16 = bf16

    def is_available(self):
        return True

    def get_device_properties(self, index):
        if index != 0:
            raise IndexError(index)
        return self.properties

    def is_bf16_supported(self):
        return self.bf16


@pytest.mark.parametrize("name", ["NVIDIA A100", "NVIDIA L40S", "NVIDIA H100"])
def test_device_gate_accepts_capabilities_not_a_specific_compute_capability(name):
    torch = SimpleNamespace(cuda=FakeCuda(name), bfloat16="bf16")
    result = validate_device(
        "cuda:0", minimum_vram_gb=40, torch_module=torch, bitsandbytes_smoke=lambda: None
    )
    assert result["name"] == name
    assert result["bitsandbytes_4bit"] is True


def test_device_gate_rejects_missing_memory_or_bf16():
    low_memory = SimpleNamespace(cuda=FakeCuda("GPU", vram_gb=24), bfloat16="bf16")
    with pytest.raises(RuntimeError, match="requires"):
        validate_device(
            "cuda:0",
            minimum_vram_gb=40,
            torch_module=low_memory,
            bitsandbytes_smoke=lambda: None,
        )
    no_bf16 = SimpleNamespace(cuda=FakeCuda("GPU", bf16=False), bfloat16="bf16")
    with pytest.raises(RuntimeError, match="BF16"):
        validate_device(
            "cuda:0",
            minimum_vram_gb=40,
            torch_module=no_bf16,
            bitsandbytes_smoke=lambda: None,
        )

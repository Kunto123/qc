from __future__ import annotations

import unittest

from backend.app.core.device_runtime import DeviceRuntimeResolver


class _Config:
    def __init__(self, *, device_mode: str = "auto", cuda_device_id: int = 0) -> None:
        self.device_mode = device_mode
        self.cuda_device_id = cuda_device_id


class DeviceRuntimeResolverTest(unittest.TestCase):
    def test_auto_uses_cuda_when_available(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="auto", cuda_device_id=1))
        resolver._cuda_status = lambda: (True, 2)  # type: ignore[assignment]

        resolution = resolver.resolve()

        self.assertEqual(resolution.requested_mode, "auto")
        self.assertEqual(resolution.effective_device, "cuda:1")
        self.assertEqual(resolution.backend, "cuda")
        self.assertIsNone(resolution.fallback_reason)

    def test_gpu_falls_back_to_cpu_when_cuda_missing(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="gpu", cuda_device_id=0))
        resolver._cuda_status = lambda: (False, 0)  # type: ignore[assignment]

        resolution = resolver.resolve()

        self.assertEqual(resolution.requested_mode, "gpu")
        self.assertEqual(resolution.effective_device, "cpu")
        self.assertEqual(resolution.backend, "cpu")
        self.assertEqual(resolution.fallback_reason, "cuda_unavailable")

    def test_invalid_cuda_index_clamps_to_first_available_device(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="auto", cuda_device_id=5))
        resolver._cuda_status = lambda: (True, 1)  # type: ignore[assignment]

        resolution = resolver.resolve()

        self.assertEqual(resolution.effective_device, "cuda:0")
        self.assertEqual(resolution.backend, "cuda")
        self.assertIn("cuda_device_id_out_of_range", resolution.fallback_reason or "")
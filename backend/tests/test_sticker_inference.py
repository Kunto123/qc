from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from backend.app.core.config import AppConfig
from backend.app.services.sticker_inference import StickerInferenceService


class _Scalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeBox:
    def __init__(self, *, xyxy: list[float], class_id: int, confidence: float) -> None:
        self.xyxy = np.array([xyxy], dtype=float)
        self.cls = [_Scalar(float(class_id))]
        self.conf = [_Scalar(float(confidence))]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class StickerInferenceFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = StickerInferenceService(AppConfig(), mock.Mock())

    def test_normalize_label_key_removes_non_alnum(self) -> None:
        self.assertEqual(self.service._normalize_label_key("K0W-HB0"), "k0whb0")
        self.assertEqual(self.service._normalize_label_key("K0W_HB0"), "k0whb0")

    def test_normalize_detections_accepts_canonical_label_match(self) -> None:
        result = _FakeResult(
            [
                _FakeBox(xyxy=[1.0, 2.0, 3.0, 4.0], class_id=0, confidence=0.91),
                _FakeBox(xyxy=[5.0, 6.0, 7.0, 8.0], class_id=1, confidence=0.88),
            ]
        )
        names = {0: "K0W_HB0", 1: "OTHER"}
        detections = self.service._normalize_detections(
            result=result,
            names=names,
            allowed_labels={"k0w-hb0"},
            allowed_label_keys={"k0whb0"},
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "K0W_HB0")
        self.assertEqual(detections[0]["class_id"], 0)


if __name__ == "__main__":
    unittest.main()

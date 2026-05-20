from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from backend.app.core.config import AppConfig
from backend.app.services.sticker_inference import StickerInferenceService
from shared.contracts.templates import StickerRule, VisionConfig


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

    def test_normalize_ocr_text_applies_regex_and_canonical_map(self) -> None:
        result = self.service.normalize_ocr_text(
            " code: k0w hb0 ",
            expected_text="K0W-HB0",
            regex=r"K0W\s*HB0",
            canonical_map={"K0WHB0": "K0W-HB0"},
        )
        self.assertEqual(result, "K0W-HB0")

    def test_anchor_ocr_payload_from_passthrough_detection(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        vision = VisionConfig(
            ocr_engine="passthrough",
            text_anchor_class="text_anchor",
            center_dot_class="center_dot",
        )
        sticker = StickerRule(
            part_name="P1",
            expected_class="K0W-HB0",
            line="L1",
            expected_dot_x=0.5,
            expected_dot_y=0.5,
        )
        payload = {
            "backend": "patched",
            "detections": [
                {
                    "label": "text_anchor",
                    "confidence": 0.92,
                    "ocr_text": "K0W HB0",
                    "ocr_confidence": 0.88,
                    "position": {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 40.0},
                },
                {
                    "label": "center_dot",
                    "confidence": 0.95,
                    "position": {"x1": 48.0, "y1": 48.0, "x2": 52.0, "y2": 52.0},
                },
            ],
        }

        result = self.service._augment_with_anchor_ocr(
            payload,
            image,
            vision,
            expected_class="K0W-HB0",
            sticker_rule=sticker,
        )

        self.assertEqual(result["anchor"]["status"], "ok")
        self.assertEqual(result["ocr"]["status"], "ok")
        self.assertTrue(result["ocr"]["match_expected"])
        self.assertEqual(result["geometry"]["anchor_offset"], {"x": 0.0, "y": 0.0, "source": "center_dot"})
        self.assertIn("ocr_ms", result["timings"])


if __name__ == "__main__":
    unittest.main()

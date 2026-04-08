from __future__ import annotations

import unittest

from backend.app.core.model_catalog import get_base_model, list_base_models, resolve_base_model


class ModelCatalogTest(unittest.TestCase):
    def test_catalog_lists_all_yolo_variants(self) -> None:
        items = list_base_models()
        self.assertEqual(len(items), 10)
        self.assertEqual(items[0]["id"], "yolov5n")
        self.assertEqual(items[-1]["id"], "yolov11x")

    def test_resolve_base_model_supports_key_and_family_variant(self) -> None:
        by_key = resolve_base_model("yolov5s")
        self.assertIsNotNone(by_key)
        self.assertEqual(by_key["family"], "yolov5")
        self.assertEqual(by_key["variant"], "s")

        by_family_variant = resolve_base_model(family="yolov11", variant="m")
        self.assertIsNotNone(by_family_variant)
        self.assertEqual(by_family_variant["id"], "yolov11m")
        self.assertEqual(by_family_variant["display_name"], "YOLOv11 Medium")

    def test_get_base_model_returns_none_for_unknown_id(self) -> None:
        self.assertIsNone(get_base_model("does-not-exist"))

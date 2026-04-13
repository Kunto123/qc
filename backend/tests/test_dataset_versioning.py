from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="qc-suite-versioning-tests-"))
atexit.register(lambda: shutil.rmtree(TEST_DATA_ROOT, ignore_errors=True))
os.environ["QC_SUITE_DATA_ROOT"] = str(TEST_DATA_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.repositories.dataset_versions_repository import DatasetVersionRepository
from backend.app.repositories.datasets_repository import DatasetsRepository


def _sample_image_bytes(color: tuple[int, int, int]) -> bytes:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = color
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode sample image")
    return encoded.tobytes()


class DatasetVersioningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.datasets_repo = DatasetsRepository()
        self.versions_repo = DatasetVersionRepository(self.datasets_repo)

    def test_create_version_exports_yolo_snapshot(self) -> None:
        dataset = self.datasets_repo.create_dataset("versioned-dataset", "smoke")
        self.datasets_repo.save_file(dataset["id"], "images", "a.jpg", _sample_image_bytes((10, 40, 80)))
        self.datasets_repo.save_file(dataset["id"], "images", "b.jpg", _sample_image_bytes((20, 60, 120)))
        self.datasets_repo.save_file(dataset["id"], "images", "c.jpg", _sample_image_bytes((30, 90, 160)))
        self.datasets_repo.save_annotation(
            dataset["id"],
            "a.jpg",
            [
                {
                    "type": "bbox",
                    "class": "bbox-class",
                    "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
                }
            ],
        )
        self.datasets_repo.save_annotation(
            dataset["id"],
            "b.jpg",
            [
                {
                    "type": "polygon",
                    "class": "poly-class",
                    "points": [
                        {"x": 30, "y": 40},
                        {"x": 60, "y": 40},
                        {"x": 60, "y": 70},
                    ],
                }
            ],
        )

        version = self.versions_repo.create_version(
            dataset["id"],
            {
                "name": "Smoke version",
                "description": "version export",
                "split_ratios": {"train": 0.6, "valid": 0.2, "test": 0.2},
            },
        )

        self.assertEqual(version["dataset_id"], dataset["id"])
        self.assertEqual(version["status"], "ready")
        self.assertTrue(version["ready_for_training"])
        self.assertEqual(version["image_count"], 3)
        self.assertEqual(version["annotated_image_count"], 2)
        self.assertEqual(len(version["class_names"]), 2)

        export_root = Path(version["export_root"])
        self.assertTrue(export_root.exists())
        self.assertTrue((export_root / "data.yaml").exists())
        self.assertTrue((export_root / "classes.txt").exists())
        data_yaml = (export_root / "data.yaml").read_text(encoding="utf-8")
        self.assertIn(f"path: {export_root.resolve().as_posix()}", data_yaml)
        self.assertIn("bbox-class", data_yaml)
        self.assertIn("poly-class", data_yaml)

        manifest = Path(version["manifest_path"])
        self.assertTrue(manifest.exists())
        self.assertIn("split_counts", manifest.read_text(encoding="utf-8"))

        image_exports = list((export_root / "images").rglob("*.jpg"))
        label_exports = list((export_root / "labels").rglob("*.txt"))
        self.assertEqual(len(image_exports), 3)
        self.assertEqual(len(label_exports), 3)

        refreshed = self.versions_repo.export_version(dataset["id"], version["id"])
        self.assertEqual(refreshed["id"], version["id"])
        self.assertTrue(Path(refreshed["export_root"]).exists())
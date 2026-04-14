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


# ---------------------------------------------------------------------------
# Augment integration — create_version with augment_jobs parameter
# ---------------------------------------------------------------------------

class AugmentIntegrationVersioningTest(unittest.TestCase):
    """create_version correctly integrates augmented images into the snapshot."""

    def setUp(self) -> None:
        self.datasets_repo = DatasetsRepository()
        self.versions_repo = DatasetVersionRepository(self.datasets_repo)
        # Create dataset with 2 original images and one annotation.
        self.dataset = self.datasets_repo.create_dataset("aug-integration-ds", "augment test")
        self.datasets_repo.save_file(self.dataset["id"], "images", "img1.jpg", _sample_image_bytes((10, 40, 80)))
        self.datasets_repo.save_file(self.dataset["id"], "images", "img2.jpg", _sample_image_bytes((20, 60, 120)))
        self.datasets_repo.save_annotation(
            self.dataset["id"],
            "img1.jpg",
            [{"type": "bbox", "class": "obj", "bbox": {"x": 10, "y": 10, "w": 20, "h": 20}}],
        )

    def _make_augment_job_stub(self, multiplier: int = 2) -> dict:
        """Create a synthetic completed augment job with photometric output files."""
        from backend.app.repositories.datasets_repository import DatasetsRepository as _DR
        ds_dir = _DR().dataset_dir(self.dataset["id"])
        job_id = "augtest001"
        output_dir = ds_dir / "augmented" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write augmented copies of img1.jpg: img1_aug001.jpg, img1_aug002.jpg
        for i in range(1, multiplier + 1):
            shutil.copy2(ds_dir / "images" / "img1.jpg", output_dir / f"img1_aug{i:03d}.jpg")
        return {
            "id": job_id,
            "dataset_id": self.dataset["id"],
            "status": "completed",
            "transforms": ["brightness", "contrast"],
            "multiplier": multiplier,
            "output_dataset_id": str(output_dir),
            "augmented_image_count": multiplier,
        }

    def test_create_version_without_augment_unchanged(self) -> None:
        """create_version with no augment_jobs behaves identically to before."""
        version = self.versions_repo.create_version(self.dataset["id"])
        self.assertEqual(version["image_count"], 2)
        self.assertEqual(version["selected_augment_job_ids"], [])
        self.assertEqual(version["augmented_image_count_in_version"], 0)
        self.assertEqual(version["total_source_image_count"], 2)

    def test_create_version_with_augment_increases_image_count(self) -> None:
        """image_count in version record = original + augmented when augment_jobs are provided."""
        job = self._make_augment_job_stub(multiplier=2)
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        # 2 original + 2 augmented copies of img1
        self.assertEqual(version["image_count"], 4)
        self.assertEqual(version["augmented_image_count_in_version"], 2)
        self.assertEqual(version["total_source_image_count"], 2)
        self.assertIn(job["id"], version["selected_augment_job_ids"])

    def test_create_version_augment_export_includes_augmented_images(self) -> None:
        """The YOLO export must contain original + augmented images in its splits."""
        job = self._make_augment_job_stub(multiplier=2)
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        export_root = Path(version["export_root"])
        all_exported_images = list((export_root / "images").rglob("*.jpg"))
        # 4 images total should be exported: img1, img2, img1_aug001, img1_aug002
        self.assertEqual(len(all_exported_images), 4)
        exported_names = {p.name for p in all_exported_images}
        self.assertIn("img1.jpg", exported_names)
        self.assertIn("img2.jpg", exported_names)
        self.assertIn("img1_aug001.jpg", exported_names)
        self.assertIn("img1_aug002.jpg", exported_names)

    def test_create_version_augmented_image_gets_original_annotation(self) -> None:
        """Augmented images inherit their original image's annotation in the snapshot labels dir."""
        job = self._make_augment_job_stub(multiplier=1)
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        source_labels_dir = Path(version["source_root"]) / "labels"
        aug_label = source_labels_dir / "img1_aug001.json"
        self.assertTrue(aug_label.exists(), f"Expected {aug_label} to exist")
        import json as _json
        payload = _json.loads(aug_label.read_text(encoding="utf-8"))
        labels = payload.get("labels") or []
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].get("class"), "obj")

    def test_create_version_missing_augment_output_dir_is_skipped(self) -> None:
        """If the augment job output_dataset_id dir doesn't exist, it's silently skipped."""
        job = {
            "id": "missing-aug",
            "dataset_id": self.dataset["id"],
            "status": "completed",
            "transforms": ["brightness"],
            "multiplier": 1,
            "output_dataset_id": "/nonexistent/path/to/augmented/missing-aug",
            "augmented_image_count": 0,
        }
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        # No augmented images should be added
        self.assertEqual(version["augmented_image_count_in_version"], 0)
        self.assertEqual(version["image_count"], 2)
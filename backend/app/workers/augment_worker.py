"""augment_worker.py — background daemon that performs image augmentation jobs.

Supported transforms (pass by name in ``transforms`` list):
    flip_h         — horizontal flip
    flip_v         — vertical flip
    brightness     — random brightness ±40
    contrast       — random contrast (0.7–1.3)
    blur           — Gaussian blur (kernel 3–7)
    rotate         — random rotation ±15°
    noise          — additive Gaussian noise (σ≈15)

Lineage is stored in the job record:
    source_image_count    — number of input images processed
    augmented_image_count — number of output images written
    output_dataset_id     — sub-folder inside the source dataset (``augmented/``)
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 5
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _apply_transforms(img, transforms: list[str]):  # type: ignore[return]
    import random

    import cv2
    import numpy as np

    for t in transforms:
        if t == "flip_h":
            img = cv2.flip(img, 1)
        elif t == "flip_v":
            img = cv2.flip(img, 0)
        elif t == "brightness":
            delta = random.randint(-40, 40)
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=delta)
        elif t == "contrast":
            alpha = random.uniform(0.7, 1.3)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        elif t == "blur":
            k = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)
        elif t == "rotate":
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        elif t == "noise":
            sigma = 15
            noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class AugmentWorker:
    def __init__(self, augment_repo, datasets_repo) -> None:
        self._augment_repo = augment_repo
        self._datasets_repo = datasets_repo
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="augment-worker", daemon=True)
        self._thread.start()
        logger.info("[augment-worker] started")

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process_queue()
            except Exception:  # noqa: BLE001
                logger.exception("[augment-worker] unhandled error in loop")
            self._stop_event.wait(timeout=_POLL_INTERVAL)

    def _process_queue(self) -> None:
        for job in self._augment_repo.list_jobs():
            if job["status"] == "queued":
                self._run_job(job)

    def _run_job(self, job: dict) -> None:
        import cv2  # import here so server starts even without OpenCV

        job_id = job["id"]
        logger.info("[augment-worker] starting job %s", job_id)
        try:
            self._augment_repo.transition(
                job_id,
                "running",
                log_line=f"Job started at {datetime.now(UTC).isoformat()}",
            )
        except ValueError:
            return

        dataset_id = job["dataset_id"]
        transforms = job.get("transforms") or ["flip_h"]
        multiplier = max(1, int(job.get("multiplier") or 2))

        dataset_dir = self._datasets_repo.dataset_dir(dataset_id)
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            self._augment_repo.transition(
                job_id,
                "failed",
                error=f"images directory not found: {images_dir}",
                log_line=f"Failed: images dir {images_dir} does not exist.",
            )
            return

        # Output goes to a sub-folder named after the job inside the same dataset
        output_dir = dataset_dir / "augmented" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        source_count = len(image_files)
        augmented_count = 0

        try:
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                for i in range(multiplier):
                    out_img = _apply_transforms(img.copy(), transforms)
                    stem = img_path.stem
                    out_name = f"{stem}_aug{i + 1:03d}{img_path.suffix}"
                    cv2.imwrite(str(output_dir / out_name), out_img)
                    augmented_count += 1

            self._augment_repo.transition(
                job_id,
                "completed",
                source_image_count=source_count,
                augmented_image_count=augmented_count,
                output_dataset_id=str(output_dir),
                log_line=(
                    f"Completed: {source_count} source images → {augmented_count} augmented images. "
                    f"Transforms: {transforms}. Output: {output_dir}"
                ),
            )
            logger.info("[augment-worker] job %s completed (%d → %d)", job_id, source_count, augmented_count)

        except ValueError:
            pass  # cancelled
        except Exception as exc:  # noqa: BLE001
            logger.exception("[augment-worker] job %s failed: %s", job_id, exc)
            try:
                self._augment_repo.transition(
                    job_id,
                    "failed",
                    error=str(exc),
                    log_line=f"Failed with exception: {exc}",
                )
            except ValueError:
                pass

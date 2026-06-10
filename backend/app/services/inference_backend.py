"""Shared inference backend helpers.

Plan #13 — Full Abstraction Backend Inference.

This module provides:
- InferenceBackend abstract base class
- Shared helper functions for YOLO output parsing, NMS, and label mapping
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

__all__ = [
    "InferenceBackend",
    "_parse_yolo_output",
    "_apply_nms",
    "_apply_names_map",
]


# ──────────────────────────────────────────────────────────────────
# Shared helpers (used by all non-Ultralytics backends)
# ──────────────────────────────────────────────────────────────────

def _parse_yolo_output(
    out: np.ndarray,
    conf_threshold: float,
    pad_left: int,
    pad_top: int,
    w_in: int,
    h_in: int,
) -> list[tuple[float, int, float, float, float, float]]:
    """Parse YOLOv11 output array into raw candidate detections.

    Handles both [N, 6] and [6, N] layouts (auto-transposes if needed).

    Returns list of (conf, class_id, x1, y1, x2, y2) with coords normalized
    to original image space (de-projected from padded model input).
    """
    candidates: list[tuple[float, int, float, float, float, float]] = []
    if out is None or out.size == 0:
        return candidates

    if out.ndim == 2:
        if out.shape[0] < out.shape[1]:
            out = out.T
        w_content = w_in - 2 * pad_left
        h_content = h_in - 2 * pad_top
        for row in out:
            class_scores = row[4:]
            conf = float(np.max(class_scores))
            if conf < conf_threshold:
                continue
            class_id = int(np.argmax(class_scores))
            xc, yc, w_b, h_b = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
            )
            x1_padded = (xc - w_b / 2) * w_in
            y1_padded = (yc - h_b / 2) * h_in
            x2_padded = (xc + w_b / 2) * w_in
            y2_padded = (yc + h_b / 2) * h_in
            x1 = max(0.0, (x1_padded - pad_left) / w_content) if w_content > 0 else 0.0
            y1 = max(0.0, (y1_padded - pad_top) / h_content) if h_content > 0 else 0.0
            x2 = min(1.0, (x2_padded - pad_left) / w_content) if w_content > 0 else 1.0
            y2 = min(1.0, (y2_padded - pad_top) / h_content) if h_content > 0 else 1.0
            candidates.append((conf, class_id, x1, y1, x2, y2))
    return candidates


def _apply_nms(
    candidates: list[tuple[float, int, float, float, float, float]],
    conf_threshold: float,
) -> list[dict[str, Any]]:
    """Apply Non-Maximum Suppression to candidate detections.

    Returns list of detection dicts with normalized position coords.
    """
    if not candidates:
        return []
    roi_h, roi_w = 1.0, 1.0  # will be scaled by caller
    boxes_cv = [[c[2], c[3], c[4] - c[2], c[5] - c[3]] for c in candidates]
    scores_cv = [c[0] for c in candidates]
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores_cv, conf_threshold, 0.45)
    detections: list[dict[str, Any]] = []
    if len(indices) > 0:
        for idx in indices.flatten():
            conf, class_id, x1, y1, x2, y2 = candidates[int(idx)]
            detections.append(
                {
                    "label": str(class_id),
                    "confidence": round(conf, 4),
                    "class_confidence": round(conf, 4),
                    "class_id": class_id,
                    "position": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return detections


def _apply_names_map(
    detections: list[dict[str, Any]],
    names_map: dict[int, str],
    allowed_labels: set[str] | None,
) -> list[dict[str, Any]]:
    """Map class_id → label name and filter by allowed_labels.

    When names_map is empty (no meta file), labels stay as string class_ids
    and label-based filtering is skipped (returns all detections).
    """
    filtered: list[dict[str, Any]] = []
    for det in detections:
        label = names_map.get(det["class_id"], str(det["class_id"]))
        det["label"] = label
        # Only filter by label when class mapping is available
        if names_map and allowed_labels is not None:
            if label.strip().lower() not in allowed_labels:
                continue
        filtered.append(det)
    return filtered


# ──────────────────────────────────────────────────────────────────
# Abstract base class
# ──────────────────────────────────────────────────────────────────

class InferenceBackend(ABC):
    """Abstract interface for sticker detection inference backends."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        vision: Any,
        expected_class: str | None = None,
    ) -> dict[str, Any]:
        """Run inference on an image and return detection results.

        Returns dict with keys:
            backend, mode, model_path, meta_path, class_names, detections,
            raw_detection_count, allowed_labels_filter, fallback_reason,
            device_mode, effective_device, device_backend,
            device_fallback_reason, gpu_available
        """
        ...

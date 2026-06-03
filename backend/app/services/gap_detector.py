"""Gap detection service — part ready via template matching.

Detects the blue clamp using HSV segmentation, extracts the gap area as a
template patch from a reference image, and uses cv2.matchTemplate to confirm
the part is at the correct position within the part_ready ROI.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.config import PROJECT_ROOT as project_root

# Default HSV range for blue clamp detection
DEFAULT_HSV_LOWER = np.array([90, 50, 50])
DEFAULT_HSV_UPPER = np.array([130, 255, 255])

# Reference storage directory
PART_READY_REF_DIR = "backend/app/assets/part_ready_refs"


def get_ref_path(template_id: int) -> Path:
    """Get the file path for a template's reference patch."""
    ref_dir = Path(project_root) / PART_READY_REF_DIR
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir / f"{template_id}.png"


def load_ref_patch(ref_path: str | None, template_id: int | None = None) -> np.ndarray | None:
    """Load reference patch PNG from disk. Returns None if file missing."""
    if ref_path:
        p = Path(ref_path)
        if p.is_file():
            img = cv2.imread(str(p))
            return img if img is not None else None
    # Fallback to standard path
    if template_id is not None:
        p = get_ref_path(template_id)
        if p.is_file():
            img = cv2.imread(str(p))
            return img if img is not None else None
    return None


def save_ref_patch(frame_bgr: np.ndarray, roi: dict, save_path: str,
                   hsv_lower: np.ndarray = DEFAULT_HSV_LOWER,
                   hsv_upper: np.ndarray = DEFAULT_HSV_UPPER,
                   padding_px: int = 20,
                   rotation: float = 0.0) -> bool:
    """Extract and save reference gap patch from a calibration frame.

    1. Crop the ROI region from the frame.
    2. HSV-segment the blue clamp.
    3. Find the lower edge of the clamp mask.
    4. Extract the gap area below the clamp.
    5. Save as PNG.
    """
    try:
        rx, ry = int(roi.get("x", 0)), int(roi.get("y", 0))
        rw, rh = int(roi.get("w", 0)), int(roi.get("h", 0))
        if rw <= 0 or rh <= 0:
            return False
        # Clamp to frame bounds
        fh, fw = frame_bgr.shape[:2]
        rx = max(0, min(rx, fw - 1))
        ry = max(0, min(ry, fh - 1))
        rw = min(rw, fw - rx)
        rh = min(rh, fh - ry)
        roi_frame = frame_bgr[ry:ry+rh, rx:rx+rw]
        if roi_frame.size == 0:
            return False

        # Deskew rotated ROI before HSV extraction
        if abs(rotation) > 0.1:
            center = (rw / 2, rh / 2)
            M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            roi_frame = cv2.warpAffine(
                roi_frame, M, (rw, rh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        # HSV segmentation for blue clamp
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find clamp lower edge
        ys = np.any(mask > 0, axis=1)
        if not np.any(ys):
            return False
        clamp_bottom = np.max(np.where(ys)[0])

        # Extract gap area: from clamp_bottom + padding to bottom of ROI
        gap_start = min(clamp_bottom + padding_px, rh - 1)
        gap_end = rh
        if gap_end - gap_start < 10:
            return False

        gap_patch = roi_frame[gap_start:gap_end, :]
        if gap_patch.size == 0:
            return False

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, gap_patch)
        return True
    except Exception:
        return False


def match_gap(frame_bgr: np.ndarray, roi: dict, ref_patch: np.ndarray,
              threshold: float = 0.85) -> dict[str, Any]:
    """Run cv2.matchTemplate of ref_patch on the ROI region.

    Returns:
        {"match": bool, "score": float, "location": (x, y)} — all native Python types
    """
    try:
        rx, ry = int(roi.get("x", 0)), int(roi.get("y", 0))
        rw, rh = int(roi.get("w", 0)), int(roi.get("h", 0))
        if rw <= 0 or rh <= 0 or ref_patch is None or ref_patch.size == 0:
            return {"match": False, "score": 0.0, "location": (0, 0)}

        fh, fw = frame_bgr.shape[:2]
        rx = max(0, min(rx, fw - 1))
        ry = max(0, min(ry, fh - 1))
        rw = min(rw, fw - rx)
        rh = min(rh, fh - ry)
        roi_frame = frame_bgr[ry:ry+rh, rx:rx+rw]
        if roi_frame.size == 0:
            return {"match": False, "score": 0.0, "location": (0, 0)}

        # Ref patch must fit inside ROI
        ph, pw = ref_patch.shape[:2]
        if ph > rh or pw > rw:
            # Resize ref patch to fit
            scale = min(rh / max(ph, 1), rw / max(pw, 1))
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            ref_patch = cv2.resize(ref_patch, (new_w, new_h))
            ph, pw = ref_patch.shape[:2]

        # Template matching
        result = cv2.matchTemplate(roi_frame, ref_patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        return {
            "match": bool(max_val >= threshold),
            "score": float(round(max_val, 4)),
            "location": (int(max_loc[0]), int(max_loc[1])),
        }
    except Exception:
        return {"match": False, "score": 0.0, "location": (0, 0)}

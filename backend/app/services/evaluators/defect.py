from __future__ import annotations

import io
import base64
from typing import Any

import cv2
import numpy as np

from shared.contracts.decision import Decision
from backend.app.services.evaluators.base import ModeEvaluator, EvalContext
from backend.app.services.anomaly_backend import get_scorer


class DefectEvaluator(ModeEvaluator):
    """Evaluator for anomaly-based defect detection mode.

    For each ROI in criteria.rois:
      1. Crop frame using geometry (x, y, w, h as fractions)
      2. Score with AnomalyScorer
      3. ok = score <= roi.threshold
    Decision.accept = all ROIs pass.

    Heatmaps are stored as base64-encoded JPEG in details (small).
    """

    mode_name = "defect"

    def evaluate(self, ctx: EvalContext) -> Decision:
        rois = ctx.criteria.get("rois", [])
        if not rois:
            return Decision.rejected("NO_DEFECT_ROIS", {
                "mode": "defect",
                "error": "No defect ROIs defined",
            })

        frame = ctx.frame
        if frame is None or frame.size == 0:
            return Decision.rejected("NO_FRAME", {
                "mode": "defect",
                "error": "No frame provided",
            })

        fh, fw = frame.shape[:2]
        all_ok = True
        roi_results = []
        reject_reason = None

        for i, roi in enumerate(rois):
            name = roi.get("name", f"ROI {i+1}")

            # Parse geometry (fractional coordinates)
            geom = roi.get("geometry", {})
            x = int(float(geom.get("x", 0.0)) * fw)
            y = int(float(geom.get("y", 0.0)) * fh)
            w = max(1, int(float(geom.get("w", 1.0)) * fw))
            h = max(1, int(float(geom.get("h", 1.0)) * fh))
            # Clamp to frame bounds
            x = max(0, min(fw - 1, x))
            y = max(0, min(fh - 1, y))
            w = min(fw - x, w)
            h = min(fh - y, h)

            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                roi_results.append({
                    "name": name,
                    "ok": False,
                    "score": float("inf"),
                    "threshold": roi.get("threshold", 0.5),
                    "error": "empty crop",
                })
                all_ok = False
                if not reject_reason:
                    reject_reason = "ANOMALY_DETECTED"
                continue

            # Get scorer (per-ROI model_path or default)
            model_path = roi.get("model_path") or ctx.criteria.get("default_model_path")
            scorer = get_scorer(model_path)

            try:
                score, heatmap = scorer.score(crop)
            except Exception as exc:
                logger = __import__("logging").getLogger(__name__)
                logger.error("[defect] ROI %s score failed: %s", name, exc)
                roi_results.append({
                    "name": name,
                    "ok": False,
                    "score": 0.0,
                    "threshold": roi.get("threshold", 0.5),
                    "error": str(exc),
                })
                all_ok = False
                if not reject_reason:
                    reject_reason = "ANOMALY_DETECTED"
                continue

            threshold = float(roi.get("threshold", 0.5))
            ok = score <= threshold

            # Encode heatmap as small base64 JPEG (optional)
            heatmap_b64 = None
            if heatmap is not None and heatmap.size > 0:
                try:
                    # Normalize heatmap to 0-255 uint8
                    hm_norm = heatmap.copy()
                    hm_min, hm_max = float(hm_norm.min()), float(hm_norm.max())
                    if hm_max > hm_min:
                        hm_norm = (hm_norm - hm_min) / (hm_max - hm_min) * 255
                    hm_uint8 = hm_norm.astype(np.uint8)
                    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                    ok_flag, hm_buf = cv2.imencode(".jpg", hm_colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok_flag:
                        heatmap_b64 = base64.b64encode(hm_buf.tobytes()).decode("ascii")
                except Exception:
                    pass

            roi_results.append({
                "name": name,
                "ok": ok,
                "score": round(float(score), 4),
                "threshold": threshold,
                "heatmap_b64": heatmap_b64,
            })

            if not ok:
                all_ok = False
                if not reject_reason:
                    reject_reason = "ANOMALY_DETECTED"

        details = {
            "mode": "defect",
            "rois": roi_results,
        }

        if all_ok:
            return Decision.accepted(details)
        return Decision.rejected(reject_reason or "ANOMALY_DETECTED", details)

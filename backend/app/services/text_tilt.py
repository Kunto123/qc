from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _hsv_bounds(values: Any, fallback: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return np.array(fallback, dtype=np.uint8)
    parsed: list[int] = []
    for index, value in enumerate(values):
        upper = 180 if index == 0 else 255
        try:
            parsed_value = int(value)
        except (TypeError, ValueError):
            parsed_value = fallback[index]
        parsed.append(max(0, min(upper, parsed_value)))
    return np.array(parsed, dtype=np.uint8)


def _normalize_rect_angle(rect) -> float:
    width, height = rect[1]
    angle = float(rect[2])
    if width < height:
        angle = 90.0 + angle
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return angle


def estimate_white_text_tilt(roi_frame, expected_tilt_degrees: float = 0.0, config: Any | None = None) -> dict[str, Any]:
    if roi_frame is None or getattr(roi_frame, "size", 0) == 0:
        return {
            "status": "unavailable",
            "angle_degrees": None,
            "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
            "deviation_degrees": None,
            "contour_area": None,
            "threshold_mode": None,
        }

    height, width = roi_frame.shape[:2]
    lower = _hsv_bounds(getattr(config, "white_hsv_lower", None), (0, 0, 160))
    upper = _hsv_bounds(getattr(config, "white_hsv_upper", None), (180, 70, 255))
    min_area_ratio = float(getattr(config, "min_text_contour_area_ratio", 0.002) or 0.002)
    min_area = max(1.0, float(height * width) * min_area_ratio)

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    if not contours:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fallback_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(fallback_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
        threshold_mode = "otsu"
    else:
        threshold_mode = "hsv_white"

    if not contours:
        return {
            "status": "unavailable",
            "angle_degrees": None,
            "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
            "deviation_degrees": None,
            "contour_area": None,
            "threshold_mode": threshold_mode,
        }

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    rect = cv2.minAreaRect(contour)
    angle = _normalize_rect_angle(rect)
    deviation = abs(angle - float(expected_tilt_degrees))
    return {
        "status": "ok",
        "angle_degrees": round(angle, 2),
        "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
        "deviation_degrees": round(deviation, 2),
        "contour_area": round(area, 2),
        "threshold_mode": threshold_mode,
        "min_contour_area": round(min_area, 2),
    }

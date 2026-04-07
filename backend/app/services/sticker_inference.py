from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import cv2

from backend.app.core.config import AppConfig
from backend.app.repositories.models_repository import ModelsRepository
from shared.contracts.templates import VisionConfig


class StickerInferenceService:
    def __init__(self, app_config: AppConfig, models_repo: ModelsRepository) -> None:
        self._config = app_config
        self._models_repo = models_repo
        self._runtime_lock = threading.RLock()
        self._loaded_models: dict[str, Any] = {}
        self._meta_cache: dict[str, dict[str, Any]] = {}

    def _resolve_mode(self) -> str:
        mode = str(self._config.sticker_inference_mode or "auto").strip().lower()
        return mode if mode in {"auto", "ultralytics", "classic"} else "auto"

    def _resolve_model_path(self, vision: VisionConfig) -> str:
        direct = str(vision.model_path or "").strip()
        if direct:
            return direct
        return str(self._config.default_sticker_model_path or "").strip()

    def _resolve_meta_path(self, vision: VisionConfig) -> str:
        direct = str(vision.model_meta_path or "").strip()
        if direct:
            return direct
        model_record = self._models_repo.find_by_path(self._resolve_model_path(vision))
        if model_record and model_record.get("meta_path"):
            return str(model_record["meta_path"]).strip()
        return str(self._config.default_sticker_model_meta_path or "").strip()

    def _load_meta(self, meta_path: str) -> dict[str, Any]:
        if not meta_path:
            return {}
        with self._runtime_lock:
            cached = self._meta_cache.get(meta_path)
            if cached is not None:
                return cached
            path = Path(meta_path)
            if not path.exists():
                self._meta_cache[meta_path] = {}
                return {}
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            self._meta_cache[meta_path] = payload
            return payload

    def _load_yolo_class(self):
        from ultralytics import YOLO  # type: ignore

        return YOLO

    def _get_ultralytics_model(self, model_path: str):
        resolved = str(Path(model_path).resolve())
        with self._runtime_lock:
            model = self._loaded_models.get(resolved)
            if model is not None:
                return model
            YOLO = self._load_yolo_class()
            model = YOLO(resolved)
            self._loaded_models[resolved] = model
            return model

    def _normalize_detections(
        self,
        *,
        result,
        names: dict[int, str] | dict[Any, Any],
        allowed_labels: set[str] | None,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id))
            if allowed_labels is not None and label.lower() not in allowed_labels:
                continue
            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "class_confidence": round(confidence, 4),
                    "class_id": class_id,
                    "position": {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                    },
                }
            )
        return detections

    def _predict_ultralytics(self, image, vision: VisionConfig) -> dict[str, Any]:
        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Sticker model path is not configured.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Sticker model not found: {resolved_model_path}")

        model = self._get_ultralytics_model(str(resolved_model_path))
        kwargs: dict[str, Any] = {
            "verbose": False,
            "conf": float(vision.conf_threshold),
        }
        if int(vision.imgsz or 0) > 0:
            kwargs["imgsz"] = int(vision.imgsz)
        result = model.predict(image, **kwargs)[0]
        names = result.names or {}
        raw_box_count = int(len(result.boxes)) if result.boxes is not None else 0
        allowed_labels = {str(label).strip().lower() for label in (vision.classes or [])} or None
        detections = self._normalize_detections(result=result, names=names, allowed_labels=allowed_labels)
        return {
            "backend": "ultralytics",
            "mode": "ultralytics",
            "model_path": str(resolved_model_path),
            "meta_path": self._resolve_meta_path(vision) or None,
            "class_names": list(self._load_meta(self._resolve_meta_path(vision)).get("class_names") or []),
            "detections": detections,
            "raw_detection_count": raw_box_count,
            "allowed_labels_filter": sorted(allowed_labels) if allowed_labels is not None else None,
            "fallback_reason": None,
        }

    def _predict_classic(self, image, vision: VisionConfig, expected_class: str | None) -> dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "backend": "classic",
                "mode": "classic",
                "model_path": self._resolve_model_path(vision) or None,
                "meta_path": self._resolve_meta_path(vision) or None,
                "class_names": list(vision.classes or []),
                "detections": [],
                "fallback_reason": None,
            }
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = float((w * h) / max(1, image.shape[0] * image.shape[1]))
        confidence = max(0.1, min(0.99, area_ratio + 0.2))
        label = str(expected_class or (vision.classes[0] if vision.classes else "sticker")).strip() or "sticker"
        return {
            "backend": "classic",
            "mode": "classic",
            "model_path": self._resolve_model_path(vision) or None,
            "meta_path": self._resolve_meta_path(vision) or None,
            "class_names": list(vision.classes or []),
            "detections": [
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "class_confidence": round(confidence, 4),
                    "class_id": None,
                    "position": {"x1": float(x), "y1": float(y), "x2": float(x + w), "y2": float(y + h)},
                }
            ],
            "fallback_reason": None,
        }

    def predict(self, image, vision: VisionConfig, *, expected_class: str | None = None) -> dict[str, Any]:
        if image is None or image.size == 0:
            return {
                "backend": "none",
                "mode": self._resolve_mode(),
                "model_path": self._resolve_model_path(vision) or None,
                "meta_path": self._resolve_meta_path(vision) or None,
                "class_names": list(vision.classes or []),
                "detections": [],
                "fallback_reason": "empty_roi",
            }

        mode = self._resolve_mode()
        if mode == "classic":
            return self._predict_classic(image, vision, expected_class)

        try:
            return self._predict_ultralytics(image, vision)
        except Exception as exc:
            if mode == "ultralytics":
                raise
            logging.warning("Sticker inference fallback to classic mode: %s", exc)
            payload = self._predict_classic(image, vision, expected_class)
            payload["fallback_reason"] = str(exc)
            return payload

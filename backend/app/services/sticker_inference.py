from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceResolution, DeviceRuntimeResolver
from backend.app.repositories.models_repository import ModelsRepository
from shared.contracts.templates import StickerRule
from shared.contracts.templates import VisionConfig


_TEXT_ANCHOR_ALIASES = {
    "textanchor",
    "anchortext",
    "text",
    "ocranchor",
    "whitetext",
    "labeltext",
}
_CENTER_DOT_ALIASES = {
    "centerdot",
    "dot",
    "centerpoint",
    "referencedot",
    "referencepoint",
}


class StickerInferenceService:
    def __init__(
        self,
        app_config: AppConfig,
        models_repo: ModelsRepository,
        device_runtime: DeviceRuntimeResolver | None = None,
    ) -> None:
        self._config = app_config
        self._models_repo = models_repo
        self._device_runtime = device_runtime or DeviceRuntimeResolver(app_config)
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

    def _resolve_device(self) -> DeviceResolution:
        return self._device_runtime.resolve()

    def _resolve_ocr_engine(self, vision: VisionConfig) -> str:
        engine = str(getattr(vision, "ocr_engine", "") or "").strip().lower()
        if engine in {"", "default", "auto"}:
            engine = str(self._config.default_ocr_engine or "disabled").strip().lower()
        return engine or "disabled"

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
        allowed_label_keys: set[str] | None,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id))
            if allowed_labels is not None:
                normalized_label = label.strip().lower()
                normalized_key = self._normalize_label_key(label)
                if normalized_label not in allowed_labels and (
                    allowed_label_keys is None or normalized_key not in allowed_label_keys
                ):
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

    @staticmethod
    def _normalize_label_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        return "".join(ch for ch in text if ch.isalnum())

    def _predict_ultralytics(self, image, vision: VisionConfig, device_resolution: DeviceResolution) -> dict[str, Any]:
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
            "device": device_resolution.effective_device,
        }
        if int(vision.imgsz or 0) > 0:
            kwargs["imgsz"] = int(vision.imgsz)
        result = model.predict(image, **kwargs)[0]
        names = result.names or {}
        raw_box_count = int(len(result.boxes)) if result.boxes is not None else 0
        allowed_label_values = [str(label) for label in (vision.classes or []) if str(label).strip()]
        if allowed_label_values:
            for label in (getattr(vision, "text_anchor_class", ""), getattr(vision, "center_dot_class", "")):
                if str(label or "").strip():
                    allowed_label_values.append(str(label))
        allowed_labels = {label.strip().lower() for label in allowed_label_values} or None
        allowed_label_keys = ({self._normalize_label_key(label) for label in allowed_label_values if self._normalize_label_key(label)} or None)
        detections = self._normalize_detections(
            result=result,
            names=names,
            allowed_labels=allowed_labels,
            allowed_label_keys=allowed_label_keys,
        )
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
            "device_mode": device_resolution.requested_mode,
            "effective_device": device_resolution.effective_device,
            "device_backend": device_resolution.backend,
            "device_fallback_reason": device_resolution.fallback_reason,
            "gpu_available": device_resolution.gpu_available,
        }

    def _select_anchor_detection(
        self,
        detections: list[dict[str, Any]],
        *,
        labels: set[str],
    ) -> dict[str, Any] | None:
        candidates = [
            item
            for item in detections
            if self._normalize_label_key(item.get("label")) in labels
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: float(item.get("confidence") or 0.0))

    @staticmethod
    def _bbox_center(position: dict[str, Any] | None) -> dict[str, float] | None:
        if not position:
            return None
        try:
            x1 = float(position.get("x1", 0.0))
            y1 = float(position.get("y1", 0.0))
            x2 = float(position.get("x2", 0.0))
            y2 = float(position.get("y2", 0.0))
        except (TypeError, ValueError):
            return None
        return {"x": round((x1 + x2) / 2.0, 2), "y": round((y1 + y2) / 2.0, 2)}

    @staticmethod
    def _round_bbox(position: dict[str, Any] | None) -> dict[str, float] | None:
        if not position:
            return None
        return {
            "x1": round(float(position.get("x1", 0.0)), 2),
            "y1": round(float(position.get("y1", 0.0)), 2),
            "x2": round(float(position.get("x2", 0.0)), 2),
            "y2": round(float(position.get("y2", 0.0)), 2),
        }

    def _build_anchor_payload(
        self,
        detections: list[dict[str, Any]],
        vision: VisionConfig,
    ) -> dict[str, Any]:
        text_labels = {
            self._normalize_label_key(getattr(vision, "text_anchor_class", "text_anchor")),
            *_TEXT_ANCHOR_ALIASES,
        }
        dot_labels = {
            self._normalize_label_key(getattr(vision, "center_dot_class", "center_dot")),
            *_CENTER_DOT_ALIASES,
        }
        text_anchor = self._select_anchor_detection(detections, labels={item for item in text_labels if item})
        dot_anchor = self._select_anchor_detection(detections, labels={item for item in dot_labels if item})
        text_bbox = self._round_bbox((text_anchor or {}).get("position"))
        dot_bbox = self._round_bbox((dot_anchor or {}).get("position"))
        return {
            "status": "ok" if text_anchor is not None and dot_anchor is not None else "missing_anchor",
            "text_anchor": text_anchor,
            "center_dot": dot_anchor,
            "text_bbox": text_bbox,
            "dot_bbox": dot_bbox,
            "text_position": self._bbox_center(text_bbox),
            "dot_position": self._bbox_center(dot_bbox),
            "text_confidence": None if text_anchor is None else round(float(text_anchor.get("confidence") or 0.0), 4),
            "dot_confidence": None if dot_anchor is None else round(float(dot_anchor.get("confidence") or 0.0), 4),
            "text_anchor_class": getattr(vision, "text_anchor_class", "text_anchor"),
            "center_dot_class": getattr(vision, "center_dot_class", "center_dot"),
        }

    def _crop_text_anchor(self, image, text_bbox: dict[str, Any] | None, vision: VisionConfig):
        if image is None or getattr(image, "size", 0) == 0 or not text_bbox:
            return None
        height, width = image.shape[:2]
        x1 = float(text_bbox.get("x1", 0.0))
        y1 = float(text_bbox.get("y1", 0.0))
        x2 = float(text_bbox.get("x2", 0.0))
        y2 = float(text_bbox.get("y2", 0.0))
        pad_ratio = max(0.0, float(getattr(vision, "anchor_crop_padding_ratio", 0.08) or 0.0))
        pad_x = (x2 - x1) * pad_ratio
        pad_y = (y2 - y1) * pad_ratio
        ix1 = max(0, int(round(x1 - pad_x)))
        iy1 = max(0, int(round(y1 - pad_y)))
        ix2 = min(width, int(round(x2 + pad_x)))
        iy2 = min(height, int(round(y2 + pad_y)))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return image[iy1:iy2, ix1:ix2]

    def _preprocess_ocr_crop(self, crop, vision: VisionConfig):
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        scale = max(1.0, float(getattr(vision, "anchor_crop_scale", 2.0) or 1.0))
        if scale > 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    def _ocr_with_tesseract(
        self,
        image,
        vision: VisionConfig,
        *,
        expected_text: str | None,
        regex: str | None,
        canonical_map: dict[str, str],
    ) -> dict[str, Any]:
        try:
            import pytesseract  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "unavailable",
                "engine": "tesseract",
                "text": "",
                "raw_text": "",
                "canonical_text": "",
                "confidence": None,
                "expected_text": expected_text,
                "match_expected": False,
                "error": str(exc),
            }

        config_parts = [f"--psm {int(getattr(vision, 'ocr_psm', 7) or 7)}"]
        allowlist = str(getattr(vision, "ocr_allowlist", "") or "").strip()
        if allowlist:
            config_parts.append(f"-c tessedit_char_whitelist={allowlist}")
        config = " ".join(config_parts)
        try:
            data = pytesseract.image_to_data(
                image,
                lang=str(getattr(vision, "ocr_language", "eng") or "eng"),
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "engine": "tesseract",
                "text": "",
                "raw_text": "",
                "canonical_text": "",
                "confidence": None,
                "expected_text": expected_text,
                "match_expected": False,
                "error": str(exc),
            }

        words: list[str] = []
        confidences: list[float] = []
        for word, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
            cleaned = str(word or "").strip()
            if cleaned:
                words.append(cleaned)
            try:
                confidence_value = float(conf)
            except (TypeError, ValueError):
                continue
            if confidence_value >= 0:
                confidences.append(confidence_value / 100.0)

        raw_text = " ".join(words).strip()
        confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        canonical_text = self.normalize_ocr_text(
            raw_text,
            expected_text=expected_text,
            regex=regex,
            canonical_map=canonical_map,
        )
        expected_key = self._normalize_label_key(expected_text)
        canonical_key = self._normalize_label_key(canonical_text)
        return {
            "status": "ok" if raw_text else "empty_text",
            "engine": "tesseract",
            "text": canonical_text,
            "raw_text": raw_text,
            "canonical_text": canonical_text,
            "confidence": confidence,
            "expected_text": expected_text,
            "match_expected": bool(expected_key and canonical_key == expected_key),
            "error": None,
        }

    def _ocr_passthrough(
        self,
        anchor: dict[str, Any],
        *,
        expected_text: str | None,
        regex: str | None,
        canonical_map: dict[str, str],
    ) -> dict[str, Any]:
        source = anchor.get("text_anchor") or {}
        raw_text = str(
            source.get("ocr_text")
            or source.get("text")
            or source.get("raw_text")
            or ""
        ).strip()
        confidence = source.get("ocr_confidence")
        if confidence is None:
            confidence = source.get("confidence")
        canonical_text = self.normalize_ocr_text(
            raw_text,
            expected_text=expected_text,
            regex=regex,
            canonical_map=canonical_map,
        )
        expected_key = self._normalize_label_key(expected_text)
        canonical_key = self._normalize_label_key(canonical_text)
        return {
            "status": "ok" if raw_text else "empty_text",
            "engine": "passthrough",
            "text": canonical_text,
            "raw_text": raw_text,
            "canonical_text": canonical_text,
            "confidence": None if confidence is None else round(float(confidence), 4),
            "expected_text": expected_text,
            "match_expected": bool(expected_key and canonical_key == expected_key),
            "error": None,
        }

    def _run_ocr(
        self,
        image,
        vision: VisionConfig,
        sticker_rule: StickerRule | None,
        anchor: dict[str, Any],
        *,
        expected_class: str | None,
    ) -> dict[str, Any]:
        expected_text = str(getattr(sticker_rule, "ocr_expected_text", None) or expected_class or "").strip() or None
        regex = getattr(sticker_rule, "ocr_regex", None) if sticker_rule is not None else None
        canonical_map = dict(getattr(sticker_rule, "ocr_canonical_map", {}) or {}) if sticker_rule is not None else {}
        engine = self._resolve_ocr_engine(vision)
        if engine in {"disabled", "none", "off"}:
            return {
                "status": "disabled",
                "engine": engine,
                "text": "",
                "raw_text": "",
                "canonical_text": "",
                "confidence": None,
                "expected_text": expected_text,
                "match_expected": False,
                "error": None,
            }
        if anchor.get("text_anchor") is None:
            return {
                "status": "anchor_not_found",
                "engine": engine,
                "text": "",
                "raw_text": "",
                "canonical_text": "",
                "confidence": None,
                "expected_text": expected_text,
                "match_expected": False,
                "error": None,
            }
        if engine == "passthrough":
            return self._ocr_passthrough(anchor, expected_text=expected_text, regex=regex, canonical_map=canonical_map)

        crop = self._crop_text_anchor(image, anchor.get("text_bbox"), vision)
        prepared = self._preprocess_ocr_crop(crop, vision)
        if prepared is None:
            return {
                "status": "empty_crop",
                "engine": engine,
                "text": "",
                "raw_text": "",
                "canonical_text": "",
                "confidence": None,
                "expected_text": expected_text,
                "match_expected": False,
                "error": None,
            }
        if engine == "tesseract":
            return self._ocr_with_tesseract(
                prepared,
                vision,
                expected_text=expected_text,
                regex=regex,
                canonical_map=canonical_map,
            )
        return {
            "status": "unsupported_engine",
            "engine": engine,
            "text": "",
            "raw_text": "",
            "canonical_text": "",
            "confidence": None,
            "expected_text": expected_text,
            "match_expected": False,
            "error": f"Unsupported OCR engine: {engine}",
        }

    @staticmethod
    def _angle_delta_degrees(value: float, expected: float) -> float:
        return abs((value - expected + 180.0) % 360.0 - 180.0)

    def _build_geometry_payload(
        self,
        image,
        sticker_rule: StickerRule | None,
        anchor: dict[str, Any],
    ) -> dict[str, Any]:
        dot_position = anchor.get("dot_position")
        text_position = anchor.get("text_position")
        if image is None or getattr(image, "size", 0) == 0:
            roi_w = roi_h = 0
        else:
            roi_h, roi_w = image.shape[:2]

        expected_dot_x_ratio = getattr(sticker_rule, "expected_dot_x", None) if sticker_rule is not None else None
        expected_dot_y_ratio = getattr(sticker_rule, "expected_dot_y", None) if sticker_rule is not None else None
        if expected_dot_x_ratio is None:
            expected_dot_x_ratio = getattr(sticker_rule, "expected_center_x", None) if sticker_rule is not None else None
        if expected_dot_y_ratio is None:
            expected_dot_y_ratio = getattr(sticker_rule, "expected_center_y", None) if sticker_rule is not None else None
        expected_dot_x_ratio = 0.5 if expected_dot_x_ratio is None else float(expected_dot_x_ratio)
        expected_dot_y_ratio = 0.5 if expected_dot_y_ratio is None else float(expected_dot_y_ratio)
        expected_dot_position = {
            "x": round(expected_dot_x_ratio * roi_w, 2),
            "y": round(expected_dot_y_ratio * roi_h, 2),
        }

        anchor_offset = None
        if dot_position is not None:
            anchor_offset = {
                "x": round(float(dot_position["x"]) - expected_dot_position["x"], 2),
                "y": round(float(dot_position["y"]) - expected_dot_position["y"], 2),
                "source": "center_dot",
            }
        pose_angle = None
        pose_deviation = None
        if dot_position is not None and text_position is not None:
            dx = float(dot_position["x"]) - float(text_position["x"])
            dy = float(dot_position["y"]) - float(text_position["y"])
            pose_angle = round(float(np.degrees(np.arctan2(dy, dx))), 2)
            expected_tilt = float(getattr(sticker_rule, "expected_tilt_degrees", 0.0) or 0.0) if sticker_rule is not None else 0.0
            pose_deviation = round(self._angle_delta_degrees(float(pose_angle), expected_tilt), 2)
        return {
            "status": "ok" if dot_position is not None and text_position is not None else "missing_anchor",
            "text_position": text_position,
            "dot_position": dot_position,
            "expected_dot_position": expected_dot_position,
            "anchor_offset": anchor_offset,
            "pose_angle": pose_angle,
            "pose_deviation": pose_deviation,
        }

    def _augment_with_anchor_ocr(
        self,
        payload: dict[str, Any],
        image,
        vision: VisionConfig,
        *,
        expected_class: str | None,
        sticker_rule: StickerRule | None,
    ) -> dict[str, Any]:
        anchor_started = time.perf_counter()
        detections = list(payload.get("detections") or [])
        anchor = self._build_anchor_payload(detections, vision)
        geometry = self._build_geometry_payload(image, sticker_rule, anchor)
        anchor_ms = round((time.perf_counter() - anchor_started) * 1000.0, 2)

        ocr_started = time.perf_counter()
        ocr = self._run_ocr(
            image,
            vision,
            sticker_rule,
            anchor,
            expected_class=expected_class,
        )
        ocr_ms = round((time.perf_counter() - ocr_started) * 1000.0, 2)

        payload["anchor"] = anchor
        payload["ocr"] = ocr
        payload["geometry"] = geometry
        payload.setdefault("timings", {})
        payload["timings"].update({"anchor_ms": anchor_ms, "ocr_ms": ocr_ms})
        return payload

    def normalize_ocr_text(
        self,
        value: Any,
        *,
        expected_text: str | None = None,
        regex: str | None = None,
        canonical_map: dict[str, str] | None = None,
    ) -> str:
        import re

        text = str(value or "").strip().upper()
        if regex:
            match = re.search(regex, text)
            if match:
                text = str(match.group(1) if match.groups() else match.group(0)).strip().upper()
        key = self._normalize_label_key(text)
        mapping = {
            self._normalize_label_key(source): str(target).strip().upper()
            for source, target in dict(canonical_map or {}).items()
            if self._normalize_label_key(source)
        }
        if key in mapping:
            return mapping[key]
        expected_key = self._normalize_label_key(expected_text)
        if expected_text and key == expected_key:
            return str(expected_text).strip()
        return text

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

    def predict(
        self,
        image,
        vision: VisionConfig,
        *,
        expected_class: str | None = None,
        sticker_rule: StickerRule | None = None,
    ) -> dict[str, Any]:
        if image is None or image.size == 0:
            return {
                "backend": "none",
                "mode": self._resolve_mode(),
                "model_path": self._resolve_model_path(vision) or None,
                "meta_path": self._resolve_meta_path(vision) or None,
                "class_names": list(vision.classes or []),
                "detections": [],
                "anchor": {},
                "ocr": {"status": "skipped", "engine": self._resolve_ocr_engine(vision)},
                "geometry": {},
                "fallback_reason": "empty_roi",
            }

        mode = self._resolve_mode()
        if mode == "classic":
            payload = self._predict_classic(image, vision, expected_class)
            return self._augment_with_anchor_ocr(
                payload,
                image,
                vision,
                expected_class=expected_class,
                sticker_rule=sticker_rule,
            )

        device_resolution = self._resolve_device()
        try:
            payload = self._predict_ultralytics(image, vision, device_resolution)
            return self._augment_with_anchor_ocr(
                payload,
                image,
                vision,
                expected_class=expected_class,
                sticker_rule=sticker_rule,
            )
        except Exception as exc:
            if mode == "ultralytics":
                raise
            logging.warning("Sticker inference fallback to classic mode: %s", exc)
            payload = self._predict_classic(image, vision, expected_class)
            payload["fallback_reason"] = str(exc)
            payload["device_mode"] = device_resolution.requested_mode
            payload["effective_device"] = device_resolution.effective_device
            payload["device_backend"] = device_resolution.backend
            payload["device_fallback_reason"] = device_resolution.fallback_reason or str(exc)
            payload["gpu_available"] = device_resolution.gpu_available
            return self._augment_with_anchor_ocr(
                payload,
                image,
                vision,
                expected_class=expected_class,
                sticker_rule=sticker_rule,
            )

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceResolution, DeviceRuntimeResolver
from backend.app.repositories.models_repository import ModelsRepository
from backend.app.services.text_tilt import estimate_white_text_tilt
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
        return mode if mode in {"auto", "ultralytics", "classic", "tflite"} else "auto"

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

    def _load_tflite_interpreter(self, model_path: str):
        """Load TFLite model via tflite_runtime (lightweight, no tensorflow needed)."""
        resolved = str(Path(model_path).resolve())
        with self._runtime_lock:
            interp = self._loaded_models.get(resolved)
            if interp is not None:
                return interp
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
            interpreter = Interpreter(model_path=resolved)
        except ImportError:
            raise ModuleNotFoundError(
                "tflite-runtime is required for TFLite inference. "
                "Install it with: pip install tflite-runtime"
            )
        interpreter.allocate_tensors()
        with self._runtime_lock:
            self._loaded_models[resolved] = interpreter
        return interpreter

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
            model = YOLO(resolved, task="detect")
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

    def _predict_ultralytics(self, image, vision: VisionConfig, device_resolution: DeviceResolution, expected_class: str | None = None) -> dict[str, Any]:
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
        # Build allowed labels from vision.classes + anchor classes + expected_class
        allowed_label_values = [str(label) for label in (vision.classes or []) if str(label).strip()]
        # Always include expected_class (from template sticker.expected_class)
        if expected_class and str(expected_class).strip():
            allowed_label_values.append(str(expected_class).strip())
        if allowed_label_values:
            for label in (getattr(vision, "text_anchor_class", ""), getattr(vision, "center_dot_class", "")):
                if str(label or "").strip():
                    allowed_label_values.append(str(label))
        # Normalize to lowercase for case-insensitive matching
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

    def _predict_tflite(self, image, vision: VisionConfig, expected_class: str | None = None) -> dict[str, Any]:
        """Run inference via TFLite interpreter (CPU-friendly)."""
        import time as _time
        t0 = _time.perf_counter()

        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Sticker model path is not configured.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {resolved_model_path}")

        logger.info("[tflite] loading model: %s", resolved_model_path)
        interpreter = self._load_tflite_interpreter(str(resolved_model_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        logger.info("[tflite] input_shape=%s dtype=%s", input_shape, input_dtype)

        # Preprocess
        h_in, w_in = int(input_shape[1]), int(input_shape[2])
        img_resized = cv2.resize(image, (w_in, h_in))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(input_dtype)
        if input_dtype == np.float32:
            img_normalized = img_normalized / 255.0
        input_data = np.expand_dims(img_normalized, axis=0)
        t1 = _time.perf_counter()
        logger.info("[tflite] preprocess=%.1fms", (t1 - t0) * 1000)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        t2 = _time.perf_counter()
        logger.info("[tflite] invoke=%.1fms", (t2 - t1) * 1000)

        # Parse output
        raw_box_count = 0
        detections = []
        if output_data is not None and output_data.size > 0:
            out = output_data[0]
            logger.info("[tflite] output_shape=%s", out.shape)
            if out.ndim == 2:
                if out.shape[0] < out.shape[1]:
                    out = out.T
                raw_box_count = int(out.shape[0])
                for row in out:
                    conf = float(row[4]) if len(row) > 4 else 0.0
                    if conf < float(vision.conf_threshold):
                        continue
                    class_id = int(row[5]) if len(row) > 5 else 0
                    xc, yc, w_b, h_b = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    x1 = max(0.0, xc - w_b / 2)
                    y1 = max(0.0, yc - h_b / 2)
                    x2 = min(1.0, xc + w_b / 2)
                    y2 = min(1.0, yc + h_b / 2)
                    detections.append({
                        "label": str(class_id),
                        "confidence": round(conf, 4),
                        "class_confidence": round(conf, 4),
                        "class_id": class_id,
                        "position": {"x1": round(x1, 4), "y1": round(y1, 4), "x2": round(x2, 4), "y2": round(y2, 4)},
                        "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                    })

        t3 = _time.perf_counter()
        logger.info("[tflite] total=%.1fms parse=%.1fms raw=%d filtered=%d",
                     (t3 - t0) * 1000, (t3 - t2) * 1000, raw_box_count, len(detections))

        # Load class names and filter
        meta = self._load_meta(self._resolve_meta_path(vision))
        class_names = meta.get("class_names", [])
        names_map = {i: name for i, name in enumerate(class_names)}

        allowed_label_values = [str(label) for label in (vision.classes or []) if str(label).strip()]
        if expected_class and str(expected_class).strip():
            allowed_label_values.append(str(expected_class).strip())
        if allowed_label_values:
            for label in (getattr(vision, "text_anchor_class", ""), getattr(vision, "center_dot_class", "")):
                if str(label or "").strip():
                    allowed_label_values.append(str(label))
        allowed_labels = {label.strip().lower() for label in allowed_label_values} or None

        filtered = []
        for det in detections:
            label = names_map.get(det["class_id"], str(det["class_id"]))
            det["label"] = label
            if allowed_labels is not None:
                if label.strip().lower() not in allowed_labels:
                    continue
            filtered.append(det)

        return {
            "backend": "tflite",
            "mode": "tflite",
            "model_path": str(resolved_model_path),
            "meta_path": self._resolve_meta_path(vision) or None,
            "class_names": class_names,
            "detections": filtered,
            "raw_detection_count": raw_box_count,
            "allowed_labels_filter": sorted(allowed_labels) if allowed_labels is not None else None,
            "fallback_reason": None,
            "device_mode": "cpu",
            "effective_device": "cpu",
            "device_backend": "tflite",
            "device_fallback_reason": None,
            "gpu_available": False,
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
        """Crop text area dari image. Perbesar area untuk improve OCR."""
        if image is None or getattr(image, "size", 0) == 0:
            return None
        height, width = image.shape[:2]

        # Kalau text_bbox tidak ada, gunakan seluruh image (fallback)
        if not text_bbox:
            return image

        x1 = float(text_bbox.get("x1", 0.0))
        y1 = float(text_bbox.get("y1", 0.0))
        x2 = float(text_bbox.get("x2", 0.0))
        y2 = float(text_bbox.get("y2", 0.0))

        # Perbesar padding untuk capture lebih banyak konteks
        pad_ratio = max(0.0, float(getattr(vision, "anchor_crop_padding_ratio", 0.15) or 0.15))
        pad_x = (x2 - x1) * pad_ratio
        pad_y = (y2 - y1) * pad_ratio

        # Tambah padding minimum 10 pixel
        pad_x = max(pad_x, 10)
        pad_y = max(pad_y, 10)

        ix1 = max(0, int(round(x1 - pad_x)))
        iy1 = max(0, int(round(y1 - pad_y)))
        ix2 = min(width, int(round(x2 + pad_x)))
        iy2 = min(height, int(round(y2 + pad_y)))

        if ix2 <= ix1 or iy2 <= iy1:
            return None

        cropped = image[iy1:iy2, ix1:ix2]
        logger.info("[ocr] crop: bbox=%.0fx%.0f pad=%.0fpx → crop=%dx%d",
                     x2-x1, y2-y1, pad_x, cropped.shape[1], cropped.shape[0])
        return cropped

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

        config_parts = [f"--psm {int(getattr(vision, 'ocr_psm', 6) or 6)}"]
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
        logger.info("[ocr] raw='%s' conf=%.2f words=%d", raw_text, confidence * 100, len(words))
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

    def _ocr_with_flip_fallback(
        self,
        image,
        vision: VisionConfig,
        *,
        expected_text: str | None,
        regex: str | None,
        canonical_map: dict[str, str],
    ) -> dict[str, Any]:
        result_normal = self._ocr_with_tesseract(
            image,
            vision,
            expected_text=expected_text,
            regex=regex,
            canonical_map=canonical_map,
        )
        result_normal["was_flipped"] = False

        if image is None or getattr(image, "size", 0) == 0:
            return result_normal

        flipped_image = cv2.flip(image, -1)
        result_flipped = self._ocr_with_tesseract(
            flipped_image,
            vision,
            expected_text=expected_text,
            regex=regex,
            canonical_map=canonical_map,
        )
        result_flipped["was_flipped"] = True

        expected_key = self._normalize_label_key(expected_text)
        if expected_key:
            normal_keys = {
                self._normalize_label_key(result_normal.get("canonical_text")),
                self._normalize_label_key(self.parse_unique_code(str(result_normal.get("canonical_text") or result_normal.get("text") or ""))),
            }
            flipped_keys = {
                self._normalize_label_key(result_flipped.get("canonical_text")),
                self._normalize_label_key(self.parse_unique_code(str(result_flipped.get("canonical_text") or result_flipped.get("text") or ""))),
            }
            normal_match = expected_key in normal_keys
            flipped_match = expected_key in flipped_keys
            if normal_match and not flipped_match:
                result_normal["match_expected"] = True
                return result_normal
            if flipped_match and not normal_match:
                result_flipped["match_expected"] = True
                return result_flipped

        normal_conf = float(result_normal.get("confidence") or 0.0)
        flipped_conf = float(result_flipped.get("confidence") or 0.0)
        return result_flipped if flipped_conf > normal_conf else result_normal

    @staticmethod
    def parse_unique_code(ocr_text: str, separator: str = "-") -> str:
        """Parse kode dari hasil OCR. Handle multi-line.

        Untuk sticker dengan 2 baris, OCR bisa menghasilkan:
        "K0W-HB0\\nK1Z-FA0" atau "K0W-HB0 K1Z-FA0"

        Strategy:
        1. Split by newline
        2. Untuk setiap baris, cari bagian terakhir setelah separator
        3. Return baris yang paling cocok (punya separator)
        """
        text = str(ocr_text or "").strip()
        if not text:
            return ""

        # Handle multi-line: split by newline, proses setiap baris
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return ""

        # Untuk setiap baris, cari kode setelah separator
        for line in lines:
            parts = line.split(separator)
            if len(parts) >= 2:
                # Ambil bagian terakhir (setelah separator terakhir)
                return parts[-1].strip()

        # Fallback: kalau tidak ada separator, return baris pertama
        return lines[0].strip()

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
        sticker_center = None
        if image is None or getattr(image, "size", 0) == 0:
            roi_w = roi_h = 0
        else:
            roi_h, roi_w = image.shape[:2]
            sticker_center = {"x": round(roi_w / 2.0, 2), "y": round(roi_h / 2.0, 2)}

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
        if text_position is not None:
            # Gunakan dot_position jika ada, fallback ke sticker bbox center
            ref_position = dot_position if dot_position is not None else sticker_center
            if ref_position is not None:
                dx = float(ref_position["x"]) - float(text_position["x"])
                dy = float(ref_position["y"]) - float(text_position["y"])
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

    def _select_sticker_detection(
        self,
        detections: list[dict[str, Any]],
        expected_class: str | None,
    ) -> dict[str, Any] | None:
        if not detections:
            return None
        expected_key = self._normalize_label_key(expected_class)
        if expected_key:
            matches = [item for item in detections if self._normalize_label_key(item.get("label")) == expected_key]
            if matches:
                return max(matches, key=lambda item: float(item.get("confidence") or 0.0))
        return max(detections, key=lambda item: float(item.get("confidence") or 0.0))

    def _augment_with_ocr_only(
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
        text_anchor = self._select_sticker_detection(detections, expected_class)
        text_bbox = self._round_bbox((text_anchor or {}).get("position"))
        text_position = self._bbox_center(text_bbox)

        if image is None or getattr(image, "size", 0) == 0:
            roi_h = roi_w = 0
        else:
            roi_h, roi_w = image.shape[:2]
        roi_center = {"x": round(roi_w / 2.0, 2), "y": round(roi_h / 2.0, 2)}

        anchor_offset = None
        if text_position is not None:
            anchor_offset = {
                "x": round(float(text_position["x"]) - roi_center["x"], 2),
                "y": round(float(text_position["y"]) - roi_center["y"], 2),
                "source": "bbox_center",
            }

        expected_tilt = float(getattr(sticker_rule, "expected_tilt_degrees", 0.0) or 0.0) if sticker_rule is not None else 0.0

        # Skip tilt calculation kalau text_anchor tidak ada (hemat CPU)
        tilt_info = {"status": "skipped", "angle_degrees": None, "deviation_degrees": None}
        if text_anchor is not None:
            tilt_info = estimate_white_text_tilt(image, expected_tilt, sticker_rule)
        geometry = {
            "status": "ok" if text_anchor is not None else "missing_anchor",
            "text_position": text_position,
            "dot_position": None,
            "expected_dot_position": roi_center,
            "anchor_offset": anchor_offset,
            "pose_angle": tilt_info.get("angle_degrees"),
            "pose_deviation": tilt_info.get("deviation_degrees"),
        }
        anchor = {
            "status": "ok" if text_anchor is not None else "missing",
            "text_anchor": text_anchor,
            "center_dot": None,
            "text_bbox": text_bbox,
            "dot_bbox": None,
            "text_position": text_position,
            "dot_position": None,
            "text_confidence": None if text_anchor is None else round(float(text_anchor.get("confidence") or 0.0), 4),
            "dot_confidence": None,
            "text_anchor_class": expected_class or getattr(vision, "text_anchor_class", "text_anchor"),
            "center_dot_class": None,
        }
        anchor_ms = round((time.perf_counter() - anchor_started) * 1000.0, 2)

        ocr_started = time.perf_counter()
        expected_code = str(getattr(sticker_rule, "ocr_expected_code", "") or "").strip() if sticker_rule is not None else ""
        expected_text = expected_code or str(getattr(sticker_rule, "ocr_expected_text", None) or expected_class or "").strip() or None
        regex = getattr(sticker_rule, "ocr_regex", None) if sticker_rule is not None else None
        canonical_map = dict(getattr(sticker_rule, "ocr_canonical_map", {}) or {}) if sticker_rule is not None else {}
        use_ocr = bool(getattr(sticker_rule, "use_ocr", False)) if sticker_rule is not None else False
        engine = self._resolve_ocr_engine(vision)
        if not use_ocr:
            ocr = {
                "status": "skipped",
                "engine": engine,
                "text": "",
                "raw_text": "",
                "canonical_text": "",
                "confidence": None,
                "expected_text": expected_text,
                "match_expected": False,
                "error": None,
                "was_flipped": False,
            }
        elif text_anchor is None:
            ocr = {
                "status": "anchor_not_found",
                "engine": engine,
                "text": "",
                "raw_text": "",
                "canonical_text": "",
                "confidence": None,
                "expected_text": expected_text,
                "match_expected": False,
                "error": None,
                "was_flipped": False,
            }
        elif engine == "passthrough":
            ocr = self._ocr_passthrough(anchor, expected_text=expected_text, regex=regex, canonical_map=canonical_map)
            ocr["was_flipped"] = False
        elif engine in {"disabled", "none", "off"}:
            ocr = {
                "status": "disabled",
                "engine": engine,
                "text": "",
                "raw_text": "",
                "canonical_text": "",
                "confidence": None,
                "expected_text": expected_text,
                "match_expected": False,
                "error": None,
                "was_flipped": False,
            }
        else:
            crop = self._crop_text_anchor(image, text_bbox, vision)
            prepared = self._preprocess_ocr_crop(crop, vision)
            if prepared is None:
                ocr = {
                    "status": "empty_crop",
                    "engine": engine,
                    "text": "",
                    "raw_text": "",
                    "canonical_text": "",
                    "confidence": None,
                    "expected_text": expected_text,
                    "match_expected": False,
                    "error": None,
                    "was_flipped": False,
                }
            elif engine == "tesseract":
                if bool(getattr(sticker_rule, "ocr_flip_fallback", True)):
                    ocr = self._ocr_with_flip_fallback(
                        prepared,
                        vision,
                        expected_text=expected_text,
                        regex=regex,
                        canonical_map=canonical_map,
                    )
                else:
                    ocr = self._ocr_with_tesseract(
                        prepared,
                        vision,
                        expected_text=expected_text,
                        regex=regex,
                        canonical_map=canonical_map,
                    )
                    ocr["was_flipped"] = False
            else:
                ocr = {
                    "status": "unsupported_engine",
                    "engine": engine,
                    "text": "",
                    "raw_text": "",
                    "canonical_text": "",
                    "confidence": None,
                    "expected_text": expected_text,
                    "match_expected": False,
                    "error": f"Unsupported OCR engine: {engine}",
                    "was_flipped": False,
                }

        unique_code = self.parse_unique_code(str(ocr.get("canonical_text") or ocr.get("text") or ""))
        if expected_code:
            ocr["match_expected"] = self._normalize_label_key(unique_code) == self._normalize_label_key(expected_code)
        ocr_ms = round((time.perf_counter() - ocr_started) * 1000.0, 2)

        payload["anchor"] = anchor
        payload["ocr"] = ocr
        payload["geometry"] = geometry
        payload["unique_code"] = unique_code
        payload["tilt_info"] = tilt_info
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

        validator_mode = str(getattr(sticker_rule, "validator_mode", "") or "").strip().lower() if sticker_rule is not None else ""
        use_sticker_only = bool(getattr(sticker_rule, "use_ocr", False)) or validator_mode in {
            "sticker_only",
            "ocr_only",
            "ocr_sticker",
            "sticker_ocr",
        }

        mode = self._resolve_mode()
        if mode == "classic":
            payload = self._predict_classic(image, vision, expected_class)
            if use_sticker_only:
                return self._augment_with_ocr_only(
                    payload, image, vision,
                    expected_class=expected_class, sticker_rule=sticker_rule,
                )
            return self._augment_with_anchor_ocr(
                payload, image, vision,
                expected_class=expected_class, sticker_rule=sticker_rule,
            )

        # Auto-detect TFLite from file extension when mode is "auto"
        if mode == "auto":
            model_path = self._resolve_model_path(vision)
            if model_path and Path(model_path).suffix.lower() == ".tflite":
                mode = "tflite"
            else:
                mode = "ultralytics"

        # TFLite mode: CPU-only, no GPU device resolution needed
        if mode == "tflite":
            try:
                payload = self._predict_tflite(image, vision, expected_class=expected_class)
                if use_sticker_only:
                    return self._augment_with_ocr_only(
                        payload, image, vision,
                        expected_class=expected_class, sticker_rule=sticker_rule,
                    )
                return self._augment_with_anchor_ocr(
                    payload, image, vision,
                    expected_class=expected_class, sticker_rule=sticker_rule,
                )
            except Exception as exc:
                logging.warning("TFLite inference failed, fallback to classic: %s", exc)
                payload = self._predict_classic(image, vision, expected_class)
                payload["fallback_reason"] = f"tflite_error: {exc}"
                return self._augment_with_anchor_ocr(
                    payload, image, vision,
                    expected_class=expected_class, sticker_rule=sticker_rule,
                )

        # Ultralytics mode (auto/ultralytics)
        device_resolution = self._resolve_device()
        try:
            payload = self._predict_ultralytics(image, vision, device_resolution, expected_class=expected_class)
            if use_sticker_only:
                return self._augment_with_ocr_only(
                    payload, image, vision,
                    expected_class=expected_class, sticker_rule=sticker_rule,
                )
            return self._augment_with_anchor_ocr(
                payload, image, vision,
                expected_class=expected_class, sticker_rule=sticker_rule,
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
            if use_sticker_only:
                return self._augment_with_ocr_only(
                    payload,
                    image,
                    vision,
                    expected_class=expected_class,
                    sticker_rule=sticker_rule,
                )
            return self._augment_with_anchor_ocr(
                payload,
                image,
                vision,
                expected_class=expected_class,
                sticker_rule=sticker_rule,
            )

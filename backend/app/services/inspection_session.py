from __future__ import annotations

import base64
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import cv2
import numpy as np

from backend.app.core.config import AppConfig
from backend.app.models.session_state import SessionState
from backend.app.repositories.inspection_results_repository import InspectionResultsRepository
from backend.app.repositories.profiles_repository import ProfilesRepository
from backend.app.services.calibration import CalibrationService
from backend.app.services.sticker_inference import StickerInferenceService
from backend.app.services.template_runtime import TemplateRuntimeService
from shared.contracts.enums import DecisionCode, InspectionEventState, RejectReasonCode, SessionStatus
from shared.contracts.templates import RoiGeometry


KNOWN_REJECT_CODES = (
    RejectReasonCode.NOT_FOUND.value,
    RejectReasonCode.WRONG_TYPE.value,
    RejectReasonCode.LOW_ROI_CONF.value,
    RejectReasonCode.LOW_CLASS_CONF.value,
    RejectReasonCode.OUT_OF_POSITION.value,
    RejectReasonCode.PART_NOT_READY.value,
    RejectReasonCode.ERROR.value,
)
MAX_RECENT_EVENTS = 8
COMMIT_STABLE_FRAMES = 1
COMMIT_COOLDOWN_MS = 800
PRESENCE_MIN_AREA_RATIO = 0.01
PRESENCE_MIN_STD = 8.0
PRESENCE_MIN_MEAN = 6.0
ROI_CLASS_VALIDATOR_MODES = {
    "ml_roi_class",
    "ml_roi_classification",
    "roi_class",
    "roi_partial",
}


def _decode_image(image_b64: str):
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image payload.")
    return image


def _encode_image(image) -> str:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _empty_reject_breakdown() -> dict[str, int]:
    return {code: 0 for code in KNOWN_REJECT_CODES}


def _round_bbox(position: dict[str, Any] | None) -> dict[str, float] | None:
    if not position:
        return None
    return {
        "x1": round(float(position.get("x1", 0.0)), 2),
        "y1": round(float(position.get("y1", 0.0)), 2),
        "x2": round(float(position.get("x2", 0.0)), 2),
        "y2": round(float(position.get("y2", 0.0)), 2),
    }


class InspectionSessionService:
    def __init__(
        self,
        template_runtime: TemplateRuntimeService,
        profiles_repo: ProfilesRepository,
        results_repo: InspectionResultsRepository,
        sticker_inference: StickerInferenceService,
        app_config: AppConfig | None = None,
    ) -> None:
        self._template_runtime = template_runtime
        self._profiles_repo = profiles_repo
        self._results_repo = results_repo
        self._sticker_inference = sticker_inference
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        # System-wide settle default: used when a template's part_ready_settle_ms is None.
        self._default_settle_ms: int = (
            max(0, int(app_config.part_ready_settle_ms_default))
            if app_config is not None
            else 0
        )

    def start_session(
        self,
        *,
        client_id: str,
        camera_index: int,
        template_version_id: int,
        line_id: str | None = None,
        station_id: str | None = None,
    ) -> dict[str, Any]:
        template = self._template_runtime.resolve_template_by_version(template_version_id)
        session_id = uuid.uuid4().hex
        state = SessionState(
            session_id=session_id,
            client_id=client_id,
            camera_index=int(camera_index),
            template=template,
            status=SessionStatus.RUNNING,
            line_id=line_id,
            station_id=station_id,
            session_reject_breakdown=_empty_reject_breakdown(),
        )
        with self._lock:
            self._sessions[session_id] = state
        return self._session_payload(state)

    def update_roi(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        state = self._require_session(session_id)
        allowed = ("x", "y", "w", "h", "width", "height")
        before_part_ready_signature = self._roi_signature(
            self._merged_roi_payload(state.template.part_ready_roi, state.part_ready_roi_override)
        )

        legacy_updates = {key: updates[key] for key in allowed if key in updates}
        legacy_nested = updates.get("roi") if isinstance(updates.get("roi"), dict) else None
        if legacy_nested:
            legacy_updates.update({key: legacy_nested[key] for key in allowed if key in legacy_nested})
        if legacy_updates:
            state.part_ready_roi_override.update(legacy_updates)
            state.sticker_roi_override.update(legacy_updates)

        part_ready_updates = updates.get("part_ready_roi")
        if isinstance(part_ready_updates, dict):
            state.part_ready_roi_override.update(
                {key: part_ready_updates[key] for key in allowed if key in part_ready_updates}
            )

        sticker_updates = updates.get("sticker_roi")
        if isinstance(sticker_updates, dict):
            state.sticker_roi_override.update(
                {key: sticker_updates[key] for key in allowed if key in sticker_updates}
            )

        after_part_ready_signature = self._roi_signature(
            self._merged_roi_payload(state.template.part_ready_roi, state.part_ready_roi_override)
        )
        if after_part_ready_signature != before_part_ready_signature:
            state.part_ready_ratio_history.clear()
        return self._session_payload(state)

    def get_latest_preview(self) -> dict[str, Any] | None:
        with self._lock:
            for state in self._sessions.values():
                if state.last_overlay_b64:
                    return {
                        "overlay_image_b64": state.last_overlay_b64,
                        "session_id": state.session_id,
                        "frame_index": state.frame_index,
                    }
        return None

    def stop_session(self, session_id: str) -> dict[str, Any]:
        state = self._require_session(session_id)
        state.status = SessionStatus.STOPPED
        with self._lock:
            self._sessions.pop(session_id, None)
        return self._session_payload(state)

    def process_frame(
        self,
        session_id: str,
        *,
        image_b64: str,
        response_mode: str | None = None,
        username: str | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        timings: dict[str, float] = {}

        def _elapsed_ms(started_at: float) -> float:
            return round((time.perf_counter() - started_at) * 1000.0, 2)

        state = self._require_session(session_id)
        state.frame_index += 1

        decode_started = time.perf_counter()
        frame = _decode_image(image_b64)
        timings["decode_ms"] = _elapsed_ms(decode_started)

        part_ready_started = time.perf_counter()
        part_ready_frame, part_ready_roi_meta = self._crop_stage_roi(
            frame,
            state.template.part_ready_roi,
            state.part_ready_roi_override,
        )
        part_ready = self._evaluate_part_ready(part_ready_frame, state)
        presence = self._detect_part_presence(part_ready_frame)
        timings["part_ready_eval_ms"] = _elapsed_ms(part_ready_started)

        roi_crop_started = time.perf_counter()
        sticker_frame, sticker_roi_meta = self._crop_stage_roi(
            frame,
            state.template.sticker_roi,
            state.sticker_roi_override,
        )
        timings["sticker_roi_crop_ms"] = _elapsed_ms(roi_crop_started)

        # ------------------------------------------------------------------
        # Settle-time debounce
        # Hold inference and commit for settle_ms after part_ready first
        # transitions to True.  settle_ms = 0 restores legacy behaviour.
        # Resolution priority: template (when not None) > env default > 0.
        # ------------------------------------------------------------------
        _template_settle = getattr(state.template.sticker, "part_ready_settle_ms", None)
        settle_ms = (
            max(0, int(_template_settle))
            if _template_settle is not None
            else self._default_settle_ms
        )
        _settle_now = datetime.now(UTC)
        _raw_part_ready = part_ready.get("part_ready", False)
        if _raw_part_ready and presence.get("present", False):
            if state.part_ready_settle_started_at is None:
                state.part_ready_settle_started_at = _settle_now
            _elapsed_settle_ms = (
                (_settle_now - state.part_ready_settle_started_at).total_seconds() * 1000.0
            )
            part_ready_settled = settle_ms == 0 or _elapsed_settle_ms >= settle_ms
            settle_remaining_ms = (
                max(0.0, float(settle_ms) - _elapsed_settle_ms)
                if not part_ready_settled
                else 0.0
            )
        else:
            state.part_ready_settle_started_at = None
            part_ready_settled = False
            settle_remaining_ms = 0.0

        # Augment part_ready dict in-place with settle observability fields.
        part_ready["part_ready_settled"] = part_ready_settled
        part_ready["part_ready_settle_ms"] = settle_ms
        part_ready["part_ready_settle_remaining_ms"] = round(settle_remaining_ms, 1)

        # effective_part_ready is used for all downstream logic (validation,
        # event state).  During settle it looks like part_not_ready to avoid
        # premature commits, while the original part_ready dict is reported.
        _effective_pr_ready = _raw_part_ready and part_ready_settled
        if _raw_part_ready and not part_ready_settled:
            effective_part_ready: dict[str, Any] = {
                **part_ready,
                "part_ready": False,
                "reject_reason_code": RejectReasonCode.PART_NOT_READY.value,
                "status": "settling",
            }
        else:
            effective_part_ready = part_ready

        detections: list[dict[str, Any]] = []
        inference_ms = 0.0
        if _effective_pr_ready:
            inference_started = time.perf_counter()
            inference_payload = self._run_sticker_inference(sticker_frame, state)
            inference_ms = _elapsed_ms(inference_started)
            detections = list(inference_payload.get("detections") or [])
            sticker_detection = self._build_sticker_detection_payload(
                detections,
                skipped=False,
                backend=str(inference_payload.get("backend") or "unknown"),
                model_path=inference_payload.get("model_path"),
                meta_path=inference_payload.get("meta_path"),
                class_names=inference_payload.get("class_names") or [],
                fallback_reason=inference_payload.get("fallback_reason"),
            )
            sticker_detection.update(
                {
                    "device_mode": inference_payload.get("device_mode"),
                    "effective_device": inference_payload.get("effective_device"),
                    "device_backend": inference_payload.get("device_backend"),
                    "device_fallback_reason": inference_payload.get("device_fallback_reason"),
                    "gpu_available": inference_payload.get("gpu_available"),
                }
            )
        else:
            _skip_reason = (
                "part_ready_settling"
                if _raw_part_ready and not part_ready_settled
                else (part_ready.get("reject_reason_code") or "part_not_ready")
            )
            sticker_detection = self._build_sticker_detection_payload(
                [],
                skipped=True,
                reason=_skip_reason,
                backend="skipped",
                model_path=state.template.vision.model_path,
                meta_path=state.template.vision.model_meta_path,
                class_names=state.template.vision.classes,
            )
        timings["inference_ms"] = round(inference_ms, 2)

        validation_started = time.perf_counter()
        validation = self._validate_sticker(
            roi_frame=sticker_frame,
            state=state,
            detections=detections,
            detection_payload=sticker_detection,
            part_ready_payload=effective_part_ready,
            username=username,
            user_id=user_id,
        )
        timings["validation_ms"] = _elapsed_ms(validation_started)
        validation_details = validation.get("validation_details") or {}
        if validation_details:
            sticker_detection["selected_candidate"] = validation_details.get("selected_candidate")
            sticker_detection["candidate_source"] = validation_details.get("candidate_source")
            sticker_detection["matching_candidate_count"] = validation_details.get("matching_candidate_count")

        event_state_started = time.perf_counter()
        event_state, event_id, count_committed = self._advance_event_state(
            state=state,
            validation=validation,
            part_ready_payload=effective_part_ready,
            presence=presence,
            now=datetime.now(UTC),
        )
        timings["event_state_ms"] = _elapsed_ms(event_state_started)

        persistence_started = time.perf_counter()
        db_write = {"written": False, "reason": "not_committed"}
        if count_committed:
            db_write = self._maybe_persist(
                validation,
                state,
                part_ready=part_ready,
                part_ready_roi_meta=part_ready_roi_meta,
                sticker_roi_meta=sticker_roi_meta,
            )
            self._register_committed_result(
                state=state,
                validation=validation,
                part_ready_payload=part_ready,
                sticker_detection=sticker_detection,
                db_write=db_write,
                event_id=event_id,
                part_ready_roi_meta=part_ready_roi_meta,
                sticker_roi_meta=sticker_roi_meta,
                committed_at=datetime.now(UTC),
            )
        timings["persistence_ms"] = _elapsed_ms(persistence_started)

        overlay_started = time.perf_counter()
        overlay = self._compose_overlay(
            full_frame=frame,
            part_ready_roi_meta=part_ready_roi_meta,
            sticker_roi_meta=sticker_roi_meta,
            detections=detections,
            validation=validation,
            part_ready=part_ready,
            event_state=event_state,
            state=state,
        )
        timings["overlay_compose_ms"] = _elapsed_ms(overlay_started)

        normalized_response_mode = str(response_mode or "").strip().lower()
        compact_response = normalized_response_mode in {"compact", "minimal", "overlay"}

        encode_started = time.perf_counter()
        overlay_image_b64 = _encode_image(overlay)
        preview_image_b64 = None
        part_ready_preview_image_b64 = None
        sticker_preview_image_b64 = None
        if not compact_response:
            preview_image_b64 = _encode_image(sticker_frame)
            part_ready_preview_image_b64 = _encode_image(part_ready_frame)
            sticker_preview_image_b64 = preview_image_b64
        timings["encode_ms"] = _elapsed_ms(encode_started)

        timings_payload = {
            **timings,
            "inference_skipped": not _effective_pr_ready,
            "compact_response": compact_response,
            "total_ms": _elapsed_ms(total_started),
        }

        payload = {
            "session": self._session_payload(state),
            "roi": sticker_roi_meta,
            "part_ready_roi_meta": part_ready_roi_meta,
            "sticker_roi_meta": sticker_roi_meta,
            "detections": detections,
            "sticker_detection": sticker_detection,
            "presence": presence,
            "part_ready": part_ready,
            "validation": validation,
            "event_state": event_state,
            "event_id": event_id,
            "count_committed": count_committed,
            "count_source": "session" if count_committed else None,
            "counters": self._counter_payload(state),
            "last_committed_result": state.last_committed_result,
            "recent_events": list(state.recent_events),
            "db_write": db_write,
            "timings": timings_payload,
            "response_mode": "compact" if compact_response else "full",
            "overlay_image_b64": overlay_image_b64,
            "preview_image_b64": preview_image_b64,
            "part_ready_preview_image_b64": part_ready_preview_image_b64,
            "sticker_preview_image_b64": sticker_preview_image_b64,
        }
        state.last_overlay_b64 = overlay_image_b64
        state.latest_result = payload
        return payload

    def _require_session(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
        if not state:
            raise ValueError("Inspection session not found.")
        return state

    def _merged_roi_payload(self, base: RoiGeometry, override: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "x": base.x,
            "y": base.y,
            "w": base.w,
            "h": base.h,
            "width": base.width,
            "height": base.height,
        }
        payload.update(override)
        return payload

    def _session_payload(self, state: SessionState) -> dict[str, Any]:
        part_ready_roi = self._merged_roi_payload(state.template.part_ready_roi, state.part_ready_roi_override)
        sticker_roi = self._merged_roi_payload(state.template.sticker_roi, state.sticker_roi_override)
        return {
            "session_id": state.session_id,
            "client_id": state.client_id,
            "camera_index": state.camera_index,
            "template_version_id": state.template.version_id,
            "line_id": state.line_id,
            "station_id": state.station_id,
            "status": state.status.value,
            "template_name": state.template.name,
            "part_ready_roi": part_ready_roi,
            "sticker_roi": sticker_roi,
            "roi": sticker_roi,
        }

    @staticmethod
    def _roi_signature(roi_payload: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            round(float(roi_payload.get("x", 0.0) or 0.0), 6),
            round(float(roi_payload.get("y", 0.0) or 0.0), 6),
            round(float(roi_payload.get("w", 1.0) or 1.0), 6),
            round(float(roi_payload.get("h", 1.0) or 1.0), 6),
        )

    def _counter_payload(self, state: SessionState) -> dict[str, Any]:
        breakdown = _empty_reject_breakdown()
        breakdown.update(state.session_reject_breakdown)
        return {
            "scope": "session",
            "session_total": state.session_total,
            "session_accept": state.session_accept,
            "session_reject": state.session_reject,
            "session_reject_breakdown": breakdown,
        }

    def _crop_stage_roi(self, frame, base_roi: RoiGeometry, override: dict[str, Any]):
        roi = self._merged_roi_payload(base_roi, override)
        height, width = frame.shape[:2]
        x = max(0, min(width - 1, int(float(roi.get("x", 0.0)) * width)))
        y = max(0, min(height - 1, int(float(roi.get("y", 0.0)) * height)))
        roi_w = max(1, int(float(roi.get("w", 1.0)) * width))
        roi_h = max(1, int(float(roi.get("h", 1.0)) * height))
        x2 = min(width, x + roi_w)
        y2 = min(height, y + roi_h)
        cropped = frame[y:y2, x:x2]
        meta = {"x": x, "y": y, "width": x2 - x, "height": y2 - y}
        return cropped, meta

    def _run_sticker_inference(self, frame, state: SessionState) -> dict[str, Any]:
        return self._sticker_inference.predict(
            frame,
            state.template.vision,
            expected_class=state.template.sticker.expected_class,
        )

    def _build_sticker_detection_payload(
        self,
        detections: list[dict[str, Any]],
        *,
        skipped: bool,
        reason: str | None = None,
        backend: str | None = None,
        model_path: str | None = None,
        meta_path: str | None = None,
        class_names: list[str] | None = None,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        best = max(detections, key=lambda item: float(item.get("confidence") or 0.0), default=None)
        return {
            "status": "skipped" if skipped else "ok",
            "reason": reason,
            "backend": backend,
            "model_path": model_path,
            "meta_path": meta_path,
            "class_names": list(class_names or []),
            "fallback_reason": fallback_reason,
            "count": len(detections),
            "items": detections,
            "best": best,
        }

    def _detect_part_presence(self, frame) -> dict[str, Any]:
        if frame.size == 0:
            return {"present": False, "score": 0.0, "reason": "empty_roi", "area_ratio": 0.0}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(gray.mean())
        std_intensity = float(gray.std())
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = max((cv2.contourArea(contour) for contour in contours), default=0.0)
        area_ratio = float(largest_area / max(1, frame.shape[0] * frame.shape[1]))
        present = (
            area_ratio >= PRESENCE_MIN_AREA_RATIO
            or std_intensity >= PRESENCE_MIN_STD
            or mean_intensity >= PRESENCE_MIN_MEAN
        )
        reason = "texture" if std_intensity >= PRESENCE_MIN_STD else "brightness"
        score = min(1.0, max(area_ratio * 8.0, std_intensity / 32.0, mean_intensity / 255.0))
        return {
            "present": present,
            "score": round(score, 4),
            "reason": reason if present else "idle",
            "area_ratio": round(area_ratio, 6),
        }

    def _evaluate_part_ready(self, frame, state: SessionState) -> dict[str, Any]:
        config = state.template.part_ready
        if not config.enabled:
            return {
                "enabled": False,
                "part_ready": True,
                "part_ready_confidence": 1.0,
                "decision": DecisionCode.ACCEPT.value,
                "reject_reason_code": None,
                "status": "disabled",
                "match_ratio": None,
                "mean_distance": None,
                "color_profile_id": None,
            }
        if not config.color_profile_id:
            return {
                "enabled": False,
                "part_ready": True,
                "part_ready_confidence": 1.0,
                "decision": DecisionCode.ACCEPT.value,
                "reject_reason_code": None,
                "status": "skipped",
                "match_ratio": None,
                "mean_distance": None,
                "color_profile_id": None,
            }
        record = self._profiles_repo.get(config.color_profile_id)
        if not record:
            return {
                "enabled": True,
                "part_ready": False,
                "part_ready_confidence": 0.0,
                "decision": DecisionCode.REJECT.value,
                "reject_reason_code": RejectReasonCode.PART_NOT_READY.value,
                "status": "missing_profile",
                "match_ratio": 0.0,
                "mean_distance": None,
                "color_profile_id": config.color_profile_id,
            }
        evaluation = CalibrationService.evaluate_color_match(
            frame,
            record["profile"],
            colorspace=config.colorspace,
            distance_threshold=config.distance_threshold,
            min_match_ratio=config.min_match_ratio,
        )
        raw_ratio = float(evaluation["match_ratio"])

        # Rolling mean over last 5 frames to absorb transient occlusion / auto-exposure flicker.
        _PART_READY_WINDOW = 5
        history = state.part_ready_ratio_history
        history.append(raw_ratio)
        if len(history) > _PART_READY_WINDOW:
            del history[0]
        smoothed_ratio = round(sum(history) / len(history), 6)

        resolved_min = float(evaluation["min_match_ratio"])
        ready = smoothed_ratio >= resolved_min
        return {
            "enabled": True,
            "part_ready": ready,
            "part_ready_confidence": smoothed_ratio,
            "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
            "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
            "status": "ready" if ready else "not_ready",
            "match_ratio": smoothed_ratio,
            "raw_match_ratio": raw_ratio,
            "mean_distance": evaluation["mean_distance"],
            "distance_threshold": evaluation["distance_threshold"],
            "min_match_ratio": resolved_min,
            "color_profile_id": config.color_profile_id,
            "colorspace": evaluation["colorspace"],
        }

    def _normalize_label(self, value: Any) -> str:
        return str(value or "").strip().lower()

    def _build_validation_candidate_summaries(
        self,
        detections: list[dict[str, Any]],
        roi_frame,
        expected_class: str,
        sticker_config=None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        roi_h, roi_w = roi_frame.shape[:2]
        cx = float((sticker_config.expected_center_x if sticker_config and sticker_config.expected_center_x is not None else None) or 0.5)
        cy = float((sticker_config.expected_center_y if sticker_config and sticker_config.expected_center_y is not None else None) or 0.5)
        expected_center = {
            "x": round(cx * roi_w, 2),
            "y": round(cy * roi_h, 2),
        }
        expected_label = self._normalize_label(expected_class)
        candidates: list[dict[str, Any]] = []
        for index, det in enumerate(detections):
            pos = det.get("position") or {}
            center_x = (float(pos.get("x1", 0.0)) + float(pos.get("x2", 0.0))) / 2.0
            center_y = (float(pos.get("y1", 0.0)) + float(pos.get("y2", 0.0))) / 2.0
            offset_x = center_x - expected_center["x"]
            offset_y = center_y - expected_center["y"]
            candidates.append(
                {
                    "index": index,
                    "label": str(det.get("label") or ""),
                    "normalized_label": self._normalize_label(det.get("label")),
                    "confidence": round(float(det.get("confidence") or 0.0), 4),
                    "class_confidence": round(float(det.get("class_confidence") or 0.0), 4),
                    "bbox": _round_bbox(pos),
                    "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
                    "offset": {"x": round(offset_x, 2), "y": round(offset_y, 2)},
                    "match_expected": self._normalize_label(det.get("label")) == expected_label,
                }
            )
        return candidates, expected_center

    def _select_validation_candidate(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str]:
        if not candidates:
            return None, "none"
        expected_matches = [item for item in candidates if item.get("match_expected")]
        if expected_matches:
            selected = max(
                expected_matches,
                key=lambda item: (
                    float(item.get("confidence") or 0.0),
                    float(item.get("class_confidence") or 0.0),
                ),
            )
            return selected, "expected_class"
        selected = max(
            candidates,
            key=lambda item: (
                float(item.get("confidence") or 0.0),
                float(item.get("class_confidence") or 0.0),
            ),
        )
        return selected, "highest_confidence"

    def _validate_sticker(
        self,
        *,
        roi_frame,
        state: SessionState,
        detections: list[dict[str, Any]],
        detection_payload: dict[str, Any],
        part_ready_payload: dict[str, Any],
        username: str | None,
        user_id: int | None,
    ) -> dict[str, Any]:
        sticker = state.template.sticker
        validator_mode = str(getattr(sticker, "validator_mode", "ml_detection") or "ml_detection").strip().lower()
        position_gate_enabled = validator_mode not in ROI_CLASS_VALIDATOR_MODES
        line_id = state.line_id or sticker.line
        thresholds = {
            "min_roi_confidence": float(sticker.min_roi_confidence or 0.0),
            "min_class_confidence": (
                None if sticker.min_class_confidence is None else float(sticker.min_class_confidence)
            ),
            "max_offset_x": None if sticker.max_offset_x is None else float(sticker.max_offset_x),
            "max_offset_y": None if sticker.max_offset_y is None else float(sticker.max_offset_y),
            "validator_mode": validator_mode,
            "position_gate_enabled": position_gate_enabled,
        }
        detection_context = {
            "backend": detection_payload.get("backend"),
            "model_path": detection_payload.get("model_path"),
            "meta_path": detection_payload.get("meta_path"),
            "class_names": list(detection_payload.get("class_names") or []),
            "fallback_reason": detection_payload.get("fallback_reason"),
        }
        if not sticker.enabled:
            return {
                "decision": DecisionCode.ACCEPT.value,
                "decision_code": DecisionCode.ACCEPT.value,
                "reject_reason_code": None,
                "part_name": sticker.part_name,
                "line_id": line_id,
                "station_id": state.station_id,
                # Contract: data1 = part_ready confidence, data2 = sticker confidence
                "data1": part_ready_payload.get("part_ready_confidence"),
                "data2": None,
                "targets": [],
                "operator_user_id": user_id,
                "mp_check": username,
                "detected_class": None,
                "expected_class": sticker.expected_class,
                "sticker_confidence": None,
                "sticker_bbox": None,
                "sticker_backend": detection_context["backend"],
                "validation_details": {
                    "status": "disabled",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": len(detections),
                    "matching_candidate_count": 0,
                    "expected_center": None,
                    "thresholds": thresholds,
                },
            }
        if not part_ready_payload.get("part_ready", False):
            return {
                "decision": DecisionCode.REJECT.value,
                "decision_code": DecisionCode.REJECT.value,
                "reject_reason_code": RejectReasonCode.PART_NOT_READY.value,
                "part_name": sticker.part_name,
                "line_id": line_id,
                "station_id": state.station_id,
                # Contract: data1 = part_ready confidence, data2 = sticker confidence
                "data1": part_ready_payload.get("part_ready_confidence"),
                "data2": None,
                "targets": [],
                "operator_user_id": user_id,
                "mp_check": username,
                "detected_class": None,
                "expected_class": sticker.expected_class,
                "sticker_confidence": None,
                "sticker_bbox": None,
                "sticker_backend": detection_context["backend"],
                "validation_details": {
                    "status": "part_not_ready",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": len(detections),
                    "matching_candidate_count": 0,
                    "expected_center": None,
                    "thresholds": thresholds,
                },
            }
        if not detections:
            return {
                "decision": DecisionCode.REJECT.value,
                "decision_code": DecisionCode.REJECT.value,
                "reject_reason_code": RejectReasonCode.NOT_FOUND.value,
                "part_name": sticker.part_name,
                "line_id": line_id,
                "station_id": state.station_id,
                # Contract: data1 = part_ready confidence, data2 = sticker confidence
                "data1": part_ready_payload.get("part_ready_confidence"),
                "data2": None,
                "targets": [],
                "operator_user_id": user_id,
                "mp_check": username,
                "detected_class": None,
                "expected_class": sticker.expected_class,
                "sticker_confidence": None,
                "sticker_bbox": None,
                "sticker_backend": detection_context["backend"],
                "validation_details": {
                    "status": "not_found",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": 0,
                    "matching_candidate_count": 0,
                    "expected_center": None,
                    "thresholds": thresholds,
                },
            }

        candidates, expected_center = self._build_validation_candidate_summaries(
            detections,
            roi_frame,
            sticker.expected_class,
            sticker,
        )
        selected_candidate, candidate_source = self._select_validation_candidate(candidates)
        matching_candidate_count = sum(1 for item in candidates if item.get("match_expected"))
        if selected_candidate is None:
            return {
                "decision": DecisionCode.REJECT.value,
                "decision_code": DecisionCode.REJECT.value,
                "reject_reason_code": RejectReasonCode.NOT_FOUND.value,
                "part_name": sticker.part_name,
                "line_id": line_id,
                "station_id": state.station_id,
                # Contract: data1 = part_ready confidence, data2 = sticker confidence
                "data1": part_ready_payload.get("part_ready_confidence"),
                "data2": None,
                "targets": [],
                "operator_user_id": user_id,
                "mp_check": username,
                "detected_class": None,
                "expected_class": sticker.expected_class,
                "sticker_confidence": None,
                "sticker_bbox": None,
                "sticker_backend": detection_context["backend"],
                "validation_details": {
                    "status": "not_found",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": len(candidates),
                    "matching_candidate_count": matching_candidate_count,
                    "expected_center": expected_center,
                    "thresholds": thresholds,
                    "candidates": candidates,
                },
            }

        offset_x = float((selected_candidate.get("offset") or {}).get("x", 0.0))
        offset_y = float((selected_candidate.get("offset") or {}).get("y", 0.0))
        reject_reason = None
        if float(selected_candidate.get("confidence") or 0.0) < thresholds["min_roi_confidence"]:
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif not bool(selected_candidate.get("match_expected")):
            reject_reason = RejectReasonCode.WRONG_TYPE.value
        elif thresholds["min_class_confidence"] is not None and float(selected_candidate.get("class_confidence") or 0.0) < float(thresholds["min_class_confidence"]):
            reject_reason = RejectReasonCode.LOW_CLASS_CONF.value
        elif position_gate_enabled and thresholds["max_offset_x"] is not None and abs(offset_x) > float(thresholds["max_offset_x"]):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        elif position_gate_enabled and thresholds["max_offset_y"] is not None and abs(offset_y) > float(thresholds["max_offset_y"]):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        decision = DecisionCode.ACCEPT.value if reject_reason is None else DecisionCode.REJECT.value
        status = "accepted" if reject_reason is None else reject_reason.lower()
        target = {
            "target_id": "target-1",
            "part_name": sticker.part_name,
            "expected_class": sticker.expected_class,
            "detected_class": selected_candidate.get("label"),
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "data1": selected_candidate.get("confidence"),
            "data2": selected_candidate.get("class_confidence"),
            "position": dict(selected_candidate.get("center") or {}),
            "offset": {"x": round(offset_x, 2), "y": round(offset_y, 2)},
            "candidate_source": candidate_source,
        }
        return {
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "part_name": sticker.part_name,
            "line_id": line_id,
            "station_id": state.station_id,
            # Contract: data1 = part_ready confidence, data2 = sticker confidence
            "data1": part_ready_payload.get("part_ready_confidence"),
            "data2": selected_candidate.get("confidence"),
            "targets": [target],
            "operator_user_id": user_id,
            "mp_check": username,
            "detected_class": selected_candidate.get("label"),
            "expected_class": sticker.expected_class,
            "sticker_confidence": selected_candidate.get("confidence"),
            "sticker_bbox": dict(selected_candidate.get("bbox") or {}) or None,
            "sticker_backend": detection_context["backend"],
            "validation_details": {
                "status": status,
                "candidate_source": candidate_source,
                "selected_candidate": selected_candidate,
                "candidate_count": len(candidates),
                "matching_candidate_count": matching_candidate_count,
                "expected_center": expected_center,
                "thresholds": thresholds,
                "candidates": candidates,
                "model": detection_context,
            },
        }

    def _advance_event_state(
        self,
        *,
        state: SessionState,
        validation: dict[str, Any],
        part_ready_payload: dict[str, Any],
        presence: dict[str, Any],
        now: datetime,
    ) -> tuple[str, str | None, bool]:
        if not presence.get("present", False):
            state.current_presence = False
            state.current_event_id = None
            state.current_event_key = None
            state.current_event_started_at = None
            state.current_event_stable_frames = 0
            state.current_event_committed = False
            state.cooldown_until = None
            state.part_ready_ratio_history.clear()
            return InspectionEventState.IDLE.value, None, False

        event_key = (
            f"{part_ready_payload.get('part_ready')}::"
            f"{validation.get('decision')}::"
            f"{validation.get('reject_reason_code') or 'OK'}::"
            f"{validation.get('part_name') or '-'}"
        )
        if not state.current_presence or state.current_event_id is None:
            state.event_sequence += 1
            state.current_presence = True
            state.current_event_id = f"evt-{state.event_sequence:05d}"
            state.current_event_key = event_key
            state.current_event_started_at = now
            state.current_event_stable_frames = 1
            state.current_event_committed = False
        elif state.current_event_key == event_key:
            # Same outcome: honour cooldown to prevent double-counting.
            if state.current_event_committed:
                return InspectionEventState.COOLDOWN.value, state.current_event_id, False
            state.current_event_stable_frames += 1
        else:
            # Outcome changed (e.g. REJECT → ACCEPT after config fix) → new event.
            state.event_sequence += 1
            state.current_event_id = f"evt-{state.event_sequence:05d}"
            state.current_event_key = event_key
            state.current_event_started_at = now
            state.current_event_stable_frames = 1
            state.current_event_committed = False

        if not part_ready_payload.get("part_ready", False):
            return InspectionEventState.PART_DETECTED.value, state.current_event_id, False

        commit_threshold = int(getattr(state.template.sticker, "commit_stable_frames", None) or COMMIT_STABLE_FRAMES)
        if state.current_event_stable_frames >= commit_threshold:
            state.current_event_committed = True
            state.cooldown_until = now + timedelta(milliseconds=COMMIT_COOLDOWN_MS)
            return InspectionEventState.DECISION_COMMITTED.value, state.current_event_id, True

        if state.current_event_stable_frames == 1:
            return InspectionEventState.PART_READY.value, state.current_event_id, False
        return InspectionEventState.DECISION_PENDING.value, state.current_event_id, False

    def _register_committed_result(
        self,
        *,
        state: SessionState,
        validation: dict[str, Any],
        part_ready_payload: dict[str, Any],
        sticker_detection: dict[str, Any],
        db_write: dict[str, Any],
        event_id: str | None,
        part_ready_roi_meta: dict[str, Any],
        sticker_roi_meta: dict[str, Any],
        committed_at: datetime,
    ) -> None:
        self._increment_session_counters(state, validation)
        committed_payload = {
            "event_id": event_id,
            "committed_at": committed_at.isoformat(),
            "validation": dict(validation),
            "part_ready": dict(part_ready_payload),
            "sticker_detection": dict(sticker_detection),
            "part_ready_roi_meta": dict(part_ready_roi_meta),
            "sticker_roi_meta": dict(sticker_roi_meta),
            "db_write": dict(db_write),
            "count_source": "session",
        }
        state.last_committed_result = committed_payload
        state.recent_events.insert(
            0,
            {
                "event_id": event_id,
                "committed_at": committed_payload["committed_at"],
                "decision": validation.get("decision"),
                "reject_reason_code": validation.get("reject_reason_code"),
                "part_name": validation.get("part_name"),
                "line_id": validation.get("line_id"),
                "station_id": validation.get("station_id"),
                "db_written": bool(db_write.get("written")),
            },
        )
        del state.recent_events[MAX_RECENT_EVENTS:]

    def _increment_session_counters(self, state: SessionState, validation: dict[str, Any]) -> None:
        if validation.get("decision") == DecisionCode.ACCEPT.value:
            state.session_total += 1
            state.session_accept += 1
            return
        # REJECT: tracked in reject counters only, does not increment session_total
        state.session_reject += 1
        reject_reason = str(validation.get("reject_reason_code") or RejectReasonCode.ERROR.value)
        state.session_reject_breakdown.setdefault(reject_reason, 0)
        state.session_reject_breakdown[reject_reason] += 1

    def _compose_overlay(
        self,
        *,
        full_frame,
        part_ready_roi_meta: dict[str, Any],
        sticker_roi_meta: dict[str, Any],
        detections: list[dict[str, Any]],
        validation: dict[str, Any],
        part_ready: dict[str, Any],
        event_state: str,
        state: SessionState,
    ):
        overlay = full_frame.copy()
        decision = validation.get("decision")
        reject_reason = validation.get("reject_reason_code") or "OK"
        decision_color = (0, 180, 0) if decision == DecisionCode.ACCEPT.value else (0, 0, 220)

        cv2.rectangle(
            overlay,
            (int(part_ready_roi_meta["x"]), int(part_ready_roi_meta["y"])),
            (
                int(part_ready_roi_meta["x"] + part_ready_roi_meta["width"]),
                int(part_ready_roi_meta["y"] + part_ready_roi_meta["height"]),
            ),
            (50, 180, 255),
            2,
        )
        cv2.putText(
            overlay,
            "PART READY ROI",
            (int(part_ready_roi_meta["x"]), max(16, int(part_ready_roi_meta["y"]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (50, 180, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            overlay,
            (int(sticker_roi_meta["x"]), int(sticker_roi_meta["y"])),
            (
                int(sticker_roi_meta["x"] + sticker_roi_meta["width"]),
                int(sticker_roi_meta["y"] + sticker_roi_meta["height"]),
            ),
            (255, 200, 0),
            2,
        )
        cv2.putText(
            overlay,
            "STICKER ROI",
            (int(sticker_roi_meta["x"]), max(16, int(sticker_roi_meta["y"]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            overlay,
            f"{decision} / {reject_reason}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            decision_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            (
                f"part_ready={part_ready.get('part_ready')} "
                f"ratio={part_ready.get('match_ratio', part_ready.get('part_ready_confidence', '-'))}"
            ),
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"state={event_state} total={state.session_total} acc={state.session_accept} rej={state.session_reject}",
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for det in detections:
            pos = det["position"]
            x1 = int(sticker_roi_meta["x"] + float(pos["x1"]))
            y1 = int(sticker_roi_meta["y"] + float(pos["y1"]))
            x2 = int(sticker_roi_meta["x"] + float(pos["x2"]))
            y2 = int(sticker_roi_meta["y"] + float(pos["y2"]))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                overlay,
                f"{det['label']} {det['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        sticker = state.template.sticker
        cx_ratio = float(sticker.expected_center_x if sticker.expected_center_x is not None else 0.5)
        cy_ratio = float(sticker.expected_center_y if sticker.expected_center_y is not None else 0.5)
        exp_x = int(sticker_roi_meta["x"] + cx_ratio * sticker_roi_meta["width"])
        exp_y = int(sticker_roi_meta["y"] + cy_ratio * sticker_roi_meta["height"])
        arm = 18
        cv2.line(overlay, (exp_x - arm, exp_y), (exp_x + arm, exp_y), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, (exp_x, exp_y - arm), (exp_x, exp_y + arm), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (exp_x, exp_y), 5, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            "EXP",
            (exp_x + arm + 4, exp_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        return overlay

    def _maybe_persist(
        self,
        validation: dict[str, Any],
        state: SessionState,
        *,
        part_ready: dict[str, Any],
        part_ready_roi_meta: dict[str, Any],
        sticker_roi_meta: dict[str, Any],
    ) -> dict[str, Any]:
        if not state.template.persistence.write_to_db:
            return {"written": False, "reason": "disabled"}
        persist_key = (
            f"{state.current_event_id or 'event'}:"
            f"{validation.get('decision')}:"
            f"{validation.get('reject_reason_code') or 'OK'}:"
            f"{validation.get('part_name')}"
        )
        if state.last_persisted_key == persist_key:
            return {"written": False, "reason": "duplicate_event"}
        record = self._results_repo.create_result(
            {
                "template_version_id": state.template.version_id,
                "line_id": validation.get("line_id"),
                "station_id": validation.get("station_id"),
                "part_name": validation.get("part_name"),
                "mp_check": validation.get("mp_check"),
                # data1/data2 mirror SQL contract: data1=part_ready confidence, data2=sticker confidence
                "data1": validation.get("data1"),
                "data2": validation.get("data2"),
                "decision": validation.get("decision"),
                "decision_code": validation.get("decision_code"),
                "reject_reason_code": validation.get("reject_reason_code"),
                "push_status": "pending",
                "retry_count": 0,
                "operator_user_id": validation.get("operator_user_id"),
                "part_ready_status": part_ready.get("status"),
                "part_ready_match_ratio": part_ready.get("match_ratio"),
                "part_ready_distance": part_ready.get("mean_distance"),
                "detected_class": validation.get("detected_class"),
                "expected_class": validation.get("expected_class"),
                "sticker_confidence": validation.get("sticker_confidence"),
                "sticker_bbox": validation.get("sticker_bbox"),
                "sticker_backend": validation.get("sticker_backend"),
                "validation_details": validation.get("validation_details"),
                "part_ready_roi_meta": dict(part_ready_roi_meta),
                "sticker_roi_meta": dict(sticker_roi_meta),
                "targets": validation.get("targets") or [],
            }
        )
        state.last_persisted_at = datetime.now(UTC)
        state.last_persisted_key = persist_key
        return {"written": True, "result_id": record["id"]}

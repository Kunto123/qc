from __future__ import annotations

import base64
import logging
import os
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
from backend.app.repositories.reject_log_repository import RejectLogRepository
from backend.app.services.operator_state_machine import OperatorInspectionStateMachine
from backend.app.services.part_ready_detector import evaluate_color_profile_match, evaluate_hsv_black_ratio
from backend.app.services.sticker_inference import StickerInferenceService
from backend.app.services.template_runtime import TemplateRuntimeService
from backend.app.services.text_tilt import estimate_white_text_tilt
from shared.contracts.enums import DecisionCode, InspectionEventState, RejectReasonCode, SessionStatus
from shared.contracts.templates import RoiGeometry


logger = logging.getLogger(__name__)


KNOWN_REJECT_CODES = (
    RejectReasonCode.NOT_FOUND.value,
    RejectReasonCode.WRONG_TYPE.value,
    RejectReasonCode.WRONG_TEXT.value,
    RejectReasonCode.LOW_ROI_CONF.value,
    RejectReasonCode.LOW_CLASS_CONF.value,
    RejectReasonCode.LOW_OCR_CONF.value,
    RejectReasonCode.OUT_OF_POSITION.value,
    RejectReasonCode.OUT_OF_ANGLE.value,
    RejectReasonCode.ANCHOR_NOT_FOUND.value,
    RejectReasonCode.ANCHOR_MISMATCH.value,
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


_OVERLAY_JPEG_QUALITY = int(os.getenv("QC_SUITE_OVERLAY_JPEG_QUALITY", "80"))
_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, max(40, min(100, _OVERLAY_JPEG_QUALITY))]


def _encode_image(image) -> str:
    ok, encoded = cv2.imencode(".jpg", image, _ENCODE_PARAMS)
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


def _estimate_tilt_from_roi(roi_frame, expected_tilt_degrees: float, config: Any | None = None) -> dict[str, Any]:
    if config is not None:
        return estimate_white_text_tilt(roi_frame, expected_tilt_degrees, config)
    if roi_frame is None or getattr(roi_frame, "size", 0) == 0:
        return {
            "status": "unavailable",
            "angle_degrees": None,
            "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
            "deviation_degrees": None,
            "contour_area": None,
            "threshold_mode": None,
        }

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates: list[dict[str, Any]] = []
    for threshold_mode in (cv2.THRESH_BINARY + cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU):
        try:
            _, thresh = cv2.threshold(gray, 0, 255, threshold_mode)
        except Exception:  # noqa: BLE001
            continue
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue
        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        angle_degrees = float(rect[2])
        if rect_w < rect_h:
            angle_degrees = 90.0 + angle_degrees
        candidates.append(
            {
                "angle_degrees": angle_degrees,
                "contour_area": contour_area,
                "threshold_mode": "binary_inv" if threshold_mode == cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU else "binary",
            }
        )

    if not candidates:
        return {
            "status": "unavailable",
            "angle_degrees": None,
            "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
            "deviation_degrees": None,
            "contour_area": None,
            "threshold_mode": None,
        }

    best = max(candidates, key=lambda item: float(item.get("contour_area") or 0.0))
    angle_degrees = float(best["angle_degrees"])
    deviation_degrees = abs(angle_degrees - float(expected_tilt_degrees))
    return {
        "status": "ok",
        "angle_degrees": round(angle_degrees, 2),
        "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
        "deviation_degrees": round(deviation_degrees, 2),
        "contour_area": round(float(best["contour_area"]), 2),
        "threshold_mode": best["threshold_mode"],
    }


class InspectionSessionService:
    def __init__(
        self,
        template_runtime: TemplateRuntimeService,
        profiles_repo: ProfilesRepository,
        results_repo: InspectionResultsRepository,
        sticker_inference: StickerInferenceService,
        app_config: AppConfig | None = None,
        plc_worker=None,
        reject_log_repo: RejectLogRepository | None = None,
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
        self._default_ocr_mode = (
            str(getattr(app_config, "sticker_ocr_mode", "legacy") or "legacy").strip().lower()
            if app_config is not None
            else "legacy"
        )
        self._default_ocr_min_confidence = (
            max(0.0, min(1.0, float(getattr(app_config, "default_ocr_min_confidence", 0.70))))
            if app_config is not None
            else 0.70
        )
        self._plc_worker = plc_worker
        self._reject_log_repo = reject_log_repo
        self._operator_state_machine = OperatorInspectionStateMachine()

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

    def has_session(self, session_id: str) -> bool:
        """Return True if a session with this id is currently active."""
        with self._lock:
            return session_id in self._sessions

    def process_frame(
        self,
        session_id: str,
        *,
        image_b64: str,
        response_mode: str | None = None,
        username: str | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        decode_started = time.perf_counter()
        frame = _decode_image(image_b64)
        decode_ms = round((time.perf_counter() - decode_started) * 1000.0, 2)
        return self.process_frame_decoded(
            session_id,
            frame=frame,
            decode_ms=decode_ms,
            response_mode=response_mode,
            username=username,
            user_id=user_id,
        )

    def process_frame_decoded(
        self,
        session_id: str,
        *,
        frame,
        decode_ms: float = 0.0,
        response_mode: str | None = None,
        username: str | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Process a pre-decoded numpy BGR frame for the given session.

        Called by ``process_frame`` (which decodes base64 first) and by the
        WebSocket streaming handler (which receives raw JPEG bytes directly).
        ``decode_ms`` carries the caller's decode timing so it is reflected in
        the returned timings payload.
        """
        total_started = time.perf_counter()
        timings: dict[str, float] = {"decode_ms": float(decode_ms)}

        def _elapsed_ms(started_at: float) -> float:
            return round((time.perf_counter() - started_at) * 1000.0, 2)

        state = self._require_session(session_id)
        state.frame_index += 1

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
            state.plc_part_ready_triggered = False

        # Trigger PLC clamp hold on the first frame where part is settled.
        if part_ready_settled and not state.plc_part_ready_triggered:
            state.plc_part_ready_triggered = True
            if self._plc_worker is not None:
                self._plc_worker.enqueue_part_ready(event_id=state.current_event_id)

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

        operator_state_decision = self._operator_state_machine.update(
            state,
            part_ready=bool(_raw_part_ready),
            present=bool(presence.get("present", False)),
            settled=bool(part_ready_settled),
        )
        if operator_state_decision.use_cached_result and state.inspection_result_cache:
            cached_payload = dict(state.inspection_result_cache)
            cached_timings = dict(cached_payload.get("timings") or {})
            cached_timings.update(
                {
                    **timings,
                    "inference_ms": 0.0,
                    "inference_skipped": True,
                    "operator_state": operator_state_decision.state.value,
                    "total_ms": _elapsed_ms(total_started),
                }
            )
            cached_payload.update(
                {
                    "session": self._session_payload(state),
                    "presence": presence,
                    "part_ready": part_ready,
                    "event_state": InspectionEventState.COOLDOWN.value,
                    "operator_state": operator_state_decision.state.value,
                    "count_committed": False,
                    "count_source": None,
                    "counters": self._counter_payload(state),
                    "last_committed_result": state.last_committed_result,
                    "recent_events": list(state.recent_events),
                    "timings": cached_timings,
                }
            )
            state.latest_result = cached_payload
            return cached_payload

        _effective_pr_ready = bool(operator_state_decision.run_inspection)
        detections: list[dict[str, Any]] = []
        inference_ms = 0.0
        if _effective_pr_ready:
            inference_started = time.perf_counter()
            # DEBUG: log sticker ROI and frame info
            logger.info(
                "[inference] sticker_roi=%s frame=%dx%d crop=%dx%d",
                state.template.sticker_roi,
                frame.shape[1], frame.shape[0],
                sticker_frame.shape[1], sticker_frame.shape[0],
            )
            inference_payload = self._run_sticker_inference(sticker_frame, state)
            inference_ms = _elapsed_ms(inference_started)
            detections = list(inference_payload.get("detections") or [])
            # DEBUG: log detection results
            logger.info(
                "[inference] raw=%d filtered=%d expected_class=%s",
                inference_payload.get("raw_detection_count", 0),
                len(detections),
                state.template.sticker.expected_class,
            )
            stage_timings = dict(inference_payload.get("timings") or {})
            for key, value in stage_timings.items():
                try:
                    timings[f"inference_{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
            sticker_detection = self._build_sticker_detection_payload(
                detections,
                skipped=False,
                backend=str(inference_payload.get("backend") or "unknown"),
                model_path=inference_payload.get("model_path"),
                meta_path=inference_payload.get("meta_path"),
                class_names=inference_payload.get("class_names") or [],
                fallback_reason=inference_payload.get("fallback_reason"),
                raw_detection_count=inference_payload.get("raw_detection_count"),
                allowed_labels_filter=inference_payload.get("allowed_labels_filter"),
                anchor=inference_payload.get("anchor"),
                ocr=inference_payload.get("ocr"),
                geometry=inference_payload.get("geometry"),
                stage_timings=stage_timings,
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
        validation = self._attach_ocr_observability(validation, sticker_detection, state)
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
            settle_ms=settle_ms,
        )
        timings["event_state_ms"] = _elapsed_ms(event_state_started)

        persistence_started = time.perf_counter()
        db_write = {"written": False, "reason": "not_committed"}
        if count_committed:
            # Notify PLC worker of inspection decision
            decision = validation.get("decision", "")
            if self._plc_worker is not None and decision in ("ACCEPT", "REJECT"):
                try:
                    self._plc_worker.notify_decision(decision, event_id=event_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error("[inspection] plc_worker.notify_decision failed: %s", exc)
            db_write = self._maybe_persist(
                validation,
                state,
                part_ready=part_ready,
                part_ready_roi_meta=part_ready_roi_meta,
                sticker_roi_meta=sticker_roi_meta,
                    sticker_detection=sticker_detection,
                    event_id=event_id,
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

        normalized_response_mode = str(response_mode or "").strip().lower()
        # "stream": skip overlay compose and encode entirely — client renders locally.
        # "compact"/"minimal"/"overlay": encode overlay but skip heavy preview images.
        # (default) "full": encode everything.
        stream_response = normalized_response_mode in {"stream"}
        compact_response = stream_response or normalized_response_mode in {"compact", "minimal", "overlay"}

        overlay_started = time.perf_counter()
        overlay_image_b64 = ""
        if not stream_response:
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
            overlay_image_b64 = _encode_image(overlay)
            state.last_overlay_b64 = overlay_image_b64
        timings["overlay_compose_ms"] = _elapsed_ms(overlay_started)

        encode_started = time.perf_counter()
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
            "operator_state": state.operator_state,
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
        if count_committed:
            payload["operator_state"] = "RESULT"
            self._operator_state_machine.mark_result(state, payload)
            payload["operator_state"] = state.operator_state
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
            "operator_state": state.operator_state,
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
            sticker_rule=state.template.sticker,
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
        raw_detection_count: int | None = None,
        allowed_labels_filter: list[str] | None = None,
        anchor: dict[str, Any] | None = None,
        ocr: dict[str, Any] | None = None,
        geometry: dict[str, Any] | None = None,
        stage_timings: dict[str, Any] | None = None,
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
            "raw_detection_count": raw_detection_count,
            "allowed_labels_filter": allowed_labels_filter,
            "anchor": dict(anchor or {}),
            "ocr": dict(ocr or {}),
            "geometry": dict(geometry or {}),
            "stage_timings": dict(stage_timings or {}),
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
            if str(getattr(config, "method", "") or "").strip().lower() == "hsv_black_ratio":
                evaluation = evaluate_hsv_black_ratio(frame, config)
                raw_ratio = float(evaluation["match_ratio"])
                _PART_READY_WINDOW = 5
                history = state.part_ready_ratio_history
                history.append(raw_ratio)
                if len(history) > _PART_READY_WINDOW:
                    del history[0]
                smoothed_ratio = round(sum(history) / len(history), 6)
                resolved_min = float(evaluation["min_match_ratio"])
                ready = smoothed_ratio >= resolved_min
                evaluation.update(
                    {
                        "part_ready": ready,
                        "part_ready_confidence": smoothed_ratio,
                        "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
                        "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
                        "status": "ready" if ready else "not_ready",
                        "match_ratio": smoothed_ratio,
                        "raw_match_ratio": raw_ratio,
                    }
                )
                return evaluation
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
        evaluation = evaluate_color_profile_match(frame, config=config, profile=record["profile"])
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
            **evaluation,
            "part_ready": ready,
            "part_ready_confidence": smoothed_ratio,
            "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
            "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
            "status": "ready" if ready else "not_ready",
            "match_ratio": smoothed_ratio,
            "raw_match_ratio": raw_ratio,
            "min_match_ratio": resolved_min,
        }

    def _normalize_label(self, value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _normalize_tilt_180(angle: float | None) -> float | None:
        if angle is None:
            return None
        normalized = float(angle)
        while normalized > 90.0:
            normalized -= 180.0
        while normalized < -90.0:
            normalized += 180.0
        if abs(normalized) == 0:
            normalized = 0.0
        return round(normalized, 2)

    @staticmethod
    def _normalize_code(value: Any) -> str:
        return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())

    def _resolve_ocr_mode(self, state: SessionState) -> str:
        sticker = state.template.sticker
        explicit = str(getattr(sticker, "ocr_mode", "") or "").strip().lower()
        validator_mode = str(getattr(sticker, "validator_mode", "") or "").strip().lower()
        raw_mode = explicit or self._default_ocr_mode
        if bool(getattr(sticker, "use_ocr", False)) and not explicit:
            raw_mode = "primary"
        if not explicit and validator_mode in {"ocr", "ocr_anchor", "anchor_ocr", "ocr_primary"}:
            raw_mode = "primary"
        if raw_mode in {"primary", "ocr", "ocr_primary", "anchor_ocr"}:
            return "primary"
        if raw_mode in {"shadow", "ocr_shadow"}:
            return "shadow"
        return "legacy"

    @staticmethod
    def _ocr_validation_fields(detection_payload: dict[str, Any]) -> dict[str, Any]:
        anchor = detection_payload.get("anchor") or {}
        ocr = detection_payload.get("ocr") or {}
        geometry = detection_payload.get("geometry") or {}
        return {
            "ocr_text": ocr.get("canonical_text") or ocr.get("text") or None,
            "ocr_confidence": ocr.get("confidence"),
            "ocr_engine": ocr.get("engine"),
            "ocr_status": ocr.get("status"),
            "text_bbox": anchor.get("text_bbox"),
            "dot_bbox": anchor.get("dot_bbox"),
            "dot_position": geometry.get("dot_position") or anchor.get("dot_position"),
            "anchor_offset": geometry.get("anchor_offset"),
            "pose_angle": geometry.get("pose_angle"),
        }

    def _attach_ocr_observability(
        self,
        validation: dict[str, Any],
        detection_payload: dict[str, Any],
        state: SessionState,
    ) -> dict[str, Any]:
        result = dict(validation)
        for key, value in self._ocr_validation_fields(detection_payload).items():
            if result.get(key) is None:
                result[key] = value
        details = result.get("validation_details")
        if isinstance(details, dict):
            ocr_mode = self._resolve_ocr_mode(state)
            ocr_payload = dict(detection_payload.get("ocr") or {})
            anchor_payload = dict(detection_payload.get("anchor") or {})
            geometry_payload = dict(detection_payload.get("geometry") or {})
            details.setdefault("ocr_mode", ocr_mode)
            details.setdefault("ocr", ocr_payload)
            details.setdefault("anchor", anchor_payload)
            details.setdefault("geometry", geometry_payload)
            if ocr_mode == "shadow":
                details["ocr_shadow"] = {
                    "anchor_status": anchor_payload.get("status"),
                    "ocr_status": ocr_payload.get("status"),
                    "geometry_status": geometry_payload.get("status"),
                    "ocr_text": ocr_payload.get("canonical_text") or ocr_payload.get("text"),
                    "ocr_confidence": ocr_payload.get("confidence"),
                    "match_expected": bool(ocr_payload.get("match_expected")),
                    "anchor_offset": geometry_payload.get("anchor_offset"),
                    "pose_angle": geometry_payload.get("pose_angle"),
                }
        return result

    def _validate_ocr_anchor(
        self,
        *,
        state: SessionState,
        detection_payload: dict[str, Any],
        part_ready_payload: dict[str, Any],
        username: str | None,
        user_id: int | None,
        line_id: str,
        thresholds: dict[str, Any],
        detection_context: dict[str, Any],
        max_tilt_degrees_value: float | None,
    ) -> dict[str, Any]:
        sticker = state.template.sticker
        anchor = detection_payload.get("anchor") or {}
        ocr = detection_payload.get("ocr") or {}
        geometry = detection_payload.get("geometry") or {}

        anchor_min_confidence = (
            thresholds["min_roi_confidence"]
            if getattr(sticker, "anchor_min_confidence", None) is None
            else float(getattr(sticker, "anchor_min_confidence") or 0.0)
        )
        dot_min_confidence = (
            anchor_min_confidence
            if getattr(sticker, "dot_min_confidence", None) is None
            else float(getattr(sticker, "dot_min_confidence") or 0.0)
        )
        ocr_min_confidence = (
            self._default_ocr_min_confidence
            if getattr(sticker, "ocr_min_confidence", None) is None
            else float(getattr(sticker, "ocr_min_confidence") or 0.0)
        )
        offset_limit_x = (
            getattr(sticker, "max_anchor_offset_x", None)
            if getattr(sticker, "max_anchor_offset_x", None) is not None
            else sticker.max_offset_x
        )
        offset_limit_y = (
            getattr(sticker, "max_anchor_offset_y", None)
            if getattr(sticker, "max_anchor_offset_y", None) is not None
            else sticker.max_offset_y
        )
        offset_x = None
        offset_y = None
        anchor_offset = geometry.get("anchor_offset") or {}
        if anchor_offset:
            offset_x = float(anchor_offset.get("x", 0.0))
            offset_y = float(anchor_offset.get("y", 0.0))

        reject_reason = None
        if anchor.get("text_anchor") is None or anchor.get("center_dot") is None:
            reject_reason = RejectReasonCode.ANCHOR_NOT_FOUND.value
        elif float(anchor.get("text_confidence") or 0.0) < anchor_min_confidence:
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif float(anchor.get("dot_confidence") or 0.0) < dot_min_confidence:
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif str(ocr.get("status") or "") != "ok":
            reject_reason = RejectReasonCode.LOW_OCR_CONF.value
        elif ocr.get("confidence") is not None and float(ocr.get("confidence") or 0.0) < ocr_min_confidence:
            reject_reason = RejectReasonCode.LOW_OCR_CONF.value
        elif not bool(ocr.get("match_expected")):
            reject_reason = RejectReasonCode.WRONG_TEXT.value
        elif offset_x is None or offset_y is None:
            reject_reason = RejectReasonCode.ANCHOR_MISMATCH.value
        elif offset_limit_x is not None and abs(offset_x) > float(offset_limit_x):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        elif offset_limit_y is not None and abs(offset_y) > float(offset_limit_y):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        elif (
            thresholds["tilt_gate_enabled"]
            and max_tilt_degrees_value is not None
            and geometry.get("pose_deviation") is not None
            and float(geometry.get("pose_deviation") or 0.0) > max_tilt_degrees_value
        ):
            reject_reason = RejectReasonCode.OUT_OF_ANGLE.value

        decision = DecisionCode.ACCEPT.value if reject_reason is None else DecisionCode.REJECT.value
        status = "accepted" if reject_reason is None else reject_reason.lower()
        detected_text = ocr.get("canonical_text") or ocr.get("text")
        selected_candidate = {
            "label": detected_text,
            "normalized_label": self._normalize_label(detected_text),
            "confidence": ocr.get("confidence"),
            "class_confidence": ocr.get("confidence"),
            "bbox": anchor.get("text_bbox"),
            "center": geometry.get("dot_position") or anchor.get("dot_position"),
            "offset": {"x": round(offset_x or 0.0, 2), "y": round(offset_y or 0.0, 2)} if offset_x is not None and offset_y is not None else None,
            "match_expected": bool(ocr.get("match_expected")),
            "source": "ocr_anchor",
        }
        target = {
            "target_id": "target-1",
            "part_name": sticker.part_name,
            "expected_class": sticker.expected_class,
            "detected_class": detected_text,
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "data1": ocr.get("confidence"),
            "data2": anchor.get("dot_confidence"),
            "position": geometry.get("dot_position") or {},
            "offset": selected_candidate.get("offset") or {},
            "candidate_source": "ocr_anchor",
        }
        return {
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "part_name": sticker.part_name,
            "line_id": line_id,
            "station_id": state.station_id,
            "data1": part_ready_payload.get("part_ready_confidence"),
            "data2": ocr.get("confidence"),
            "targets": [target],
            "operator_user_id": user_id,
            "mp_check": username,
            "detected_class": detected_text,
            "expected_class": sticker.expected_class,
            "sticker_confidence": ocr.get("confidence"),
            "sticker_bbox": anchor.get("text_bbox"),
            "sticker_backend": detection_context["backend"],
            "sticker_tilt_angle": geometry.get("pose_angle"),
            "sticker_tilt_expected": thresholds["expected_tilt_degrees"],
            "sticker_tilt_deviation": geometry.get("pose_deviation"),
            "sticker_tilt_threshold": max_tilt_degrees_value,
            "validation_details": {
                "status": status,
                "candidate_source": "ocr_anchor",
                "selected_candidate": selected_candidate,
                "candidate_count": int(detection_payload.get("count") or 0),
                "matching_candidate_count": 1 if ocr.get("match_expected") else 0,
                "expected_center": geometry.get("expected_dot_position"),
                "tilt": {
                    "status": geometry.get("status"),
                    "angle_degrees": geometry.get("pose_angle"),
                    "expected_tilt_degrees": thresholds["expected_tilt_degrees"],
                    "deviation_degrees": geometry.get("pose_deviation"),
                    "source": "anchor_geometry",
                },
                "thresholds": {
                    **thresholds,
                    "anchor_min_confidence": anchor_min_confidence,
                    "dot_min_confidence": dot_min_confidence,
                    "ocr_min_confidence": ocr_min_confidence,
                    "max_anchor_offset_x": None if offset_limit_x is None else float(offset_limit_x),
                    "max_anchor_offset_y": None if offset_limit_y is None else float(offset_limit_y),
                },
                "ocr": ocr,
                "anchor": anchor,
                "geometry": geometry,
                "model": detection_context,
            },
        }

    def _validate_sticker_ocr_only(
        self,
        *,
        roi_frame,
        state: SessionState,
        detections: list[dict[str, Any]],
        detection_payload: dict[str, Any],
        part_ready_payload: dict[str, Any],
        username: str | None,
        user_id: int | None,
        line_id: str,
        thresholds: dict[str, Any],
        detection_context: dict[str, Any],
        max_tilt_degrees_value: float | None,
    ) -> dict[str, Any]:
        sticker = state.template.sticker
        candidates, expected_center = self._build_validation_candidate_summaries(
            detections,
            roi_frame,
            sticker.expected_class,
            sticker,
        )
        selected_candidate, candidate_source = self._select_validation_candidate(candidates)
        matching_candidate_count = sum(1 for item in candidates if item.get("match_expected"))

        tilt_info = dict(detection_payload.get("tilt_info") or {})
        geometry = detection_payload.get("geometry") or {}
        if not tilt_info:
            tilt_info = {
                "status": geometry.get("status"),
                "angle_degrees": geometry.get("pose_angle"),
                "expected_tilt_degrees": thresholds["expected_tilt_degrees"],
                "deviation_degrees": geometry.get("pose_deviation"),
                "source": "sticker_only_geometry",
            }
        raw_angle = tilt_info.get("angle_degrees")
        normalized_angle = self._normalize_tilt_180(None if raw_angle is None else float(raw_angle))
        expected_tilt = float(thresholds["expected_tilt_degrees"] or 0.0)
        normalized_deviation = None
        if normalized_angle is not None:
            normalized_deviation = round(abs(float(normalized_angle) - expected_tilt), 2)
        tilt_info["normalized_angle_degrees"] = normalized_angle
        tilt_info["normalized_deviation_degrees"] = normalized_deviation

        anchor_offset = geometry.get("anchor_offset") or {}
        if not anchor_offset and selected_candidate is not None:
            anchor_offset = selected_candidate.get("offset") or {}
        offset_x = float(anchor_offset.get("x", 0.0)) if anchor_offset else 0.0
        offset_y = float(anchor_offset.get("y", 0.0)) if anchor_offset else 0.0
        offset_limit_x = (
            getattr(sticker, "max_anchor_offset_x", None)
            if getattr(sticker, "max_anchor_offset_x", None) is not None
            else thresholds["max_offset_x"]
        )
        offset_limit_y = (
            getattr(sticker, "max_anchor_offset_y", None)
            if getattr(sticker, "max_anchor_offset_y", None) is not None
            else thresholds["max_offset_y"]
        )

        ocr_payload = detection_payload.get("ocr") or {}
        unique_code = str(detection_payload.get("unique_code") or "").strip()
        expected_code = str(
            getattr(sticker, "ocr_expected_code", "")
            or getattr(sticker, "ocr_expected_text", "")
            or sticker.expected_class
            or ""
        ).strip()
        use_ocr = bool(getattr(sticker, "use_ocr", False))
        ocr_min_confidence = (
            self._default_ocr_min_confidence
            if getattr(sticker, "ocr_min_confidence", None) is None
            else float(getattr(sticker, "ocr_min_confidence") or 0.0)
        )

        reject_reason = None
        if selected_candidate is None:
            reject_reason = RejectReasonCode.NOT_FOUND.value
        elif float(selected_candidate.get("confidence") or 0.0) < thresholds["min_roi_confidence"]:
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif not bool(selected_candidate.get("match_expected")):
            reject_reason = RejectReasonCode.WRONG_TYPE.value
        elif thresholds["min_class_confidence"] is not None and float(selected_candidate.get("class_confidence") or 0.0) < float(thresholds["min_class_confidence"]):
            reject_reason = RejectReasonCode.LOW_CLASS_CONF.value
        elif thresholds["tilt_gate_enabled"] and max_tilt_degrees_value is not None and normalized_deviation is not None and normalized_deviation > max_tilt_degrees_value:
            reject_reason = RejectReasonCode.OUT_OF_ANGLE.value
        elif offset_limit_x is not None and abs(offset_x) > float(offset_limit_x):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        elif offset_limit_y is not None and abs(offset_y) > float(offset_limit_y):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        elif use_ocr and str(ocr_payload.get("status") or "") != "ok":
            reject_reason = RejectReasonCode.LOW_OCR_CONF.value
        elif use_ocr and ocr_payload.get("confidence") is not None and float(ocr_payload.get("confidence") or 0.0) < ocr_min_confidence:
            reject_reason = RejectReasonCode.LOW_OCR_CONF.value
        elif use_ocr and expected_code and self._normalize_code(unique_code) != self._normalize_code(expected_code):
            reject_reason = RejectReasonCode.WRONG_TEXT.value

        decision = DecisionCode.ACCEPT.value if reject_reason is None else DecisionCode.REJECT.value
        status = "accepted" if reject_reason is None else reject_reason.lower()
        bbox = dict((selected_candidate or {}).get("bbox") or {}) or None
        selected_label = None if selected_candidate is None else selected_candidate.get("label")
        confidence = None if selected_candidate is None else selected_candidate.get("confidence")
        selected_center = dict((selected_candidate or {}).get("center") or {})
        selected_offset = {"x": round(offset_x, 2), "y": round(offset_y, 2)} if selected_candidate is not None else {}
        target = {
            "target_id": "target-1",
            "part_name": sticker.part_name,
            "expected_class": sticker.expected_class,
            "detected_class": selected_label,
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "data1": confidence,
            "data2": None if selected_candidate is None else selected_candidate.get("class_confidence"),
            "position": selected_center,
            "offset": selected_offset,
            "candidate_source": candidate_source,
        }
        result = {
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "part_name": sticker.part_name,
            "line_id": line_id,
            "station_id": state.station_id,
            "data1": part_ready_payload.get("part_ready_confidence"),
            "data2": confidence,
            "targets": [] if selected_candidate is None else [target],
            "operator_user_id": user_id,
            "mp_check": username,
            "detected_class": selected_label,
            "expected_class": sticker.expected_class,
            "unique_code": unique_code,
            "sticker_confidence": confidence,
            "sticker_bbox": bbox,
            "sticker_backend": detection_context["backend"],
            "sticker_tilt_angle": normalized_angle,
            "sticker_tilt_expected": expected_tilt,
            "sticker_tilt_deviation": normalized_deviation,
            "sticker_tilt_threshold": max_tilt_degrees_value,
            "validation_details": {
                "status": status,
                "candidate_source": "sticker_only" if candidate_source == "expected_class" else candidate_source,
                "selected_candidate": selected_candidate,
                "candidate_count": len(candidates),
                "matching_candidate_count": matching_candidate_count,
                "expected_center": expected_center,
                "tilt": tilt_info,
                "thresholds": {
                    **thresholds,
                    "ocr_min_confidence": ocr_min_confidence,
                    "max_anchor_offset_x": None if offset_limit_x is None else float(offset_limit_x),
                    "max_anchor_offset_y": None if offset_limit_y is None else float(offset_limit_y),
                },
                "candidates": candidates,
                "ocr": ocr_payload,
                "anchor": detection_payload.get("anchor") or {},
                "geometry": geometry,
                "unique_code": unique_code,
                "model": detection_context,
            },
        }
        for key, value in self._ocr_validation_fields(detection_payload).items():
            result[key] = value
        return result

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
        expected_tilt_degrees = float(getattr(sticker, "expected_tilt_degrees", 0.0) or 0.0)
        max_tilt_degrees = getattr(sticker, "max_tilt_degrees", None)
        max_tilt_degrees_value = None if max_tilt_degrees is None else float(max_tilt_degrees)
        tilt_gate_enabled = bool(getattr(sticker, "tilt_gate_enabled", False))
        tilt_info = _estimate_tilt_from_roi(roi_frame, expected_tilt_degrees, sticker)
        ocr_mode = self._resolve_ocr_mode(state)
        thresholds = {
            "min_roi_confidence": float(sticker.min_roi_confidence or 0.0),
            "min_class_confidence": (
                None if sticker.min_class_confidence is None else float(sticker.min_class_confidence)
            ),
            "max_offset_x": None if sticker.max_offset_x is None else float(sticker.max_offset_x),
            "max_offset_y": None if sticker.max_offset_y is None else float(sticker.max_offset_y),
            "validator_mode": validator_mode,
            "position_gate_enabled": position_gate_enabled,
            "tilt_gate_enabled": tilt_gate_enabled,
            "max_tilt_degrees": max_tilt_degrees_value,
            "expected_tilt_degrees": expected_tilt_degrees,
            "ocr_mode": ocr_mode,
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
                "sticker_tilt_angle": tilt_info.get("angle_degrees"),
                "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
                "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
                "sticker_tilt_threshold": max_tilt_degrees_value,
                "validation_details": {
                    "status": "disabled",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": len(detections),
                    "matching_candidate_count": 0,
                    "expected_center": None,
                    "tilt": tilt_info,
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
                "sticker_tilt_angle": tilt_info.get("angle_degrees"),
                "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
                "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
                "sticker_tilt_threshold": max_tilt_degrees_value,
                "validation_details": {
                    "status": "part_not_ready",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": len(detections),
                    "matching_candidate_count": 0,
                    "expected_center": None,
                    "tilt": tilt_info,
                    "thresholds": thresholds,
                },
            }
        if bool(getattr(sticker, "use_ocr", False)) or validator_mode in {"sticker_only", "ocr_only", "ocr_sticker", "sticker_ocr"}:
            return self._validate_sticker_ocr_only(
                roi_frame=roi_frame,
                state=state,
                detections=detections,
                detection_payload=detection_payload,
                part_ready_payload=part_ready_payload,
                username=username,
                user_id=user_id,
                line_id=line_id,
                thresholds=thresholds,
                detection_context=detection_context,
                max_tilt_degrees_value=max_tilt_degrees_value,
            )
        if ocr_mode == "primary":
            return self._validate_ocr_anchor(
                state=state,
                detection_payload=detection_payload,
                part_ready_payload=part_ready_payload,
                username=username,
                user_id=user_id,
                line_id=line_id,
                thresholds=thresholds,
                detection_context=detection_context,
                max_tilt_degrees_value=max_tilt_degrees_value,
            )
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
                "sticker_tilt_angle": tilt_info.get("angle_degrees"),
                "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
                "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
                "sticker_tilt_threshold": max_tilt_degrees_value,
                "validation_details": {
                    "status": "not_found",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": 0,
                    "matching_candidate_count": 0,
                    "expected_center": None,
                    "tilt": tilt_info,
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
                "sticker_tilt_angle": tilt_info.get("angle_degrees"),
                "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
                "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
                "sticker_tilt_threshold": max_tilt_degrees_value,
                "validation_details": {
                    "status": "not_found",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": len(candidates),
                    "matching_candidate_count": matching_candidate_count,
                    "expected_center": expected_center,
                    "tilt": tilt_info,
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
        elif tilt_gate_enabled and max_tilt_degrees_value is not None and tilt_info.get("angle_degrees") is not None and float(tilt_info.get("deviation_degrees") or 0.0) > max_tilt_degrees_value:
            reject_reason = RejectReasonCode.OUT_OF_ANGLE.value
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
            "sticker_tilt_angle": tilt_info.get("angle_degrees"),
            "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
            "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
            "sticker_tilt_threshold": max_tilt_degrees_value,
            "validation_details": {
                "status": status,
                "candidate_source": candidate_source,
                "selected_candidate": selected_candidate,
                "candidate_count": len(candidates),
                "matching_candidate_count": matching_candidate_count,
                "expected_center": expected_center,
                "tilt": tilt_info,
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
        settle_ms: int = 0,
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

        # Time-based commit driven by settle_ms (the same value that gates inference).
        # settle_ms=0 → commit immediately on the first stable post-ready frame.
        # settle_ms>0 → commit only after the current event has been continuously
        # stable for at least settle_ms milliseconds since current_event_started_at.
        if settle_ms > 0 and state.current_event_started_at is not None:
            elapsed_event_ms = (now - state.current_event_started_at).total_seconds() * 1000.0
            commit_ready = elapsed_event_ms >= settle_ms
        else:
            commit_ready = True

        if commit_ready:
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
        ocr_text = validation.get("ocr_text") or "-"
        ocr_conf = validation.get("ocr_confidence")
        cv2.putText(
            overlay,
            f"ocr={ocr_text} conf={ocr_conf if ocr_conf is not None else '-'}",
            (12, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        anchor_offset = validation.get("anchor_offset") or {}
        if anchor_offset:
            cv2.putText(
                overlay,
                f"anchor dx={anchor_offset.get('x', '-')} dy={anchor_offset.get('y', '-')}",
                (12, 114),
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
        sticker_detection: dict[str, Any],
        event_id: str | None,
    ) -> dict[str, Any]:
        decision = str(validation.get("decision") or "").strip().upper()
        if decision != DecisionCode.ACCEPT.value:
            reject_log_written = False
            reject_entry = None
            if self._reject_log_repo is not None:
                try:
                    reject_entry = self._reject_log_repo.log_reject(
                        {
                            "session_id": state.session_id,
                            "event_id": event_id or state.current_event_id,
                            "template_version_id": state.template.version_id,
                            "line_id": validation.get("line_id") or state.line_id,
                            "station_id": validation.get("station_id") or state.station_id,
                            "part_name": validation.get("part_name"),
                            "decision_code": decision or DecisionCode.REJECT.value,
                            "reject_reason_code": validation.get("reject_reason_code") or RejectReasonCode.ERROR.value,
                            "operator_user_id": validation.get("operator_user_id"),
                            "mp_check": validation.get("mp_check"),
                            "validation_details": validation.get("validation_details"),
                            "part_ready": part_ready,
                            "sticker_detection": sticker_detection,
                            "part_ready_roi_meta": dict(part_ready_roi_meta),
                            "sticker_roi_meta": dict(sticker_roi_meta),
                        }
                    )
                    reject_log_written = True
                except Exception as exc:  # noqa: BLE001
                    logger.error("[inspection] reject log write failed: %s", exc, exc_info=True)
            return {
                "written": False,
                "reason": "reject_logged" if reject_log_written else "reject_log_error",
                "reject_log_written": reject_log_written,
                "reject_log_entry": reject_entry,
            }

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
                "ocr_text": validation.get("ocr_text"),
                "ocr_confidence": validation.get("ocr_confidence"),
                "ocr_engine": validation.get("ocr_engine"),
                "ocr_status": validation.get("ocr_status"),
                "text_bbox": validation.get("text_bbox"),
                "dot_bbox": validation.get("dot_bbox"),
                "dot_position": validation.get("dot_position"),
                "anchor_offset": validation.get("anchor_offset"),
                "pose_angle": validation.get("pose_angle"),
                "validation_details": validation.get("validation_details"),
                "part_ready_roi_meta": dict(part_ready_roi_meta),
                "sticker_roi_meta": dict(sticker_roi_meta),
                "targets": validation.get("targets") or [],
            }
        )
        state.last_persisted_at = datetime.now(UTC)
        state.last_persisted_key = persist_key
        return {"written": True, "result_id": record["id"]}

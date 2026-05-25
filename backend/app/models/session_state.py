from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from shared.contracts.enums import SessionStatus
from shared.contracts.templates import InspectionTemplate


@dataclass(slots=True)
class SessionState:
    session_id: str
    client_id: str
    camera_index: int
    template: InspectionTemplate
    status: SessionStatus = SessionStatus.IDLE
    line_id: str | None = None
    station_id: str | None = None
    part_ready_roi_override: dict[str, Any] = field(default_factory=dict)
    sticker_roi_override: dict[str, Any] = field(default_factory=dict)
    latest_result: dict[str, Any] | None = None
    last_persisted_at: datetime | None = None
    last_persisted_key: str | None = None
    frame_index: int = 0
    current_presence: bool = False
    current_event_id: str | None = None
    current_event_key: str | None = None
    current_event_started_at: datetime | None = None
    current_event_stable_frames: int = 0
    current_event_committed: bool = False
    cooldown_until: datetime | None = None
    event_sequence: int = 0
    session_total: int = 0
    session_accept: int = 0
    session_reject: int = 0
    session_reject_breakdown: dict[str, int] = field(default_factory=dict)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    last_committed_result: dict[str, Any] | None = None
    part_ready_ratio_history: list[float] = field(default_factory=list)
    last_overlay_b64: str | None = None
    # Settle-time debounce: timestamp of the first frame where part_ready was True
    # in the current ready-run.  Reset to None whenever part_ready becomes False or
    # presence is lost.
    part_ready_settle_started_at: datetime | None = None
    # PLC constant-output mode: True once enqueue_part_ready() has been called for
    # the current ready-run, preventing duplicate triggers.  Reset when part leaves.
    plc_part_ready_triggered: bool = False
    operator_state: str = "IDLE"
    inspection_has_run_for_current_part: bool = False
    inspection_result_cache: dict[str, Any] | None = None
    part_removed_seen_at: datetime | None = None
    # Hysteresis counter: number of consecutive settled frames.
    # Reset to 0 when part_ready/presence is lost.
    settle_frame_count: int = 0
    # Inference cooldown: timestamp (ms) of last inference run.
    # Prevents inference from running more than once per second.
    last_inference_ms: int = 0
    # Inference interval (ms): minimum time between inference runs.
    # 0 = unlimited (every frame), 200 = max ~5 fps inference.
    inference_interval_ms: int = 200
    # Manual release COOLDOWN: timestamp (seconds since epoch) until which
    # re-clamp is blocked after IN1 (manual release). Prevents instant re-clamp.
    manual_release_cooldown_until: float = 0.0

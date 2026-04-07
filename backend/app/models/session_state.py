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

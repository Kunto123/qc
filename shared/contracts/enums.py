from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "admin"
    OPERATOR = "operator"
    ENGINEER = "engineer"


class DecisionCode(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ERROR = "ERROR"


class RejectReasonCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    WRONG_TYPE = "WRONG_TYPE"
    LOW_ROI_CONF = "LOW_ROI_CONF"
    LOW_CLASS_CONF = "LOW_CLASS_CONF"
    OUT_OF_POSITION = "OUT_OF_POSITION"
    OUT_OF_ANGLE = "OUT_OF_ANGLE"
    PART_NOT_READY = "PART_NOT_READY"
    ERROR = "ERROR"


class SessionStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class InspectionEventState(StrEnum):
    IDLE = "idle"
    PART_DETECTED = "part_detected"
    PART_READY = "part_ready"
    DECISION_PENDING = "decision_pending"
    DECISION_COMMITTED = "decision_committed"
    COOLDOWN = "cooldown"

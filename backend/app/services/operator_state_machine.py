from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class OperatorRuntimeState(StrEnum):
    IDLE = "IDLE"
    INSPECTION = "INSPECTION"
    RESULT = "RESULT"


@dataclass(frozen=True, slots=True)
class OperatorStateDecision:
    state: OperatorRuntimeState
    run_inspection: bool
    use_cached_result: bool
    reset_event: bool


class OperatorInspectionStateMachine:
    """Small per-session state policy for production inspection.

    The service owns only transition policy. Frame decoding, inference,
    validation, persistence, and overlay composition remain in
    InspectionSessionService so existing API contracts stay stable.
    """

    def update(
        self,
        session_state,
        *,
        part_ready: bool,
        present: bool,
        settled: bool,
    ) -> OperatorStateDecision:
        if not part_ready or not present:
            session_state.operator_state = OperatorRuntimeState.IDLE.value
            session_state.inspection_result_cache = None
            session_state.inspection_has_run_for_current_part = False
            session_state.part_removed_seen_at = datetime.now(UTC)
            return OperatorStateDecision(
                state=OperatorRuntimeState.IDLE,
                run_inspection=False,
                use_cached_result=False,
                reset_event=True,
            )

        if session_state.operator_state == OperatorRuntimeState.RESULT.value and session_state.inspection_result_cache:
            return OperatorStateDecision(
                state=OperatorRuntimeState.RESULT,
                run_inspection=False,
                use_cached_result=True,
                reset_event=False,
            )

        if settled:
            session_state.operator_state = OperatorRuntimeState.INSPECTION.value
            session_state.inspection_has_run_for_current_part = True
            return OperatorStateDecision(
                state=OperatorRuntimeState.INSPECTION,
                run_inspection=True,
                use_cached_result=False,
                reset_event=False,
            )

        session_state.operator_state = OperatorRuntimeState.IDLE.value
        return OperatorStateDecision(
            state=OperatorRuntimeState.IDLE,
            run_inspection=False,
            use_cached_result=False,
            reset_event=False,
        )

    def mark_result(self, session_state, payload: dict) -> None:
        session_state.operator_state = OperatorRuntimeState.RESULT.value
        session_state.inspection_result_cache = dict(payload)

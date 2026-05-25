from __future__ import annotations

import time
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

    # Minimum number of consecutive settled frames before transitioning
    # from IDLE -> INSPECTION.  Prevents premature inference triggered by
    # a transient part_ready blip (auto-exposure flicker, vibration).
    SETTLE_MIN_FRAMES = 0  # Changed: 0 = immediate inference on part_ready

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
            session_state.settle_frame_count = 0
            session_state.last_inference_ms = 0
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
            # Hysteresis: require N consecutive settled frames before
            # committing to INSPECTION state.
            session_state.settle_frame_count = int(getattr(session_state, "settle_frame_count", 0)) + 1
            if session_state.settle_frame_count >= self.SETTLE_MIN_FRAMES:
                # Cooldown: don't run inference more than once per second
                last_inference_ms = int(getattr(session_state, "last_inference_ms", 0))
                now_ms = int(time.time() * 1000)
                if last_inference_ms > 0 and (now_ms - last_inference_ms) < 1000:
                    # Still in cooldown — hold at IDLE, use cache if available
                    session_state.operator_state = OperatorRuntimeState.IDLE.value
                    if session_state.inspection_result_cache:
                        return OperatorStateDecision(
                            state=OperatorRuntimeState.RESULT,
                            run_inspection=False,
                            use_cached_result=True,
                            reset_event=False,
                        )
                    return OperatorStateDecision(
                        state=OperatorRuntimeState.IDLE,
                        run_inspection=False,
                        use_cached_result=False,
                        reset_event=False,
                    )
                session_state.last_inference_ms = now_ms
                session_state.operator_state = OperatorRuntimeState.INSPECTION.value
                session_state.inspection_has_run_for_current_part = True
                return OperatorStateDecision(
                    state=OperatorRuntimeState.INSPECTION,
                    run_inspection=True,
                    use_cached_result=False,
                    reset_event=False,
                )
            # Still in hysteresis window — hold at IDLE.
            session_state.operator_state = OperatorRuntimeState.IDLE.value
            return OperatorStateDecision(
                state=OperatorRuntimeState.IDLE,
                run_inspection=False,
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

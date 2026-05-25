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
    """Per-session state policy for production inspection.

    Inference interval mode:
    - Settle immediately (SETTLE_MIN_FRAMES=0)
    - After first inference → state=RESULT, cache=result
    - Every inference_interval_ms (default 200ms), run inference again
    - If interval not yet elapsed → use cached result (skip)
    - Part leave → reset everything
    """

    SETTLE_MIN_FRAMES = 0

    def update(
        self,
        session_state,
        *,
        part_ready: bool,
        present: bool,
        settled: bool,
    ) -> OperatorStateDecision:
        # ── Part leave or not present → reset to IDLE ──
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

        # ── Settled → check inference interval ──
        if settled:
            session_state.settle_frame_count = int(getattr(session_state, "settle_frame_count", 0)) + 1

            if session_state.settle_frame_count >= self.SETTLE_MIN_FRAMES:
                interval_ms = int(getattr(session_state, "inference_interval_ms", 200))
                last_inf_ms = int(getattr(session_state, "last_inference_ms", 0))
                now_ms = int(time.time() * 1000)

                # Check if inference interval has elapsed
                if last_inf_ms > 0 and (now_ms - last_inf_ms) < interval_ms and interval_ms > 0:
                    # Interval not yet elapsed → use cache if available
                    session_state.operator_state = OperatorRuntimeState.RESULT.value
                    if session_state.inspection_result_cache:
                        return OperatorStateDecision(
                            state=OperatorRuntimeState.RESULT,
                            run_inspection=False,
                            use_cached_result=True,
                            reset_event=False,
                        )
                    # No cache yet → run inspection anyway
                    session_state.last_inference_ms = now_ms
                    session_state.operator_state = OperatorRuntimeState.INSPECTION.value
                    session_state.inspection_has_run_for_current_part = True
                    return OperatorStateDecision(
                        state=OperatorRuntimeState.INSPECTION,
                        run_inspection=True,
                        use_cached_result=False,
                        reset_event=False,
                    )

                # Interval elapsed (or first time) → run inference
                session_state.last_inference_ms = now_ms
                session_state.operator_state = OperatorRuntimeState.INSPECTION.value
                session_state.inspection_has_run_for_current_part = True
                return OperatorStateDecision(
                    state=OperatorRuntimeState.INSPECTION,
                    run_inspection=True,
                    use_cached_result=False,
                    reset_event=False,
                )

        # ── Not settled → stay IDLE ──
        session_state.settle_frame_count = 0
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

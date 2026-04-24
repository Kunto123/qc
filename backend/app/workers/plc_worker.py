from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.services.plc_adapter import PlcAdapter

logger = logging.getLogger(__name__)

_QUEUE_TIMEOUT_S = 1.0


class PlcWorker:
    """Background worker that translates committed inspection events into PLC commands.

    Flow per committed inspection:
      1. send clamp_hold immediately
      2. wait hold_ms (interruptible)
      3. send clamp_release automatically

    Manual override: force_release() enqueues a release command that skips
    the hold wait — useful for admin intervention on stuck parts.

    Double-send protection: the commit hook in InspectionSessionService only
    fires once per event (count_committed is gated by current_event_committed),
    so PlcWorker itself does not need dedup logic.

    Safe release: if the clamp is still engaged when stop() is called (e.g. the
    process is interrupted mid-hold), a release is sent after the worker thread
    exits as a safety net before the adapter is disconnected.
    """

    def __init__(
        self,
        adapter: "PlcAdapter",
        *,
        hold_ms: int = 2000,
    ) -> None:
        self._adapter = adapter
        self._hold_ms = max(0, hold_ms)
        self._queue: queue.Queue[dict] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_command: str | None = None
        self._last_command_at: float | None = None
        self._clamp_engaged: bool = False

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="qc-plc-worker", daemon=True)
        self._thread.start()
        logger.info("[plc-worker] started (hold_ms=%d, adapter=%s)", self._hold_ms, type(self._adapter).__name__)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        # Safety net: if thread exited without completing a release, send it now
        with self._lock:
            engaged = self._clamp_engaged
        if engaged:
            logger.warning("[plc-worker] clamp still engaged after shutdown — sending safe release")
            try:
                self._adapter.send_clamp_release(event_id=None, reason="shutdown")
                with self._lock:
                    self._clamp_engaged = False
            except Exception as exc:  # noqa: BLE001
                logger.error("[plc-worker] safe release failed on shutdown: %s", exc)
        try:
            self._adapter.disconnect()
        except Exception:  # noqa: BLE001
            pass
        logger.info("[plc-worker] stopped")

    def enqueue_commit(self, *, event_id: str | None, decision: str) -> None:
        """Enqueue a committed inspection for PLC signalling. Non-blocking."""
        self._queue.put({"type": "commit", "event_id": event_id, "decision": decision})

    def force_release(self, *, reason: str = "manual") -> None:
        """Admin manual release — skips hold wait, sends release immediately."""
        self._queue.put({"type": "release", "event_id": None, "reason": reason})

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "queue_size": self._queue.qsize(),
                "last_command": self._last_command,
                "last_command_at": self._last_command_at,
                "clamp_engaged": self._clamp_engaged,
                **self._adapter.status(),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=_QUEUE_TIMEOUT_S)
            except queue.Empty:
                continue
            try:
                self._handle(item)
            except Exception as exc:  # noqa: BLE001
                logger.error("[plc-worker] error handling %s: %s", item.get("type"), exc, exc_info=True)

    def _handle(self, item: dict) -> None:
        cmd_type = item.get("type")
        if cmd_type == "commit":
            self._do_hold_then_release(event_id=item.get("event_id"), decision=str(item.get("decision") or ""))
        elif cmd_type == "release":
            self._do_release(event_id=item.get("event_id"), reason=str(item.get("reason") or "manual"))

    def _do_hold_then_release(self, *, event_id: str | None, decision: str) -> None:
        self._do_hold(event_id=event_id, decision=decision)
        if self._hold_ms > 0:
            deadline = time.perf_counter() + self._hold_ms / 1000.0
            while time.perf_counter() < deadline and not self._stop_event.is_set():
                time.sleep(min(0.05, max(0.0, deadline - time.perf_counter())))
        self._do_release(event_id=event_id, reason="auto")

    def _do_hold(self, *, event_id: str | None, decision: str) -> None:
        try:
            self._adapter.send_clamp_hold(event_id=event_id, decision=decision)
            with self._lock:
                self._last_command = "clamp_hold"
                self._last_command_at = time.time()
                self._clamp_engaged = True
        except Exception as exc:  # noqa: BLE001
            logger.error("[plc-worker] clamp_hold failed: %s", exc)

    def _do_release(self, *, event_id: str | None, reason: str) -> None:
        try:
            self._adapter.send_clamp_release(event_id=event_id, reason=reason)
            with self._lock:
                self._last_command = "clamp_release"
                self._last_command_at = time.time()
                self._clamp_engaged = False
        except Exception as exc:  # noqa: BLE001
            logger.error("[plc-worker] clamp_release failed: %s", exc)

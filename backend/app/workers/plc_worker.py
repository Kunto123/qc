"""
PLC Worker — simple, berdasarkan testall.py yang sudah terbukti bekerja.

Flow:
  IDLE → CLAMPING → ACCEPT (CH0=OFF, CH1=ON 1s → CH1=OFF → IDLE)
                   → REJECT (CH2=ON, CH0=ON → wait Input 1 → IDLE)
  Any state → Input 1 → IDLE (all off)
  Any state → Input 2 → cycle template

Relay Map:
  CH0 = Clamp (ON=clamp, OFF=release)
  CH1 = Buzzer + Lampu Accept/Reject (1 relay, 3 fungsi via NC/NO)
  CH2 = Buzzer Reject

Input Map:
  Input 1 (address 0) = Manual Release
  Input 2 (address 1) = Ganti Template
  Input 3 (address 2) = Clamp engaged feedback (optional)

Command Queue Pattern:
  Semua command (enqueue_part_ready, notify_decision, force_release)
  masuk ke queue. PLC worker thread yang consume dan execute.
  Backend thread TIDAK langsung write_coil() — hanya enqueue.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.services.plc_adapter import PlcAdapter

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.1  # 100ms seperti testall.py
_INPUT_READ_COUNT = 8   # sesuai firmware
_CMD_QUEUE_MAX = 64     # max queue depth, drop oldest if full


class PlcWorker:
    def __init__(
        self,
        adapter: "PlcAdapter",
        accept_pulse_ms: int = 1000,
        num_channels: int = 4,
        input_release_address: int = 0,
        input_template_address: int = 1,
        input_clamp_engaged_address: int = 2,
        clamp_feedback_enabled: bool = False,
    ) -> None:
        self._adapter = adapter
        self._accept_pulse_ms = int(accept_pulse_ms)
        self._num_channels = num_channels
        self._input_release_address = max(0, int(input_release_address))
        self._input_template_address = max(0, int(input_template_address))
        self._input_clamp_engaged_address = max(0, int(input_clamp_engaged_address))
        self._clamp_feedback_enabled = bool(clamp_feedback_enabled)
        # State
        self._state: str = "IDLE"
        self._accept_pulse_end: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Command queue — backend thread enqueue, worker thread consume
        self._cmd_queue: list[dict] = []
        self._cmd_event = threading.Event()  # signal worker thread
        # Input debounce
        self._last_input_press: dict[int, float] = {}
        self._input_debounce_s: float = 0.5
        self._template_cycle_event_id: int = 0
        self._last_template_cycle_at: float | None = None
        self._last_input_snapshot: list[bool] = []
        # Template cycling callback
        self._template_cycle_callback = None
        # State change callback (called when PLC state changes)
        self._on_state_change_callback = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        # Connect dan semua relay OFF
        try:
            self._adapter.connect()
            self._adapter.all_off(self._num_channels)
        except Exception as exc:
            logger.error("[plc-worker] start failed: %s", exc)
            return
        self._state = "IDLE"
        self._thread = threading.Thread(target=self._loop, name="qc-plc-worker", daemon=True)
        self._thread.start()
        logger.info("[plc-worker] started (state=IDLE)")

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        self._cmd_event.set()  # wake up worker thread
        # Safety: semua relay OFF
        try:
            self._adapter.all_off(self._num_channels)
        except Exception as exc:
            logger.error("[plc-worker] stop all_off failed: %s", exc)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        try:
            self._adapter.disconnect()
        except Exception:
            pass
        self._state = "IDLE"
        logger.info("[plc-worker] stopped")

    # ------------------------------------------------------------------
    # Public API — semua return immediately, enqueue ke worker thread
    # ------------------------------------------------------------------

    def enqueue_part_ready(self, *, event_id: str | None) -> None:
        """Part ready → enqueue CLAMPING command. Non-blocking."""
        self._enqueue_cmd({
            "type": "part_ready",
            "event_id": event_id,
        })

    def notify_decision(self, decision: str, *, event_id: str | None = None) -> None:
        """Inspection decision → enqueue ACCEPT/REJECT command. Non-blocking."""
        self._enqueue_cmd({
            "type": "decision",
            "decision": decision,
            "event_id": event_id,
        })

    def force_release(self, *, reason: str = "manual") -> None:
        """Manual release → enqueue force_release command. Non-blocking."""
        self._enqueue_cmd({
            "type": "force_release",
            "reason": reason,
        })

    def set_template_cycle_callback(self, callback) -> None:
        self._template_cycle_callback = callback

    def set_on_state_change_callback(self, callback) -> None:
        """Callback dipanggil saat PLC state berubah. old_state, new_state."""
        self._on_state_change_callback = callback

    def status(self) -> dict:
        with self._lock:
            clamp_engaged = self._clamp_engaged_from_snapshot_locked()
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "state": self._state,
                "connected": self._adapter.is_connected(),
                "clamp_feedback_enabled": self._clamp_feedback_enabled,
                "clamp_feedback_address": self._input_clamp_engaged_address,
                "clamp_engaged": clamp_engaged,
                "template_cycle_event_id": self._template_cycle_event_id,
                "last_template_cycle_at": self._last_template_cycle_at,
                "last_input_snapshot": list(self._last_input_snapshot),
                "cmd_queue_depth": len(self._cmd_queue),
                **self._adapter.status(),
            }

    def clamp_engaged(self) -> bool:
        with self._lock:
            return self._clamp_engaged_from_snapshot_locked()

    def _clamp_engaged_from_snapshot_locked(self) -> bool:
        if self._clamp_feedback_enabled:
            if self._input_clamp_engaged_address < len(self._last_input_snapshot):
                return bool(self._last_input_snapshot[self._input_clamp_engaged_address])
            return False
        return self._state in {"CLAMPING", "CLAMPED", "REJECT_BUZZER"}

    # ------------------------------------------------------------------
    # Command Queue
    # ------------------------------------------------------------------

    def _enqueue_cmd(self, cmd: dict) -> None:
        with self._lock:
            if len(self._cmd_queue) >= _CMD_QUEUE_MAX:
                # Drop oldest non-critical commands
                self._cmd_queue.pop(0)
                logger.warning("[plc-worker] cmd queue full, dropped oldest")
            self._cmd_queue.append(cmd)
        self._cmd_event.set()

    def _dequeue_cmd(self) -> dict | None:
        with self._lock:
            if self._cmd_queue:
                return self._cmd_queue.pop(0)
        return None

    # ------------------------------------------------------------------
    # Internal — executed in PLC worker thread only
    # ------------------------------------------------------------------

    def _set_state(self, new_state: str) -> None:
        with self._lock:
            old = self._state
            self._state = new_state
        logger.info("[plc-worker] %s → %s", old, new_state)
        # Notify callback if registered
        if self._on_state_change_callback and old != new_state:
            try:
                self._on_state_change_callback(old, new_state)
            except Exception as exc:
                logger.error("[plc-worker] state change callback error: %s", exc)

    def _write_coil(self, addr: int, value: bool) -> None:
        """Write coil with timeout protection. Raises on error."""
        try:
            self._adapter.write_coil(addr, value)
        except Exception as exc:
            logger.error("[plc-worker] write_coil addr=%d failed: %s", addr, exc)

    def _all_off(self, reason: str) -> None:
        try:
            self._adapter.all_off(self._num_channels)
        except Exception as exc:
            logger.error("[plc-worker] all_off failed: %s", exc)
        self._set_state("IDLE")
        logger.info("[plc-worker] ALL OFF — %s", reason)

    # -- ACCEPT: release clamp, CH1 pulse, back to IDLE --

    def _on_accept(self) -> None:
        """
        ACCEPT:
        1. CH0=OFF (release clamp)
        2. CH1=ON (Green+Buzzer, Red OFF via NC)
        3. Timer 1s → CH1=OFF (Red ON via NC) → IDLE
        """
        self._set_state("ACCEPT_PULSE")
        self._write_coil(0, False)  # CH0=OFF (release)
        self._write_coil(1, True)   # CH1=ON (Green+Buzzer)
        self._accept_pulse_end = time.time() + (self._accept_pulse_ms / 1000.0)
        logger.info("[plc-worker] ACCEPT — CH1 pulse %dms", self._accept_pulse_ms)

    def _finish_accept_pulse(self) -> None:
        self._write_coil(1, False)  # CH1=OFF (Red ON via NC)
        self._accept_pulse_end = None
        self._set_state("IDLE")
        logger.info("[plc-worker] ACCEPT done → IDLE")

    # -- REJECT: CH2=ON, CH0=ON, wait Input 1 --

    def _on_reject(self) -> None:
        """
        REJECT:
        1. CH2=ON (Buzzer Reject)
        2. CH0=ON (clamp stays)
        3. CH1=OFF (Red ON via NC)
        4. Wait Input 1
        """
        self._set_state("REJECT_BUZZER")
        self._write_coil(2, True)   # CH2=ON (Buzzer Reject)
        # CH0 stays ON, CH1=OFF (Red ON)
        logger.info("[plc-worker] REJECT — CH2 ON, waiting Input 1")

    # ------------------------------------------------------------------
    # Command handlers — executed in PLC worker thread
    # ------------------------------------------------------------------

    def _handle_cmd(self, cmd: dict) -> None:
        cmd_type = cmd.get("type")
        try:
            if cmd_type == "part_ready":
                self._cmd_part_ready(cmd)
            elif cmd_type == "decision":
                self._cmd_decision(cmd)
            elif cmd_type == "force_release":
                self._cmd_force_release(cmd)
            else:
                logger.warning("[plc-worker] unknown cmd type: %s", cmd_type)
        except Exception as exc:
            logger.error("[plc-worker] cmd %s error: %s", cmd_type, exc)

    def _cmd_part_ready(self, cmd: dict) -> None:
        """Handle part_ready command in worker thread."""
        with self._lock:
            if self._state != "IDLE":
                logger.info("[plc-worker] part_ready ignored (state=%s)", self._state)
                return
        self._write_coil(0, True)   # CH0=ON (clamp)
        self._write_coil(1, False)  # CH1=OFF (Red ON via NC)
        self._write_coil(2, False)  # CH2=OFF
        self._set_state("CLAMPING")
        logger.info("[plc-worker] CLAMPING — event=%s", cmd.get("event_id"))

    def _cmd_decision(self, cmd: dict) -> None:
        """Handle decision command in worker thread."""
        decision = cmd.get("decision")
        with self._lock:
            if self._state not in {"CLAMPING", "CLAMPED"}:
                return
        if decision == "ACCEPT":
            self._on_accept()
        elif decision == "REJECT":
            self._on_reject()

    def _cmd_force_release(self, cmd: dict) -> None:
        """Handle force_release command in worker thread."""
        self._all_off(cmd.get("reason", "manual"))

    # ------------------------------------------------------------------
    # Main loop (polling + state machine + command processing)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            # Process all pending commands first
            while True:
                cmd = self._dequeue_cmd()
                if cmd is None:
                    break
                self._handle_cmd(cmd)

            # Check accept pulse timeout
            if self._accept_pulse_end is not None and time.time() >= self._accept_pulse_end:
                self._finish_accept_pulse()

            # Poll inputs
            self._poll_inputs()

            # Wait for next cycle or command signal
            self._cmd_event.wait(timeout=_POLL_INTERVAL_S)
            self._cmd_event.clear()

    def _poll_inputs(self) -> None:
        try:
            inputs = self._adapter.read_inputs(address=0, count=_INPUT_READ_COUNT)
        except Exception as exc:
            logger.warning("[plc-worker] read_inputs error: %s", exc)
            return
        if not inputs or len(inputs) < 2:
            return

        now = time.time()
        with self._lock:
            self._last_input_snapshot = list(inputs[:_INPUT_READ_COUNT])

        # Input 1 (index 0): Manual Release
        if self._input_release_address < len(inputs) and inputs[self._input_release_address]:
            last = self._last_input_press.get(self._input_release_address, 0.0)
            if now - last > self._input_debounce_s:
                self._last_input_press[self._input_release_address] = now
                logger.info("[plc-worker] INPUT 1 — Manual Release")
                self._all_off("input_1")

        # Input 2 (index 1): Ganti Template
        if self._input_template_address < len(inputs) and inputs[self._input_template_address]:
            last = self._last_input_press.get(self._input_template_address, 0.0)
            if now - last > self._input_debounce_s:
                self._last_input_press[self._input_template_address] = now
                logger.info("[plc-worker] INPUT 2 — Ganti Template")
                with self._lock:
                    self._template_cycle_event_id += 1
                    self._last_template_cycle_at = now
                    self._last_input_snapshot = list(inputs[:8])
                if self._template_cycle_callback:
                    try:
                        self._template_cycle_callback()
                    except Exception as exc:
                        logger.error("[plc-worker] template cycle error: %s", exc)

        if (
            self._clamp_feedback_enabled
            and self._input_clamp_engaged_address < len(inputs)
            and inputs[self._input_clamp_engaged_address]
        ):
            with self._lock:
                state = self._state
            if state == "CLAMPING":
                self._set_state("CLAMPED")

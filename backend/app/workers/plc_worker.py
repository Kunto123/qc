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


class PlcWorker:
    def __init__(
        self,
        adapter: "PlcAdapter",
        accept_pulse_ms: int = 1000,
        num_channels: int = 4,
    ) -> None:
        self._adapter = adapter
        self._accept_pulse_ms = int(accept_pulse_ms)
        self._num_channels = num_channels
        # State
        self._state: str = "IDLE"
        self._accept_pulse_end: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Input debounce
        self._last_input_press: dict[int, float] = {}
        self._input_debounce_s: float = 0.5
        # Template cycling callback
        self._template_cycle_callback = None

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

    def enqueue_part_ready(self, *, event_id: str | None) -> None:
        """Part ready → CLAMPING (CH0=ON, CH1=OFF/Red ON)."""
        with self._lock:
            if self._state != "IDLE":
                logger.info("[plc-worker] part_ready ignored (state=%s)", self._state)
                return
        self._write_coil(0, True)   # CH0=ON (clamp)
        self._write_coil(1, False)  # CH1=OFF (Red ON via NC)
        self._write_coil(2, False)  # CH2=OFF
        self._set_state("CLAMPING")
        logger.info("[plc-worker] CLAMPING — event=%s", event_id)

    def notify_decision(self, decision: str, *, event_id: str | None = None) -> None:
        """Inspection decision → ACCEPT atau REJECT."""
        with self._lock:
            if self._state != "CLAMPING":
                return
        if decision == "ACCEPT":
            self._on_accept()
        elif decision == "REJECT":
            self._on_reject()

    def force_release(self, *, reason: str = "manual") -> None:
        """Manual release → semua OFF, IDLE."""
        self._all_off(reason)

    def set_template_cycle_callback(self, callback) -> None:
        self._template_cycle_callback = callback

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "connected": self._adapter.is_connected(),
                **self._adapter.status(),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_state(self, new_state: str) -> None:
        with self._lock:
            old = self._state
            self._state = new_state
        logger.info("[plc-worker] %s → %s", old, new_state)

    def _write_coil(self, addr: int, value: bool) -> None:
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
    # Main loop (polling + state machine)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            # Check accept pulse timeout
            if self._accept_pulse_end is not None and time.time() >= self._accept_pulse_end:
                self._finish_accept_pulse()
            # Poll inputs
            self._poll_inputs()
            # Sleep 100ms
            self._stop_event.wait(_POLL_INTERVAL_S)

    def _poll_inputs(self) -> None:
        try:
            inputs = self._adapter.read_inputs(address=0, count=_INPUT_READ_COUNT)
        except Exception as exc:
            logger.warning("[plc-worker] read_inputs error: %s", exc)
            return
        if not inputs or len(inputs) < 2:
            return

        now = time.time()

        # Input 1 (index 0): Manual Release
        if inputs[0]:
            last = self._last_input_press.get(0, 0.0)
            if now - last > self._input_debounce_s:
                self._last_input_press[0] = now
                logger.info("[plc-worker] INPUT 1 — Manual Release")
                self._all_off("input_1")

        # Input 2 (index 1): Ganti Template
        if inputs[1]:
            last = self._last_input_press.get(1, 0.0)
            if now - last > self._input_debounce_s:
                self._last_input_press[1] = now
                logger.info("[plc-worker] INPUT 2 — Ganti Template")
                if self._template_cycle_callback:
                    try:
                        self._template_cycle_callback()
                    except Exception as exc:
                        logger.error("[plc-worker] template cycle error: %s", exc)

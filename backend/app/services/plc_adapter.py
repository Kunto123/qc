from __future__ import annotations

import json
import logging
import socket
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PlcAdapter(ABC):
    """Protocol-agnostic interface for a PLC/remote-IO clamp controller.

    Concrete implementations plug in below this interface without touching
    the worker or session logic.  Replace TcpPlcAdapter._send with Modbus
    TCP, OPC UA, or digital-output calls once the PLC data sheet arrives.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def send_clamp_hold(self, *, event_id: str | None = None, decision: str | None = None) -> None: ...

    @abstractmethod
    def send_clamp_release(self, *, event_id: str | None = None, reason: str | None = None) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    def status(self) -> dict:
        return {"adapter": type(self).__name__}


class DryRunPlcAdapter(PlcAdapter):
    """No-op adapter: logs commands without touching any hardware.

    Used when QC_SUITE_PLC_DRY_RUN=1 (default) or PLC is disabled.
    """

    def connect(self) -> None:
        logger.info("[plc-dry-run] connect (no-op)")

    def send_clamp_hold(self, *, event_id: str | None = None, decision: str | None = None) -> None:
        logger.info("[plc-dry-run] CLAMP HOLD — event=%s decision=%s", event_id, decision)

    def send_clamp_release(self, *, event_id: str | None = None, reason: str | None = None) -> None:
        logger.info("[plc-dry-run] CLAMP RELEASE — event=%s reason=%s", event_id, reason)

    def disconnect(self) -> None:
        logger.info("[plc-dry-run] disconnect (no-op)")

    def status(self) -> dict:
        return {"adapter": "DryRunPlcAdapter", "connected": True}


class TcpPlcAdapter(PlcAdapter):
    """TCP socket adapter: sends line-framed JSON commands to a PLC gateway.

    Protocol placeholder — swap _send() internals for Modbus TCP, OPC UA,
    or raw digital-output once the target PLC data sheet is available.
    Signal map (current JSON field "signal"):
      "clamp_hold"    — energise clamp / close gripper
      "clamp_release" — de-energise clamp / open gripper
    """

    def __init__(self, host: str, port: int, *, timeout_s: float = 1.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout_s
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        logger.info("[plc-tcp] connected to %s:%d", self._host, self._port)

    def send_clamp_hold(self, *, event_id: str | None = None, decision: str | None = None) -> None:
        self._send({"signal": "clamp_hold", "event_id": event_id, "decision": decision})

    def send_clamp_release(self, *, event_id: str | None = None, reason: str | None = None) -> None:
        self._send({"signal": "clamp_release", "event_id": event_id, "reason": reason})

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
            logger.info("[plc-tcp] disconnected")

    def _send(self, payload: dict) -> None:
        if self._sock is None:
            self.connect()
        try:
            data = json.dumps(payload).encode() + b"\n"
            self._sock.sendall(data)  # type: ignore[union-attr]
        except OSError as exc:
            logger.error("[plc-tcp] send failed: %s — socket reset", exc)
            self._sock = None
            raise

    def status(self) -> dict:
        return {
            "adapter": "TcpPlcAdapter",
            "host": self._host,
            "port": self._port,
            "connected": self._sock is not None,
        }

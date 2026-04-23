from __future__ import annotations

import logging
import socket
import struct
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PlcAdapter(ABC):
    """Protocol-agnostic interface for a PLC/remote-IO clamp controller."""

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


class ModbusTcpPlcAdapter(PlcAdapter):
    """Modbus TCP adapter for a clamp-enabled remote I/O bridge.

    The adapter supports both write-single-coil and write-single-register
    command modes. Optional readback verification can confirm the remote I/O
    state after each command when a status point is available.
    """

    _FUNCTION_WRITE_SINGLE_COIL = 0x05
    _FUNCTION_WRITE_SINGLE_REGISTER = 0x06
    _FUNCTION_READ_COILS = 0x01
    _FUNCTION_READ_HOLDING_REGISTERS = 0x03
    _VALID_COMMAND_MODES = {"coil", "holding_register"}
    _VALID_READBACK_MODES = {"none", "coil", "holding_register"}
    _MODE_ALIASES = {
        "register": "holding_register",
        "holding-register": "holding_register",
        "coils": "coil",
    }

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = 1.0,
        unit_id: int = 1,
        command_mode: str = "coil",
        hold_address: int = 0,
        release_address: int = 0,
        hold_value: int = 1,
        release_value: int = 0,
        zero_based_addressing: bool = True,
        readback_enabled: bool = False,
        readback_mode: str = "coil",
        readback_address: int = 0,
        readback_expected_hold_value: int = 1,
        readback_expected_release_value: int = 0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self._host = host
        self._port = port
        self._timeout_s = float(timeout_s)
        self._unit_id = self._validate_unit_id(unit_id)
        self._command_mode = self._normalize_mode(command_mode, self._VALID_COMMAND_MODES, "command_mode")
        self._hold_address = self._validate_address(hold_address, "hold_address")
        self._release_address = self._validate_address(release_address, "release_address")
        self._hold_value = self._validate_register_value(hold_value, "hold_value")
        self._release_value = self._validate_register_value(release_value, "release_value")
        self._zero_based_addressing = bool(zero_based_addressing)
        self._readback_enabled = bool(readback_enabled)
        self._readback_mode = self._normalize_mode(readback_mode, self._VALID_READBACK_MODES, "readback_mode")
        self._readback_address = self._validate_address(readback_address, "readback_address")
        self._readback_expected_hold_value = self._validate_register_value(
            readback_expected_hold_value,
            "readback_expected_hold_value",
        )
        self._readback_expected_release_value = self._validate_register_value(
            readback_expected_release_value,
            "readback_expected_release_value",
        )
        if self._readback_enabled and self._readback_mode == "none":
            raise ValueError("readback_enabled requires readback_mode to be 'coil' or 'holding_register'")

        self._sock: socket.socket | None = None
        self._transaction_id = 0

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
        sock.settimeout(self._timeout_s)
        self._sock = sock
        logger.info(
            "[plc-modbus] connected to %s:%d (unit_id=%d, mode=%s)",
            self._host,
            self._port,
            self._unit_id,
            self._command_mode,
        )

    def send_clamp_hold(self, *, event_id: str | None = None, decision: str | None = None) -> None:
        self._write_command(
            label="clamp_hold",
            event_id=event_id,
            detail=decision,
            address=self._hold_address,
            value=self._hold_value,
            expected_readback=self._readback_expected_hold_value,
        )

    def send_clamp_release(self, *, event_id: str | None = None, reason: str | None = None) -> None:
        self._write_command(
            label="clamp_release",
            event_id=event_id,
            detail=reason,
            address=self._release_address,
            value=self._release_value,
            expected_readback=self._readback_expected_release_value,
        )

    def disconnect(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None
        logger.info("[plc-modbus] disconnected")

    def status(self) -> dict:
        return {
            "adapter": type(self).__name__,
            "host": self._host,
            "port": self._port,
            "unit_id": self._unit_id,
            "command_mode": self._command_mode,
            "hold_address": self._hold_address,
            "release_address": self._release_address,
            "readback_enabled": self._readback_enabled,
            "readback_mode": self._readback_mode,
            "readback_address": self._readback_address,
            "connected": self._sock is not None,
        }

    def _write_command(
        self,
        *,
        label: str,
        event_id: str | None,
        detail: str | None,
        address: int,
        value: int,
        expected_readback: int,
    ) -> None:
        wire_address = self._to_wire_address(address)
        function_code, payload, expected_value = self._build_write_payload(wire_address, value)
        response = self._exchange(payload)
        self._validate_write_response(
            response,
            label=label,
            function_code=function_code,
            wire_address=wire_address,
            expected_value=expected_value,
        )
        logger.info(
            "[plc-modbus] %s event=%s detail=%s address=%d value=%d mode=%s",
            label,
            event_id,
            detail,
            wire_address,
            expected_value,
            self._command_mode,
        )
        if self._readback_enabled:
            self._verify_readback(label=label, event_id=event_id, detail=detail, expected_value=expected_readback)

    def _exchange(self, payload: bytes) -> bytes:
        sock = self._ensure_socket()
        transaction_id = self._next_transaction_id()
        packet = struct.pack(
            ">HHHB",
            transaction_id,
            0,
            len(payload) + 1,
            self._unit_id,
        ) + payload

        try:
            sock.sendall(packet)
            header = self._read_exact(7)
            rx_transaction_id, rx_protocol_id, rx_length, rx_unit_id = struct.unpack(
                ">HHHB",
                header,
            )
            if rx_transaction_id != transaction_id:
                raise RuntimeError(
                    f"modbus transaction mismatch: expected {transaction_id}, received {rx_transaction_id}",
                )
            if rx_protocol_id != 0:
                raise RuntimeError(f"unexpected modbus protocol id {rx_protocol_id}")
            if rx_unit_id != self._unit_id:
                raise RuntimeError(f"unexpected modbus unit id {rx_unit_id}")
            if rx_length < 2:
                raise RuntimeError(f"invalid modbus response length {rx_length}")

            body = self._read_exact(rx_length - 1)
            if not body:
                raise RuntimeError("empty modbus response body")

            function_code = body[0]
            if function_code & 0x80:
                exception_code = body[1] if len(body) > 1 else 0
                raise RuntimeError(
                    f"modbus exception on function 0x{function_code & 0x7F:02X}: code 0x{exception_code:02X}",
                )
            return body
        except (OSError, RuntimeError):
            self._reset_socket()
            raise

    def _verify_readback(
        self,
        *,
        label: str,
        event_id: str | None,
        detail: str | None,
        expected_value: int,
    ) -> None:
        observed_value = self._readback_value()
        if observed_value != expected_value:
            raise RuntimeError(
                f"modbus readback mismatch after {label}: expected {expected_value}, observed {observed_value}",
            )
        logger.info(
            "[plc-modbus] %s verified event=%s detail=%s readback=%d",
            label,
            event_id,
            detail,
            observed_value,
        )

    def _readback_value(self) -> int:
        if self._readback_mode == "coil":
            response = self._exchange(
                struct.pack(
                    ">BHH",
                    self._FUNCTION_READ_COILS,
                    self._to_wire_address(self._readback_address),
                    1,
                ),
            )
            if len(response) < 3:
                raise RuntimeError("invalid coil readback response")
            byte_count = response[1]
            if byte_count < 1:
                raise RuntimeError("invalid coil readback byte count")
            return 1 if response[2] & 0x01 else 0

        response = self._exchange(
            struct.pack(
                ">BHH",
                self._FUNCTION_READ_HOLDING_REGISTERS,
                self._to_wire_address(self._readback_address),
                1,
            ),
        )
        if len(response) < 4:
            raise RuntimeError("invalid holding-register readback response")
        byte_count = response[1]
        if byte_count < 2:
            raise RuntimeError("invalid holding-register readback byte count")
        return struct.unpack(">H", response[2:4])[0]

    def _build_write_payload(self, wire_address: int, value: int) -> tuple[int, bytes, int]:
        if self._command_mode == "coil":
            encoded_value = 0xFF00 if int(value) != 0 else 0x0000
            return (
                self._FUNCTION_WRITE_SINGLE_COIL,
                struct.pack(">BHH", self._FUNCTION_WRITE_SINGLE_COIL, wire_address, encoded_value),
                encoded_value,
            )

        register_value = self._validate_register_value(value, "register_value")
        return (
            self._FUNCTION_WRITE_SINGLE_REGISTER,
            struct.pack(">BHH", self._FUNCTION_WRITE_SINGLE_REGISTER, wire_address, register_value),
            register_value,
        )

    def _validate_write_response(
        self,
        response: bytes,
        *,
        label: str,
        function_code: int,
        wire_address: int,
        expected_value: int,
    ) -> None:
        if len(response) != 5:
            raise RuntimeError(f"unexpected modbus write response length {len(response)} for {label}")
        rx_function_code, rx_address, rx_value = struct.unpack(">BHH", response)
        if rx_function_code != function_code:
            raise RuntimeError(
                f"unexpected modbus function 0x{rx_function_code:02X} for {label}; expected 0x{function_code:02X}",
            )
        if rx_address != wire_address:
            raise RuntimeError(
                f"unexpected modbus address {rx_address} for {label}; expected {wire_address}",
            )
        if rx_value != expected_value:
            raise RuntimeError(
                f"unexpected modbus value {rx_value} for {label}; expected {expected_value}",
            )

    def _ensure_socket(self) -> socket.socket:
        if self._sock is None:
            self.connect()
        if self._sock is None:
            raise RuntimeError("modbus socket is not connected")
        return self._sock

    def _read_exact(self, size: int) -> bytes:
        if size <= 0:
            return b""
        sock = self._ensure_socket()
        chunks = bytearray()
        while len(chunks) < size:
            try:
                chunk = sock.recv(size - len(chunks))
            except OSError as exc:
                raise RuntimeError("modbus receive failed") from exc
            if not chunk:
                raise RuntimeError("modbus connection closed while waiting for response")
            chunks.extend(chunk)
        return bytes(chunks)

    def _next_transaction_id(self) -> int:
        self._transaction_id = (self._transaction_id + 1) & 0xFFFF
        return self._transaction_id

    def _reset_socket(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None

    def _to_wire_address(self, address: int) -> int:
        wire_address = self._validate_address(address, "address")
        if not self._zero_based_addressing:
            if wire_address == 0:
                raise ValueError("address must be >= 1 when zero_based_addressing is disabled")
            wire_address -= 1
        if wire_address < 0 or wire_address > 0xFFFF:
            raise ValueError(f"address {address} is out of range after normalization")
        return wire_address

    @classmethod
    def _normalize_mode(cls, value: str, allowed: set[str], field_name: str) -> str:
        normalized = cls._MODE_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
        if normalized not in allowed:
            raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
        return normalized

    @staticmethod
    def _validate_unit_id(value: int) -> int:
        unit_id = int(value)
        if unit_id < 0 or unit_id > 255:
            raise ValueError("unit_id must be between 0 and 255")
        return unit_id

    @staticmethod
    def _validate_address(value: int, field_name: str) -> int:
        address = int(value)
        if address < 0 or address > 0xFFFF:
            raise ValueError(f"{field_name} must be between 0 and 65535")
        return address

    @staticmethod
    def _validate_register_value(value: int, field_name: str) -> int:
        register_value = int(value)
        if register_value < 0 or register_value > 0xFFFF:
            raise ValueError(f"{field_name} must be between 0 and 65535")
        return register_value


class TcpPlcAdapter(ModbusTcpPlcAdapter):
    """Compatibility alias for older imports."""


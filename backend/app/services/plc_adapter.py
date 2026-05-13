from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

try:
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient
    from pymodbus.exceptions import ConnectionException, ModbusIOException
    from pymodbus.framer import FramerType
except ModuleNotFoundError:  # pragma: no cover - exercised by environments without PLC deps
    ModbusSerialClient = None
    ModbusTcpClient = None

    class ConnectionException(Exception):
        pass

    class ModbusIOException(Exception):
        pass

    class FramerType:
        RTU = "rtu"

# Exception types that indicate a transport failure and warrant a reconnect+retry.
# RuntimeError (raised on Modbus protocol errors) is intentionally excluded so a
# bad-function-code or illegal-address response never causes a duplicate write.
_TRANSPORT_ERRORS = (OSError, ConnectionException, ModbusIOException)

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
        return {"adapter": "DryRunPlcAdapter", "transport": "dry_run", "connected": True}


class ModbusTcpPlcAdapter(PlcAdapter):
    """Modbus TCP adapter using pymodbus.client.ModbusTcpClient.

    Supports coil and holding-register command modes with optional readback
    verification. Transient disconnects trigger one automatic reconnect and
    retry before the error is surfaced to the caller.
    """

    _VALID_COMMAND_MODES = {"coil", "holding_register"}
    # "discrete_input" reads FC02 (read discrete inputs) — use this when the remote I/O
    # exposes a separate physical feedback input that reflects actual mechanical state,
    # rather than reading back the same output coil that was written.
    _VALID_READBACK_MODES = {"none", "coil", "discrete_input", "holding_register"}
    _MODE_ALIASES = {
        "register": "holding_register",
        "holding-register": "holding_register",
        "coils": "coil",
        "di": "discrete_input",
        "discrete-input": "discrete_input",
        "discrete_inputs": "discrete_input",
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
        readback_mode: str = "discrete_input",
        readback_address: int = 0,
        readback_expected_hold_value: int = 1,
        readback_expected_release_value: int = 0,
        client_factory: Callable[[], Any] | None = None,
        transport_name: str = "tcp",
        connection_label: str | None = None,
        status_fields: dict[str, Any] | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if client_factory is None and ModbusTcpClient is None:
            raise RuntimeError("pymodbus is required when PLC Modbus TCP is enabled")

        self._host = self._validate_host(host)
        self._port = self._validate_port(port)
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
            raise ValueError("readback_enabled requires readback_mode to be 'coil', 'discrete_input', or 'holding_register'")

        self._transport_name = str(transport_name).strip().lower() or "tcp"
        label = str(connection_label).strip() if connection_label is not None else ""
        self._connection_label = label or f"{self._host}:{self._port}"
        default_status_fields = {
            "host": self._host,
            "port": self._port,
            "connection_target": self._connection_label,
        }
        self._status_fields = dict(status_fields) if status_fields is not None else default_status_fields
        self._client = client_factory() if client_factory is not None else ModbusTcpClient(
            self._host,
            port=self._port,
            timeout=self._timeout_s,
        )

    def connect(self) -> None:
        if self._client.connected:
            return
        if not self._client.connect():
            raise RuntimeError(f"failed to connect to modbus at {self._connection_label}")
        logger.info(
            "[plc-modbus-%s] connected to %s (unit_id=%d, mode=%s)",
            self._transport_name,
            self._connection_label,
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
        self._client.close()
        logger.info("[plc-modbus-%s] disconnected from %s", self._transport_name, self._connection_label)

    def status(self) -> dict:
        status = {
            "adapter": type(self).__name__,
            "transport": self._transport_name,
            "unit_id": self._unit_id,
            "command_mode": self._command_mode,
            "hold_address": self._hold_address,
            "release_address": self._release_address,
            "readback_enabled": self._readback_enabled,
            "readback_mode": self._readback_mode,
            "readback_address": self._readback_address,
            "connected": self._client.connected,
        }
        status.update(self._status_fields)
        return status

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

        def _do_write() -> None:
            if self._command_mode == "coil":
                response = self._client.write_coil(wire_address, bool(value), device_id=self._unit_id)
            else:
                response = self._client.write_register(wire_address, value, device_id=self._unit_id)
            if response.isError():
                raise RuntimeError(f"modbus write error for {label}: {response}")

        self._call_with_retry(_do_write)
        logger.info(
            "[plc-modbus-%s] %s event=%s detail=%s address=%d value=%d mode=%s",
            self._transport_name,
            label,
            event_id,
            detail,
            wire_address,
            value,
            self._command_mode,
        )
        if self._readback_enabled:
            # Readback gets its own retry so a transient disconnect between the
            # successful write and the read triggers reconnect+re-read, not a
            # re-write, which would be redundant and misleading in logs.
            self._call_with_retry(
                lambda: self._verify_readback(
                    label=label, event_id=event_id, detail=detail, expected_value=expected_readback
                )
            )

    def _call_with_retry(self, operation: Callable[[], Any]) -> Any:
        """Run `operation`; on transport failure reconnect once and retry.

        Only _TRANSPORT_ERRORS (OSError, pymodbus ConnectionException,
        ModbusIOException) trigger a reconnect+retry. Protocol errors
        (RuntimeError from isError() checks) propagate immediately so a
        bad-address or illegal-function response never causes a duplicate write.
        """
        self._ensure_connected()
        try:
            return operation()
        except _TRANSPORT_ERRORS as exc:
            logger.warning(
                "[plc-modbus-%s] transient transport error on %s: %s — reconnecting",
                self._transport_name,
                self._connection_label,
                exc,
            )
            self._client.close()
            try:
                if not self._client.connect():
                    raise RuntimeError(
                        f"modbus reconnect failed to {self._connection_label}"
                    ) from exc
            except Exception as reconnect_exc:
                raise RuntimeError("modbus reconnect failed") from reconnect_exc
            return operation()

    def _ensure_connected(self) -> None:
        if not self._client.connected:
            if not self._client.connect():
                raise RuntimeError(f"failed to connect to modbus at {self._connection_label}")

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
            "[plc-modbus-%s] %s verified event=%s detail=%s readback=%d",
            self._transport_name,
            label,
            event_id,
            detail,
            observed_value,
        )

    def _readback_value(self) -> int:
        wire_address = self._to_wire_address(self._readback_address)
        if self._readback_mode == "coil":
            response = self._client.read_coils(wire_address, count=1, device_id=self._unit_id)
            if response.isError():
                raise RuntimeError(f"coil readback error: {response}")
            return int(response.bits[0])
        if self._readback_mode == "discrete_input":
            # FC02: read discrete inputs — intended for a separate physical feedback
            # point (e.g. a limit switch or sensor DI) that reflects mechanical state,
            # not the same output coil that was written.
            response = self._client.read_discrete_inputs(wire_address, count=1, device_id=self._unit_id)
            if response.isError():
                raise RuntimeError(f"discrete input readback error: {response}")
            return int(response.bits[0])
        response = self._client.read_holding_registers(wire_address, count=1, device_id=self._unit_id)
        if response.isError():
            raise RuntimeError(f"holding register readback error: {response}")
        return response.registers[0]

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
    def _validate_host(value: str) -> str:
        host = str(value).strip()
        if not host:
            raise ValueError("host must not be empty")
        return host

    @staticmethod
    def _validate_port(value: int) -> int:
        port = int(value)
        if port < 1 or port > 65535:
            raise ValueError("port must be between 1 and 65535")
        return port

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


class ModbusRtuPlcAdapter(ModbusTcpPlcAdapter):
    """Modbus RTU adapter using pymodbus.client.ModbusSerialClient."""

    _VALID_PARITIES = {"N", "E", "O", "M", "S"}
    _PARITY_ALIASES = {
        "none": "N",
        "n": "N",
        "even": "E",
        "e": "E",
        "odd": "O",
        "o": "O",
        "mark": "M",
        "m": "M",
        "space": "S",
        "s": "S",
    }

    def __init__(
        self,
        serial_port: str,
        *,
        baudrate: int = 9600,
        parity: str = "N",
        bytesize: int = 8,
        stopbits: int = 1,
        timeout_s: float = 1.0,
        unit_id: int = 1,
        command_mode: str = "coil",
        hold_address: int = 0,
        release_address: int = 0,
        hold_value: int = 1,
        release_value: int = 0,
        zero_based_addressing: bool = True,
        readback_enabled: bool = False,
        readback_mode: str = "discrete_input",
        readback_address: int = 0,
        readback_expected_hold_value: int = 1,
        readback_expected_release_value: int = 0,
    ) -> None:
        if ModbusSerialClient is None:
            raise RuntimeError("pymodbus is required when PLC Modbus RTU is enabled")
        serial_port = self._validate_serial_port(serial_port)
        baudrate = self._validate_baudrate(baudrate)
        parity = self._normalize_parity(parity)
        bytesize = self._validate_bytesize(bytesize)
        stopbits = self._validate_stopbits(stopbits)

        super().__init__(
            serial_port,
            1,
            timeout_s=timeout_s,
            unit_id=unit_id,
            command_mode=command_mode,
            hold_address=hold_address,
            release_address=release_address,
            hold_value=hold_value,
            release_value=release_value,
            zero_based_addressing=zero_based_addressing,
            readback_enabled=readback_enabled,
            readback_mode=readback_mode,
            readback_address=readback_address,
            readback_expected_hold_value=readback_expected_hold_value,
            readback_expected_release_value=readback_expected_release_value,
            client_factory=lambda: ModbusSerialClient(
                port=serial_port,
                framer=FramerType.RTU,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout_s,
            ),
            transport_name="rtu",
            connection_label=serial_port,
            status_fields={
                "serial_port": serial_port,
                "baudrate": baudrate,
                "parity": parity,
                "bytesize": bytesize,
                "stopbits": stopbits,
                "connection_target": serial_port,
            },
        )

    def _transport_label(self) -> str:
        return "rtu"

    @staticmethod
    def _validate_serial_port(value: str) -> str:
        serial_port = str(value).strip()
        if not serial_port:
            raise ValueError("serial_port must not be empty")
        return serial_port

    @staticmethod
    def _validate_baudrate(value: int) -> int:
        baudrate = int(value)
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")
        return baudrate

    @classmethod
    def _normalize_parity(cls, value: str) -> str:
        normalized = cls._PARITY_ALIASES.get(str(value).strip().lower(), str(value).strip().upper())
        if normalized not in cls._VALID_PARITIES:
            raise ValueError(f"parity must be one of {sorted(cls._VALID_PARITIES)}")
        return normalized

    @staticmethod
    def _validate_bytesize(value: int) -> int:
        bytesize = int(value)
        if bytesize not in {5, 6, 7, 8}:
            raise ValueError("bytesize must be one of 5, 6, 7, or 8")
        return bytesize

    @staticmethod
    def _validate_stopbits(value: int) -> int:
        stopbits = int(value)
        if stopbits not in {1, 2}:
            raise ValueError("stopbits must be either 1 or 2")
        return stopbits


class TcpPlcAdapter(ModbusTcpPlcAdapter):
    """Compatibility alias for older imports."""


def build_plc_adapter(config: Any) -> PlcAdapter:
    """Build the active PLC adapter from runtime configuration."""

    transport = str(getattr(config, "plc_transport", "tcp")).strip().lower() or "tcp"
    if transport not in {"tcp", "rtu"}:
        raise ValueError("QC_SUITE_PLC_TRANSPORT must be 'tcp' or 'rtu'")

    if not getattr(config, "plc_enabled", False) or getattr(config, "plc_dry_run", True):
        return DryRunPlcAdapter()
    common_kwargs = {
        "timeout_s": getattr(config, "plc_timeout_ms") / 1000.0,
        "unit_id": getattr(config, "plc_modbus_unit_id"),
        "command_mode": getattr(config, "plc_modbus_command_mode"),
        "hold_address": getattr(config, "plc_modbus_hold_address"),
        "release_address": getattr(config, "plc_modbus_release_address"),
        "hold_value": getattr(config, "plc_modbus_hold_value"),
        "release_value": getattr(config, "plc_modbus_release_value"),
        "zero_based_addressing": getattr(config, "plc_modbus_zero_based_addressing"),
        "readback_enabled": getattr(config, "plc_modbus_readback_enabled"),
        "readback_mode": getattr(config, "plc_modbus_readback_mode"),
        "readback_address": getattr(config, "plc_modbus_readback_address"),
        "readback_expected_hold_value": getattr(config, "plc_modbus_readback_expected_hold_value"),
        "readback_expected_release_value": getattr(config, "plc_modbus_readback_expected_release_value"),
    }

    if transport == "tcp":
        return ModbusTcpPlcAdapter(
            getattr(config, "plc_host"),
            getattr(config, "plc_port"),
            **common_kwargs,
        )
    if transport == "rtu":
        return ModbusRtuPlcAdapter(
            getattr(config, "plc_serial_port"),
            baudrate=getattr(config, "plc_serial_baudrate"),
            parity=getattr(config, "plc_serial_parity"),
            bytesize=getattr(config, "plc_serial_bytesize"),
            stopbits=getattr(config, "plc_serial_stopbits"),
            **common_kwargs,
        )

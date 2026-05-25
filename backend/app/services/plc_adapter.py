"""
PLC Modbus adapter — simple, berdasarkan testall.py yang sudah terbukti bekerja.

Referensi: D:/pythonmodbus/testall.py
- Slave ID: 255
- Timeout: 0.5s
- Write delay: 50ms setelah setiap write
- Input read: count=8 (sesuai firmware)
- Simple connect di awal, tidak ada retry loop kompleks
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

try:
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient
except ModuleNotFoundError:
    ModbusSerialClient = None
    ModbusTcpClient = None

logger = logging.getLogger(__name__)


class PlcAdapter(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def is_connected(self) -> bool: ...
    @abstractmethod
    def write_coil(self, address: int, value: bool) -> None: ...
    @abstractmethod
    def write_coils(self, coils: dict[int, bool]) -> None: ...
    @abstractmethod
    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]: ...

    def all_off(self, num_channels: int = 4) -> None:
        for i in range(num_channels):
            self.write_coil(i, False)
            time.sleep(0.05)
    def status(self) -> dict:
        return {"adapter": type(self).__name__, "connected": self.is_connected()}


class DryRunPlcAdapter(PlcAdapter):
    def connect(self) -> None:
        logger.info("[plc-dry-run] connect")

    def disconnect(self) -> None:
        logger.info("[plc-dry-run] disconnect")

    def is_connected(self) -> bool:
        return True

    def write_coil(self, address: int, value: bool) -> None:
        logger.info("[plc-dry-run] write_coil addr=%d value=%s", address, value)

    def write_coils(self, coils: dict[int, bool]) -> None:
        for addr, val in coils.items():
            self.write_coil(addr, val)

    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]:
        return [False] * count


class ModbusRtuPlcAdapter(PlcAdapter):
    """Modbus RTU simple — berdasarkan testall.py"""

    def __init__(
        self,
        port: str = "COM7",
        baudrate: int = 9600,
        slave_id: int = 255,
        timeout: float = 0.5,
    ):
        if ModbusSerialClient is None:
            raise RuntimeError("pymodbus is not installed")
        self._slave_id = slave_id
        self._client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            parity="N",
            stopbits=1,
            bytesize=8,
            timeout=timeout,
        )
        self._port = port

    def connect(self) -> None:
        if self._client.connected:
            return
        if not self._client.connect():
            raise RuntimeError(f"failed to connect to modbus RTU on {self._port}")
        logger.info("[plc-modbus-rtu] connected to %s (slave=%d)", self._port, self._slave_id)

    def disconnect(self) -> None:
        self._client.close()
        logger.info("[plc-modbus-rtu] disconnected from %s", self._port)

    def is_connected(self) -> bool:
        return self._client.connected

    def _ensure_connected(self) -> None:
        if not self._client.connected:
            if not self._client.connect():
                raise RuntimeError(f"modbus reconnect failed on {self._port}")

    def write_coil(self, address: int, value: bool) -> None:
        self._ensure_connected()
        resp = self._client.write_coil(address, bool(value), device_id=self._slave_id)
        if resp.isError():
            raise RuntimeError(f"write_coil error addr={address}: {resp}")
        time.sleep(0.05)  # delay 50ms seperti testall.py

    def write_coils(self, coils: dict[int, bool]) -> None:
        for addr, val in coils.items():
            self.write_coil(addr, val)

    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]:
        self._ensure_connected()
        resp = self._client.read_discrete_inputs(address, count=count, device_id=self._slave_id)
        if resp.isError():
            logger.warning("[plc-modbus-rtu] read_inputs error: %s", resp)
            return [False] * count
        return [bool(resp.bits[i]) for i in range(count)]


class ModbusTcpPlcAdapter(PlcAdapter):
    """Modbus TCP simple — referensi testall.py"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 502,
        slave_id: int = 255,
        timeout: float = 0.5,
    ):
        if ModbusTcpClient is None:
            raise RuntimeError("pymodbus is not installed")
        self._slave_id = slave_id
        self._client = ModbusTcpClient(host=host, port=port, timeout=timeout)

    def connect(self) -> None:
        if self._client.connected:
            return
        if not self._client.connect():
            raise RuntimeError(f"failed to connect to modbus TCP")
        logger.info("[plc-modbus-tcp] connected (slave=%d)", self._slave_id)

    def disconnect(self) -> None:
        self._client.close()
        logger.info("[plc-modbus-tcp] disconnected")

    def is_connected(self) -> bool:
        return self._client.connected

    def _ensure_connected(self) -> None:
        if not self._client.connected:
            if not self._client.connect():
                raise RuntimeError("modbus TCP reconnect failed")

    def write_coil(self, address: int, value: bool) -> None:
        self._ensure_connected()
        resp = self._client.write_coil(address, bool(value), device_id=self._slave_id)
        if resp.isError():
            raise RuntimeError(f"write_coil error addr={address}: {resp}")
        time.sleep(0.05)

    def write_coils(self, coils: dict[int, bool]) -> None:
        for addr, val in coils.items():
            self.write_coil(addr, val)

    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]:
        self._ensure_connected()
        resp = self._client.read_discrete_inputs(address, count=count, device_id=self._slave_id)
        if resp.isError():
            logger.warning("[plc-modbus-tcp] read_inputs error: %s", resp)
            return [False] * count
        return [bool(resp.bits[i]) for i in range(count)]


def build_plc_adapter(config) -> PlcAdapter:
    """Factory: pilih adapter berdasarkan config."""
    if config.plc_dry_run:
        return DryRunPlcAdapter()

    if config.plc_transport == "rtu":
        return ModbusRtuPlcAdapter(
            port=config.plc_serial_port or "COM7",
            baudrate=config.plc_serial_baudrate,
            slave_id=config.plc_modbus_unit_id,
            timeout=0.5,
        )
    elif config.plc_transport == "tcp":
        return ModbusTcpPlcAdapter(
            host=config.plc_host or "127.0.0.1",
            port=config.plc_port or 502,
            slave_id=config.plc_modbus_unit_id,
            timeout=0.5,
        )
    else:
        return DryRunPlcAdapter()

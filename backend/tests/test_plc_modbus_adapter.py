from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymodbus.framer import FramerType

from backend.app.services.plc_adapter import DryRunPlcAdapter, ModbusRtuPlcAdapter, ModbusTcpPlcAdapter, build_plc_adapter
from backend.app.workers.plc_worker import PlcWorker


def _make_mock_client(*, coil_value: bool = True, register_value: int = 0) -> mock.MagicMock:
    """Return a MagicMock that behaves like a pymodbus ModbusTcpClient instance."""
    client = mock.MagicMock()
    client.connected = False

    def _connect() -> bool:
        client.connected = True
        return True

    client.connect.side_effect = _connect

    ok_resp = mock.MagicMock()
    ok_resp.isError.return_value = False
    client.write_coil.return_value = ok_resp
    client.write_register.return_value = ok_resp

    coil_resp = mock.MagicMock()
    coil_resp.isError.return_value = False
    coil_resp.bits = [coil_value]
    client.read_coils.return_value = coil_resp

    reg_resp = mock.MagicMock()
    reg_resp.isError.return_value = False
    reg_resp.registers = [register_value]
    client.read_holding_registers.return_value = reg_resp

    return client


def _make_adapter(mock_client: mock.MagicMock, host: str = "10.0.0.5", port: int = 502, **kwargs) -> ModbusTcpPlcAdapter:
    """Construct a ModbusTcpPlcAdapter with the given mock client injected."""
    with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mock_client):
        return ModbusTcpPlcAdapter(host, port, **kwargs)


def _make_rtu_adapter(
    mock_client: mock.MagicMock,
    serial_port: str = "COM3",
    **kwargs,
) -> ModbusRtuPlcAdapter:
    """Construct a ModbusRtuPlcAdapter with the given mock client injected."""
    with mock.patch("backend.app.services.plc_adapter.ModbusSerialClient", return_value=mock_client):
        return ModbusRtuPlcAdapter(serial_port, **kwargs)


def _make_config(**overrides) -> SimpleNamespace:
    defaults = dict(
        plc_enabled=True,
        plc_dry_run=False,
        plc_transport="tcp",
        plc_timeout_ms=750,
        plc_modbus_unit_id=7,
        plc_modbus_command_mode="coil",
        plc_modbus_hold_address=12,
        plc_modbus_release_address=12,
        plc_modbus_hold_value=1,
        plc_modbus_release_value=0,
        plc_modbus_zero_based_addressing=True,
        plc_modbus_readback_enabled=False,
        plc_modbus_readback_mode="discrete_input",
        plc_modbus_readback_address=4,
        plc_modbus_readback_expected_hold_value=1,
        plc_modbus_readback_expected_release_value=0,
        plc_host="10.0.0.5",
        plc_port=502,
        plc_serial_port="COM3",
        plc_serial_baudrate=9600,
        plc_serial_parity="N",
        plc_serial_bytesize=8,
        plc_serial_stopbits=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class ModbusTcpPlcAdapterTest(unittest.TestCase):
    def test_status_reports_adapter_name(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mc):
            adapter = ModbusTcpPlcAdapter("127.0.0.1", 502, unit_id=1)
            self.assertEqual(adapter.status()["adapter"], "ModbusTcpPlcAdapter")
            self.assertEqual(adapter.status()["transport"], "tcp")

            from backend.app.services.plc_adapter import TcpPlcAdapter
            compat_adapter = TcpPlcAdapter("127.0.0.1", 502, unit_id=1)
            self.assertEqual(compat_adapter.status()["adapter"], "TcpPlcAdapter")

    def test_build_plc_adapter_selects_dry_run_when_disabled(self) -> None:
        adapter = build_plc_adapter(_make_config(plc_enabled=False))
        self.assertIsInstance(adapter, DryRunPlcAdapter)

    def test_build_plc_adapter_selects_tcp(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mc) as MockClass:
            adapter = build_plc_adapter(_make_config(plc_transport="tcp"))

        self.assertIsInstance(adapter, ModbusTcpPlcAdapter)
        MockClass.assert_called_once_with("10.0.0.5", port=502, timeout=0.75)

    def test_build_plc_adapter_selects_rtu(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusSerialClient", return_value=mc) as MockClass:
            adapter = build_plc_adapter(_make_config(plc_transport="rtu", plc_modbus_unit_id=255))

        self.assertIsInstance(adapter, ModbusRtuPlcAdapter)
        self.assertEqual(adapter.status()["transport"], "rtu")
        MockClass.assert_called_once_with(
            port="COM3",
            framer=FramerType.RTU,
            baudrate=9600,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.75,
        )

    def test_build_plc_adapter_rejects_invalid_transport(self) -> None:
        with self.assertRaises(ValueError):
            build_plc_adapter(_make_config(plc_transport="invalid"))

    def test_coil_mode_writes_hold_and_release(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mc) as MockClass:
            adapter = ModbusTcpPlcAdapter(
                "10.0.0.5", 502, timeout_s=0.75, unit_id=7,
                command_mode="coil", hold_address=12, release_address=12,
            )

        self.assertFalse(adapter.status()["connected"])
        adapter.send_clamp_hold(event_id="evt-1", decision="ACCEPT")
        self.assertTrue(adapter.status()["connected"])
        adapter.send_clamp_release(event_id="evt-1", reason="auto")
        adapter.disconnect()

        MockClass.assert_called_once_with("10.0.0.5", port=502, timeout=0.75)
        mc.connect.assert_called_once()
        mc.write_coil.assert_any_call(12, True, device_id=7)
        mc.write_coil.assert_any_call(12, False, device_id=7)
        self.assertEqual(mc.write_coil.call_count, 2)
        mc.close.assert_called_once()

    def test_rtu_mode_writes_hold_and_release(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusSerialClient", return_value=mc) as MockClass:
            adapter = ModbusRtuPlcAdapter(
                "COM3",
                timeout_s=0.75,
                unit_id=255,
                command_mode="coil",
                hold_address=0,
                release_address=1,
            )

        adapter.send_clamp_hold(event_id="evt-rtu", decision="ACCEPT")
        adapter.send_clamp_release(event_id="evt-rtu", reason="auto")
        adapter.disconnect()

        MockClass.assert_called_once_with(
            port="COM3",
            framer=FramerType.RTU,
            baudrate=9600,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.75,
        )
        mc.write_coil.assert_any_call(0, True, device_id=255)
        mc.write_coil.assert_any_call(1, False, device_id=255)
        self.assertEqual(mc.write_coil.call_count, 2)
        mc.close.assert_called_once()

    def test_register_mode_writes_distinct_addresses(self) -> None:
        mc = _make_mock_client()
        adapter = _make_adapter(
            mc, "192.168.1.50", 5020, unit_id=3,
            command_mode="holding_register",
            hold_address=20, release_address=21,
            hold_value=2, release_value=0,
        )

        adapter.send_clamp_hold(event_id="evt-2", decision="REJECT")
        adapter.send_clamp_release(event_id="evt-2", reason="auto")

        mc.write_register.assert_any_call(20, 2, device_id=3)
        mc.write_register.assert_any_call(21, 0, device_id=3)
        self.assertEqual(mc.write_register.call_count, 2)

    def test_coil_readback_verifies_state(self) -> None:
        mc = _make_mock_client()
        hold_resp = mock.MagicMock()
        hold_resp.isError.return_value = False
        hold_resp.bits = [True]
        release_resp = mock.MagicMock()
        release_resp.isError.return_value = False
        release_resp.bits = [False]
        mc.read_coils.side_effect = [hold_resp, release_resp]

        adapter = _make_adapter(
            mc, "10.1.1.5", 502, unit_id=9,
            command_mode="coil", hold_address=8, release_address=8,
            readback_enabled=True, readback_mode="coil", readback_address=8,
        )

        adapter.send_clamp_hold(event_id="evt-3", decision="ACCEPT")
        adapter.send_clamp_release(event_id="evt-3", reason="auto")

        self.assertEqual(mc.read_coils.call_count, 2)
        mc.read_coils.assert_called_with(8, count=1, device_id=9)

    def test_readback_defaults_to_discrete_input(self) -> None:
        mc = _make_mock_client()
        di_resp = mock.MagicMock()
        di_resp.isError.return_value = False
        di_resp.bits = [True]
        mc.read_discrete_inputs.return_value = di_resp

        adapter = _make_adapter(
            mc,
            unit_id=4,
            command_mode="coil",
            hold_address=0,
            readback_enabled=True,
            readback_address=4,
        )

        self.assertEqual(adapter._readback_mode, "discrete_input")  # noqa: SLF001
        adapter.send_clamp_hold(event_id="evt-default-di", decision="ACCEPT")

        mc.read_discrete_inputs.assert_called_with(4, count=1, device_id=4)
        mc.read_coils.assert_not_called()

    def test_reconnect_on_transient_disconnect(self) -> None:
        mc = _make_mock_client()
        ok_resp = mock.MagicMock()
        ok_resp.isError.return_value = False
        # First write raises a transient error; second succeeds after reconnect
        mc.write_coil.side_effect = [ConnectionError("connection lost"), ok_resp]

        adapter = _make_adapter(mc, hold_address=0, unit_id=1)

        adapter.send_clamp_hold(event_id="evt-r", decision="ACCEPT")

        # connect called twice: once lazily on first send, once during reconnect
        self.assertEqual(mc.connect.call_count, 2)
        self.assertEqual(mc.write_coil.call_count, 2)

    def test_unreachable_host_raises_on_connect(self) -> None:
        mc = mock.MagicMock()
        mc.connected = False
        mc.connect.return_value = False  # host unreachable — connect always fails

        adapter = _make_adapter(mc)

        with self.assertRaises(RuntimeError):
            adapter.send_clamp_hold(event_id="evt-x", decision="ACCEPT")

    def test_readback_mismatch_raises(self) -> None:
        mc = _make_mock_client()
        # Readback returns 0 (coil off) but expected hold value is 1 (coil on)
        mismatch_resp = mock.MagicMock()
        mismatch_resp.isError.return_value = False
        mismatch_resp.bits = [False]
        mc.read_coils.return_value = mismatch_resp

        adapter = _make_adapter(
            mc, unit_id=1, command_mode="coil", hold_address=0,
            readback_enabled=True, readback_mode="coil", readback_address=0,
            readback_expected_hold_value=1,
        )

        with self.assertRaises(RuntimeError, msg="readback mismatch must raise"):
            adapter.send_clamp_hold(event_id="evt-m", decision="ACCEPT")

    def test_holding_register_readback(self) -> None:
        mc = _make_mock_client(register_value=5)
        hold_reg_resp = mock.MagicMock()
        hold_reg_resp.isError.return_value = False
        hold_reg_resp.registers = [5]
        mc.read_holding_registers.return_value = hold_reg_resp

        adapter = _make_adapter(
            mc, unit_id=2, command_mode="holding_register",
            hold_address=10, release_address=11,
            hold_value=5, release_value=0,
            readback_enabled=True, readback_mode="holding_register",
            readback_address=10, readback_expected_hold_value=5,
        )

        adapter.send_clamp_hold(event_id="evt-reg", decision="ACCEPT")

        mc.read_holding_registers.assert_called_with(10, count=1, device_id=2)

    def test_discrete_input_readback_uses_fc02(self) -> None:
        """readback_mode=discrete_input must call read_discrete_inputs (FC02), not read_coils."""
        mc = _make_mock_client()
        di_resp = mock.MagicMock()
        di_resp.isError.return_value = False
        di_resp.bits = [True]  # clamp engaged feedback
        mc.read_discrete_inputs.return_value = di_resp

        adapter = _make_adapter(
            mc, unit_id=1, command_mode="coil", hold_address=0,
            readback_enabled=True, readback_mode="discrete_input",
            readback_address=4,  # separate DI feedback point
            readback_expected_hold_value=1,
        )

        adapter.send_clamp_hold(event_id="evt-di", decision="ACCEPT")

        mc.read_discrete_inputs.assert_called_with(4, count=1, device_id=1)
        mc.read_coils.assert_not_called()

    def test_discrete_input_readback_mode_aliases(self) -> None:
        """'di' and 'discrete-input' aliases must resolve to discrete_input mode."""
        for alias in ("di", "discrete-input", "discrete_inputs"):
            mc = _make_mock_client()
            adapter = _make_adapter(
                mc, unit_id=1, readback_enabled=True,
                readback_mode=alias, readback_address=0,
            )
            self.assertEqual(adapter._readback_mode, "discrete_input")  # noqa: SLF001

    def test_zero_based_addressing_false_subtracts_one(self) -> None:
        mc = _make_mock_client()
        adapter = _make_adapter(mc, unit_id=1, hold_address=1, zero_based_addressing=False)

        adapter.send_clamp_hold(event_id="evt-addr", decision="ACCEPT")

        # address 1 with zero_based_addressing=False → wire address 0
        mc.write_coil.assert_called_with(0, True, device_id=1)

    def test_zero_based_addressing_false_rejects_address_zero(self) -> None:
        # ValueError is raised lazily when the address is used, not at construction
        adapter = _make_adapter(_make_mock_client(), unit_id=1, hold_address=0, zero_based_addressing=False)
        with self.assertRaises(ValueError):
            adapter.send_clamp_hold(event_id="evt-z", decision="ACCEPT")

    def test_protocol_error_does_not_retry_write(self) -> None:
        """A Modbus protocol error (isError=True) must not trigger a retry write."""
        mc = _make_mock_client()
        error_resp = mock.MagicMock()
        error_resp.isError.return_value = True
        mc.write_coil.return_value = error_resp

        adapter = _make_adapter(mc, unit_id=1, hold_address=0)

        with self.assertRaises(RuntimeError):
            adapter.send_clamp_hold(event_id="evt-proto", decision="ACCEPT")

        # Write must only have been called once — no duplicate write on protocol error
        self.assertEqual(mc.write_coil.call_count, 1)

    def test_invalid_command_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_adapter(_make_mock_client(), command_mode="invalid_mode")

    def test_negative_timeout_raises(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mc):
            with self.assertRaises(ValueError):
                ModbusTcpPlcAdapter("127.0.0.1", 502, timeout_s=-1.0)

    def test_status_reflects_connection_state(self) -> None:
        mc = _make_mock_client()
        adapter = _make_adapter(mc, unit_id=3, command_mode="coil", hold_address=5, release_address=6)

        self.assertFalse(adapter.status()["connected"])
        adapter.send_clamp_hold(event_id="e", decision="ACCEPT")
        self.assertTrue(adapter.status()["connected"])
        self.assertEqual(adapter.status()["hold_address"], 5)
        self.assertEqual(adapter.status()["release_address"], 6)
        self.assertEqual(adapter.status()["unit_id"], 3)

    def test_plc_worker_input2_increments_template_cycle_event_once_per_debounce(self) -> None:
        class FakeAdapter:
            def __init__(self) -> None:
                self.all_off_count = 0
                self.inputs = [False, True, False, False, False, False, False, False]

            def read_inputs(self, address: int = 0, count: int = 8):
                return list(self.inputs)

            def write_coil(self, address: int, value: bool) -> None:
                return None

            def all_off(self, num_channels: int) -> None:
                self.all_off_count += 1

            def is_connected(self) -> bool:
                return True

            def status(self) -> dict:
                return {}

        adapter = FakeAdapter()
        worker = PlcWorker(adapter)

        worker._poll_inputs()  # noqa: SLF001
        self.assertEqual(worker.status()["template_cycle_event_id"], 1)

        worker._poll_inputs()  # noqa: SLF001
        self.assertEqual(worker.status()["template_cycle_event_id"], 1)
        self.assertEqual(adapter.all_off_count, 0)

    def test_plc_worker_input1_manual_release_still_all_off(self) -> None:
        class FakeAdapter:
            def __init__(self) -> None:
                self.all_off_count = 0

            def read_inputs(self, address: int = 0, count: int = 8):
                return [True, False, False, False, False, False, False, False]

            def write_coil(self, address: int, value: bool) -> None:
                return None

            def all_off(self, num_channels: int) -> None:
                self.all_off_count += 1

            def is_connected(self) -> bool:
                return True

            def status(self) -> dict:
                return {}

        adapter = FakeAdapter()
        worker = PlcWorker(adapter)
        worker._poll_inputs()  # noqa: SLF001

        self.assertEqual(adapter.all_off_count, 1)
        self.assertEqual(worker.status()["state"], "IDLE")


if __name__ == "__main__":
    unittest.main()

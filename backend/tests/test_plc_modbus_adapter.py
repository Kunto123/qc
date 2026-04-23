from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.plc_adapter import ModbusTcpPlcAdapter


def _write_response(transaction_id: int, unit_id: int, function_code: int, address: int, value: int) -> bytes:
    payload = struct.pack(">BHH", function_code, address, value)
    header = struct.pack(">HHHB", transaction_id, 0, len(payload) + 1, unit_id)
    return header + payload


class _FakeSocket:
    def __init__(self, recv_bytes: bytes) -> None:
        self._recv_buffer = bytearray(recv_bytes)
        self.sent_packets: list[bytes] = []
        self.timeout: float | None = None
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def sendall(self, payload: bytes) -> None:
        self.sent_packets.append(payload)

    def recv(self, size: int) -> bytes:
        if self.closed or not self._recv_buffer:
            return b""
        chunk_size = min(size, len(self._recv_buffer))
        chunk = bytes(self._recv_buffer[:chunk_size])
        del self._recv_buffer[:chunk_size]
        return chunk

    def close(self) -> None:
        self.closed = True


class ModbusTcpPlcAdapterTest(unittest.TestCase):
    def test_status_reports_adapter_name(self) -> None:
        adapter = ModbusTcpPlcAdapter("127.0.0.1", 502, unit_id=1)
        self.assertEqual(adapter.status()["adapter"], "ModbusTcpPlcAdapter")

        from backend.app.services.plc_adapter import TcpPlcAdapter

        compat_adapter = TcpPlcAdapter("127.0.0.1", 502, unit_id=1)
        self.assertEqual(compat_adapter.status()["adapter"], "TcpPlcAdapter")

    def test_coil_mode_writes_hold_and_release(self) -> None:
        fake_socket = _FakeSocket(
            _write_response(1, 7, 0x05, 12, 0xFF00)
            + _write_response(2, 7, 0x05, 12, 0x0000)
        )

        with mock.patch(
            "backend.app.services.plc_adapter.socket.create_connection",
            return_value=fake_socket,
        ) as create_connection:
            adapter = ModbusTcpPlcAdapter(
                "10.0.0.5",
                502,
                timeout_s=0.75,
                unit_id=7,
                command_mode="coil",
                hold_address=12,
                release_address=12,
            )

            self.assertFalse(adapter.status()["connected"])
            adapter.send_clamp_hold(event_id="evt-1", decision="ACCEPT")
            self.assertTrue(adapter.status()["connected"])
            adapter.send_clamp_release(event_id="evt-1", reason="auto")
            adapter.disconnect()

        create_connection.assert_called_once_with(("10.0.0.5", 502), timeout=0.75)
        self.assertTrue(fake_socket.closed)
        self.assertEqual(len(fake_socket.sent_packets), 2)

        first_packet = fake_socket.sent_packets[0]
        tid, protocol_id, length, unit_id = struct.unpack(">HHHB", first_packet[:7])
        function_code, address, value = struct.unpack(">BHH", first_packet[7:])
        self.assertEqual((tid, protocol_id, length, unit_id), (1, 0, 6, 7))
        self.assertEqual(function_code, 0x05)
        self.assertEqual(address, 12)
        self.assertEqual(value, 0xFF00)

        second_packet = fake_socket.sent_packets[1]
        tid, protocol_id, length, unit_id = struct.unpack(">HHHB", second_packet[:7])
        function_code, address, value = struct.unpack(">BHH", second_packet[7:])
        self.assertEqual((tid, protocol_id, length, unit_id), (2, 0, 6, 7))
        self.assertEqual(function_code, 0x05)
        self.assertEqual(address, 12)
        self.assertEqual(value, 0x0000)

    def test_register_mode_writes_distinct_addresses(self) -> None:
        fake_socket = _FakeSocket(
            _write_response(1, 3, 0x06, 20, 2)
            + _write_response(2, 3, 0x06, 21, 0)
        )

        with mock.patch(
            "backend.app.services.plc_adapter.socket.create_connection",
            return_value=fake_socket,
        ):
            adapter = ModbusTcpPlcAdapter(
                "192.168.1.50",
                5020,
                unit_id=3,
                command_mode="holding_register",
                hold_address=20,
                release_address=21,
                hold_value=2,
                release_value=0,
            )

            adapter.send_clamp_hold(event_id="evt-2", decision="REJECT")
            adapter.send_clamp_release(event_id="evt-2", reason="auto")

        self.assertEqual(len(fake_socket.sent_packets), 2)

        first_packet = fake_socket.sent_packets[0]
        tid, protocol_id, length, unit_id = struct.unpack(">HHHB", first_packet[:7])
        function_code, address, value = struct.unpack(">BHH", first_packet[7:])
        self.assertEqual((tid, protocol_id, length, unit_id), (1, 0, 6, 3))
        self.assertEqual(function_code, 0x06)
        self.assertEqual(address, 20)
        self.assertEqual(value, 2)

        second_packet = fake_socket.sent_packets[1]
        tid, protocol_id, length, unit_id = struct.unpack(">HHHB", second_packet[:7])
        function_code, address, value = struct.unpack(">BHH", second_packet[7:])
        self.assertEqual((tid, protocol_id, length, unit_id), (2, 0, 6, 3))
        self.assertEqual(function_code, 0x06)
        self.assertEqual(address, 21)
        self.assertEqual(value, 0)

    def test_coil_readback_verifies_state(self) -> None:
        fake_socket = _FakeSocket(
            _write_response(1, 9, 0x05, 8, 0xFF00)
            + struct.pack(">HHHB", 2, 0, 4, 9)
            + struct.pack(">BBB", 0x01, 0x01, 0x01)
            + _write_response(3, 9, 0x05, 8, 0x0000)
            + struct.pack(">HHHB", 4, 0, 4, 9)
            + struct.pack(">BBB", 0x01, 0x01, 0x00)
        )

        with mock.patch(
            "backend.app.services.plc_adapter.socket.create_connection",
            return_value=fake_socket,
        ):
            adapter = ModbusTcpPlcAdapter(
                "10.1.1.5",
                502,
                unit_id=9,
                command_mode="coil",
                hold_address=8,
                release_address=8,
                readback_enabled=True,
                readback_mode="coil",
                readback_address=8,
            )

            adapter.send_clamp_hold(event_id="evt-3", decision="ACCEPT")
            adapter.send_clamp_release(event_id="evt-3", reason="auto")

        self.assertEqual(len(fake_socket.sent_packets), 4)
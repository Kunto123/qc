"""Machine Settings model — per-machine PLC wiring + transport config.

Stored as JSON on local disk (standalone machine). Seeded once from env vars
on first boot; after that DB is the single source of truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PlcConnectionConfig:
    enabled: bool = False
    dry_run: bool = True
    transport: str = "tcp"  # "tcp" | "rtu" | "fx"
    host: str = "127.0.0.1"
    port: int = 5020
    serial_port: str = ""
    serial_baudrate: int = 9600
    serial_parity: str = "N"
    serial_bytesize: int = 8
    serial_stopbits: int = 1
    timeout_ms: int = 1000
    modbus_unit_id: int = 255
    modbus_command_mode: str = "coil"
    modbus_hold_address: int = 0
    modbus_release_address: int = 0
    modbus_hold_value: int = 1
    modbus_release_value: int = 0
    modbus_zero_based_addressing: bool = True
    modbus_readback_enabled: bool = False
    modbus_readback_mode: str = "discrete_input"
    modbus_readback_address: int = 0
    modbus_readback_expected_hold_value: int = 1
    modbus_readback_expected_release_value: int = 0


@dataclass(slots=True)
class StickerModeConfig:
    """PLC I/O map + timing for QC Sticker mode."""
    # Relay coil addresses (CH1=Enji Buzzer, CH2=OK Light+Buzzer, CH3=Clamp, CH4=Spare)
    relay_clamp_address: int = 3
    relay_ok_light_buzzer_address: int = 2
    relay_enji_buzzer_address: int = 1
    relay_spare_address: int = 0
    # Input addresses
    input_release_address: int = 0
    input_template_address: int = 1
    input_clamp_engaged_address: int = 2
    # Feedback
    clamp_feedback_enabled: bool = False
    clamp_feedback_timeout_ms: int = 1500
    clamp_feedback_fallback_delay_ms: int = 300
    # Timing
    accept_pulse_ms: int = 1000
    clamp_hold_ms: int = 2000
    min_reclamp_interval_ms: int = 3000
    release_input_debounce_ms: int = 200


@dataclass(slots=True)
class CounterModeConfig:
    """PLC I/O map + timing for Component Counter mode (placeholder)."""
    # Relay coil addresses — same physical relays, different semantic mapping
    relay_clamp_address: int = 3
    relay_ok_light_buzzer_address: int = 2
    relay_enji_buzzer_address: int = 1
    relay_spare_address: int = 0
    # Input addresses
    input_sensor_address: int = 0
    input_release_address: int = 1
    input_template_address: int = 2
    input_clamp_engaged_address: int = 2
    # Feedback
    clamp_feedback_enabled: bool = False
    clamp_feedback_timeout_ms: int = 1500
    clamp_feedback_fallback_delay_ms: int = 300
    # Timing
    accept_pulse_ms: int = 1000
    clamp_hold_ms: int = 2000
    min_reclamp_interval_ms: int = 3000
    release_input_debounce_ms: int = 200


@dataclass(slots=True)
class MachineSettings:
    """Top-level machine settings — one record per machine."""
    version: int = 1
    seeded_from_env: bool = False
    seeded_at: str = ""
    connection: PlcConnectionConfig = field(default_factory=PlcConnectionConfig)
    sticker: StickerModeConfig = field(default_factory=StickerModeConfig)
    counter: CounterModeConfig = field(default_factory=CounterModeConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seeded_from_env": self.seeded_from_env,
            "seeded_at": self.seeded_at,
            "connection": asdict(self.connection),
            "sticker": asdict(self.sticker),
            "counter": asdict(self.counter),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineSettings:
        conn = PlcConnectionConfig(**{
            k: v for k, v in (data.get("connection") or {}).items()
            if k in PlcConnectionConfig.__slots__
        })
        sticker = StickerModeConfig(**{
            k: v for k, v in (data.get("sticker") or {}).items()
            if k in StickerModeConfig.__slots__
        })
        counter = CounterModeConfig(**{
            k: v for k, v in (data.get("counter") or {}).items()
            if k in CounterModeConfig.__slots__
        })
        return cls(
            version=int(data.get("version", 1)),
            seeded_from_env=bool(data.get("seeded_from_env", False)),
            seeded_at=str(data.get("seeded_at", "")),
            connection=conn,
            sticker=sticker,
            counter=counter,
        )

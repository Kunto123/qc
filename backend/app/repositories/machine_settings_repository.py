"""Machine Settings repository — JSON store with idempotent env seed.

Seed logic:
  - On first boot (no DB file), create from env vars, mark seeded_from_env=True.
  - On subsequent boots, if seeded_from_env=True, skip — DB is source of truth.
  - If seeded_from_env=False (manual entry), never overwrite.
  - Explicit POST /seed endpoint allows re-seed from env (admin recovery).
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig
from backend.app.models.machine_settings import (
    CounterModeConfig,
    MachineSettings,
    PlcConnectionConfig,
    StickerModeConfig,
)
from backend.app.repositories.base_json import JsonRepository

logger = logging.getLogger(__name__)

_SETTINGS_VERSION = 1


def _build_default_from_env(config: AppConfig) -> dict[str, Any]:
    """Build a MachineSettings dict from current env vars (seed)."""
    now = datetime.now(UTC).isoformat()
    return {
        "version": _SETTINGS_VERSION,
        "seeded_from_env": True,
        "seeded_at": now,
        "connection": {
            "enabled": config.plc_enabled,
            "dry_run": config.plc_dry_run,
            "transport": config.plc_transport,
            "host": config.plc_host,
            "port": config.plc_port,
            "serial_port": config.plc_serial_port,
            "serial_baudrate": config.plc_serial_baudrate,
            "serial_parity": config.plc_serial_parity,
            "serial_bytesize": config.plc_serial_bytesize,
            "serial_stopbits": config.plc_serial_stopbits,
            "timeout_ms": config.plc_timeout_ms,
            "modbus_unit_id": config.plc_modbus_unit_id,
            "modbus_command_mode": config.plc_modbus_command_mode,
            "modbus_hold_address": config.plc_modbus_hold_address,
            "modbus_release_address": config.plc_modbus_release_address,
            "modbus_hold_value": config.plc_modbus_hold_value,
            "modbus_release_value": config.plc_modbus_release_value,
            "modbus_zero_based_addressing": config.plc_modbus_zero_based_addressing,
            "modbus_readback_enabled": config.plc_modbus_readback_enabled,
            "modbus_readback_mode": config.plc_modbus_readback_mode,
            "modbus_readback_address": config.plc_modbus_readback_address,
            "modbus_readback_expected_hold_value": config.plc_modbus_readback_expected_hold_value,
            "modbus_readback_expected_release_value": config.plc_modbus_readback_expected_release_value,
        },
        "sticker": {
            "relay_clamp_address": config.plc_relay_clamp_address,
            "relay_ok_light_buzzer_address": config.plc_relay_ok_light_buzzer_address,
            "relay_enji_buzzer_address": config.plc_relay_enji_buzzer_address,
            "relay_spare_address": config.plc_relay_spare_address,
            "input_release_address": config.plc_input_release_address,
            "input_template_address": config.plc_input_template_address,
            "input_clamp_engaged_address": config.plc_input_clamp_engaged_address,
            "clamp_feedback_enabled": config.plc_clamp_feedback_enabled,
            "clamp_feedback_timeout_ms": config.plc_clamp_feedback_timeout_ms,
            "clamp_feedback_fallback_delay_ms": config.plc_clamp_feedback_fallback_delay_ms,
            "accept_pulse_ms": config.plc_accept_pulse_ms,
            "clamp_hold_ms": config.plc_clamp_hold_ms,
            "min_reclamp_interval_ms": config.plc_min_reclamp_interval_ms,
            "release_input_debounce_ms": config.plc_release_input_debounce_ms,
        },
        "counter": {
            # Counter mode defaults — same physical wiring, different semantics
            "relay_clamp_address": config.plc_relay_clamp_address,
            "relay_ok_light_buzzer_address": config.plc_relay_ok_light_buzzer_address,
            "relay_enji_buzzer_address": config.plc_relay_enji_buzzer_address,
            "relay_spare_address": config.plc_relay_spare_address,
            "input_sensor_address": 0,
            "input_release_address": config.plc_input_release_address,
            "input_template_address": config.plc_input_template_address,
            "clamp_feedback_enabled": config.plc_clamp_feedback_enabled,
            "clamp_feedback_timeout_ms": config.plc_clamp_feedback_timeout_ms,
            "clamp_feedback_fallback_delay_ms": config.plc_clamp_feedback_fallback_delay_ms,
            "accept_pulse_ms": config.plc_accept_pulse_ms,
            "clamp_hold_ms": config.plc_clamp_hold_ms,
            "min_reclamp_interval_ms": config.plc_min_reclamp_interval_ms,
            "release_input_debounce_ms": config.plc_release_input_debounce_ms,
        },
    }


class MachineSettingsRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("machine_settings.json", {})

    def load_settings(self) -> MachineSettings:
        raw = self.load()
        if not raw:
            return MachineSettings()
        return MachineSettings.from_dict(raw)

    def save_settings(self, settings: MachineSettings) -> None:
        self.save(settings.to_dict())

    def seed_from_env(self, config: AppConfig, *, force: bool = False) -> bool:
        """Seed DB from env vars. Returns True if seed was applied.

        Idempotent: only seeds if DB is empty or force=True.
        """
        raw = self.load()
        if raw and not force:
            logger.info(
                "[machine-settings] DB already exists (seeded=%s) — skipping env seed",
                raw.get("seeded_from_env", False),
            )
            return False

        data = _build_default_from_env(config)
        self.save(data)
        logger.info("[machine-settings] seeded from env vars (force=%s)", force)
        return True

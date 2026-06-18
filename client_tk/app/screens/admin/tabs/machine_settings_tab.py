"""Machine Settings tab — PLC wiring + transport config UI.

Admin-only tab for configuring:
  - Connection/transport (TCP/RTU, addresses, timeouts)
  - I/O address map per mode (Sticker / Counter)
  - Cycle timing
  - Diagnostics (live status, test coil, read inputs)
"""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import customtkinter as ctk

from client_tk.app.components.scrollable_frame import ScrollableFrame
from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    BORDER,
    PANEL_ALT_BG,
    PANEL_BG,
    SUCCESS,
    SUCCESS_HOVER,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    WARNING_HOVER,
)


class MachineSettingsTab:
    """Machine / PLC Settings tab in Admin screen."""

    def __init__(self, admin, tab_frame):
        self.admin = admin
        self.frame = tab_frame
        self._settings: dict[str, Any] = {}
        self._field_vars: dict[str, tk.StringVar] = {}
        self._build()
        self._load_settings()

    # ── Build ─────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        body = self.admin._make_scrollable_body(self.frame, "MachineSettings")

        # ── Connection section ──
        self._build_connection_section(body, 0)

        # ── Sticker Mode section ──
        self._build_sticker_section(body, 1)

        # ── Counter Mode section (placeholder) ──
        self._build_counter_section(body, 2)

        # ── Diagnostics section ──
        self._build_diagnostics_section(body, 3)

        # ── Action buttons ──
        self._build_actions(body, 4)

    def _section_frame(self, parent, row: int, title: str) -> ctk.CTkFrame:
        section = ctk.CTkFrame(parent, fg_color=PANEL_ALT_BG, corner_radius=12, border_width=1, border_color=BORDER)
        section.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 10))
        section.columnconfigure(1, weight=1)
        ctk.CTkLabel(
            section, text=title, font=("Segoe UI", 11, "bold"), text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 6))
        return section

    def _add_field(self, parent, row: int, label: str, key: str, default: str = "") -> tk.StringVar:
        var = tk.StringVar(value=default)
        self._field_vars[key] = var
        ctk.CTkLabel(parent, text=f"{label}:", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
            row=row, column=0, sticky="w", padx=(12, 8), pady=2,
        )
        entry = ctk.CTkEntry(parent, textvariable=var, width=120)
        entry.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=2)
        return var

    def _add_checkbox(self, parent, row: int, label: str, key: str, default: bool = False) -> tk.BooleanVar:
        var = tk.BooleanVar(value=default)
        self._field_vars[key] = var
        cb = ctk.CTkCheckBox(parent, text=label, variable=var, font=("Segoe UI", 9), text_color=TEXT_PRIMARY)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=2)
        return var

    # ── Connection section ───────────────────────────────────────────

    def _build_connection_section(self, parent, row: int) -> None:
        sec = self._section_frame(parent, row, "Connection / Transport")
        r = 1
        self._add_checkbox(sec, r, "Enabled", "connection.enabled"); r += 1
        self._add_checkbox(sec, r, "Dry Run (log only)", "connection.dry_run"); r += 1
        self._add_field(sec, r, "Transport (tcp/rtu)", "connection.transport", "tcp"); r += 1
        self._add_field(sec, r, "Host", "connection.host", "127.0.0.1"); r += 1
        self._add_field(sec, r, "Port", "connection.port", "5020"); r += 1
        self._add_field(sec, r, "Serial Port", "connection.serial_port", ""); r += 1
        self._add_field(sec, r, "Baudrate", "connection.serial_baudrate", "9600"); r += 1
        self._add_field(sec, r, "Parity", "connection.serial_parity", "N"); r += 1
        self._add_field(sec, r, "Byte Size", "connection.serial_bytesize", "8"); r += 1
        self._add_field(sec, r, "Stop Bits", "connection.serial_stopbits", "1"); r += 1
        self._add_field(sec, r, "Timeout (ms)", "connection.timeout_ms", "1000"); r += 1
        self._add_field(sec, r, "Modbus Unit ID", "connection.modbus_unit_id", "255"); r += 1

    # ── Sticker Mode section ─────────────────────────────────────────

    def _build_sticker_section(self, parent, row: int) -> None:
        sec = self._section_frame(parent, row, "Sticker Mode — I/O Addresses")
        r = 1
        ctk.CTkLabel(
            sec, text="Relay Coil Addresses", font=("Segoe UI", 9, "bold"), text_color=TEXT_SECONDARY,
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0)); r += 1
        self._add_field(sec, r, "Clamp (CH3)", "sticker.relay_clamp_address", "3"); r += 1
        self._add_field(sec, r, "OK Light+Buzzer (CH2)", "sticker.relay_ok_light_buzzer_address", "2"); r += 1
        self._add_field(sec, r, "Enji Buzzer (CH1)", "sticker.relay_enji_buzzer_address", "1"); r += 1
        self._add_field(sec, r, "Spare (CH4)", "sticker.relay_spare_address", "0"); r += 1

        ctk.CTkLabel(
            sec, text="Input Addresses", font=("Segoe UI", 9, "bold"), text_color=TEXT_SECONDARY,
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 0)); r += 1
        self._add_field(sec, r, "Release (IN1)", "sticker.input_release_address", "0"); r += 1
        self._add_field(sec, r, "Template Cycle (IN2)", "sticker.input_template_address", "1"); r += 1
        self._add_field(sec, r, "Clamp Feedback (IN3)", "sticker.input_clamp_engaged_address", "2"); r += 1

        ctk.CTkLabel(
            sec, text="Feedback & Timing", font=("Segoe UI", 9, "bold"), text_color=TEXT_SECONDARY,
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 0)); r += 1
        self._add_checkbox(sec, r, "Clamp Feedback Enabled", "sticker.clamp_feedback_enabled"); r += 1
        self._add_field(sec, r, "Feedback Timeout (ms)", "sticker.clamp_feedback_timeout_ms", "1500"); r += 1
        self._add_field(sec, r, "Feedback Fallback Delay (ms)", "sticker.clamp_feedback_fallback_delay_ms", "300"); r += 1
        self._add_field(sec, r, "Accept Pulse (ms)", "sticker.accept_pulse_ms", "1000"); r += 1
        self._add_field(sec, r, "Clamp Hold (ms)", "sticker.clamp_hold_ms", "2000"); r += 1
        self._add_field(sec, r, "Min Reclamp Interval (ms)", "sticker.min_reclamp_interval_ms", "3000"); r += 1
        self._add_field(sec, r, "Release Debounce (ms)", "sticker.release_input_debounce_ms", "200"); r += 1

    # ── Counter Mode section ─────────────────────────────────────────

    def _build_counter_section(self, parent, row: int) -> None:
        sec = self._section_frame(parent, row, "Counter Mode — I/O Addresses (placeholder)")
        r = 1
        ctk.CTkLabel(
            sec,
            text="Counter mode is not yet implemented. These settings are stored but not used.",
            font=("Segoe UI", 9, "italic"), text_color=WARNING,
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6)); r += 1

        self._add_field(sec, r, "Sensor Input Addr", "counter.input_sensor_address", "0"); r += 1
        self._add_field(sec, r, "Release Input Addr", "counter.input_release_address", "1"); r += 1
        self._add_field(sec, r, "Template Input Addr", "counter.input_template_address", "2"); r += 1
        self._add_field(sec, r, "Clamp Feedback Addr", "counter.input_clamp_engaged_address", "2"); r += 1
        self._add_checkbox(sec, r, "Clamp Feedback Enabled", "counter.clamp_feedback_enabled"); r += 1
        self._add_field(sec, r, "Accept Pulse (ms)", "counter.accept_pulse_ms", "1000"); r += 1
        self._add_field(sec, r, "Min Reclamp Interval (ms)", "counter.min_reclamp_interval_ms", "3000"); r += 1

    # ── Diagnostics section ──────────────────────────────────────────

    def _build_diagnostics_section(self, parent, row: int) -> None:
        sec = self._section_frame(parent, row, "Diagnostics / Commissioning")
        r = 1

        # Live status display
        self._diag_status_var = tk.StringVar(value="Not connected")
        ctk.CTkLabel(sec, text="PLC Status:", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
            row=r, column=0, sticky="w", padx=12, pady=2,
        )
        ctk.CTkLabel(sec, textvariable=self._diag_status_var, font=("Segoe UI", 9), text_color=TEXT_SECONDARY).grid(
            row=r, column=1, sticky="w", padx=(0, 12), pady=2,
        ); r += 1

        # Input snapshot display
        self._diag_inputs_var = tk.StringVar(value="No data")
        ctk.CTkLabel(sec, text="Input Snapshot:", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
            row=r, column=0, sticky="w", padx=12, pady=2,
        )
        ctk.CTkLabel(sec, textvariable=self._diag_inputs_var, font=("Segoe UI", 9), text_color=TEXT_SECONDARY).grid(
            row=r, column=1, sticky="w", padx=(0, 12), pady=2,
        ); r += 1

        # Test coil
        ctk.CTkLabel(sec, text="Test Coil Address:", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
            row=r, column=0, sticky="w", padx=12, pady=2,
        )
        self._test_coil_addr = tk.StringVar(value="0")
        ctk.CTkEntry(sec, textvariable=self._test_coil_addr, width=80).grid(
            row=r, column=1, sticky="w", padx=(0, 12), pady=2,
        ); r += 1

        ctk.CTkLabel(sec, text="Pulse Duration (ms):", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
            row=r, column=0, sticky="w", padx=12, pady=2,
        )
        self._test_coil_duration = tk.StringVar(value="500")
        ctk.CTkEntry(sec, textvariable=self._test_coil_duration, width=80).grid(
            row=r, column=1, sticky="w", padx=(0, 12), pady=2,
        ); r += 1

        # Diag buttons
        btn_frame = ctk.CTkFrame(sec, fg_color="transparent")
        btn_frame.grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 10))

        ctk.CTkButton(
            btn_frame, text="Refresh Status", width=120,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT,
            command=self._refresh_diagnostics,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_frame, text="Test Coil", width=100,
            fg_color=WARNING, hover_color=WARNING_HOVER, text_color="#000000",
            command=self._test_coil,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_frame, text="All Off (Emergency)", width=140,
            fg_color="#dc2626", hover_color="#991b1b", text_color="#ffffff",
            command=self._emergency_all_off,
        ).pack(side="left", padx=(0, 6))

    # ── Action buttons ────────────────────────────────────────────────

    def _build_actions(self, parent, row: int) -> None:
        sec = ctk.CTkFrame(parent, fg_color="transparent")
        sec.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 20))

        ctk.CTkButton(
            sec, text="Save Settings", width=140,
            fg_color=SUCCESS, hover_color=SUCCESS_HOVER, text_color=TEXT_ON_ACCENT,
            command=self._save_settings,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            sec, text="Reload from DB", width=140,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT,
            command=self._load_settings,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            sec, text="Re-seed from .env", width=140,
            fg_color=WARNING, hover_color=WARNING_HOVER, text_color="#000000",
            command=self._reseed_from_env,
        ).pack(side="left", padx=(0, 8))

        # Seed status
        self._seed_status_var = tk.StringVar(value="")
        ctk.CTkLabel(
            sec, textvariable=self._seed_status_var, font=("Segoe UI", 9), text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(12, 0))

    # ── Data loading / saving ─────────────────────────────────────────

    def _load_settings(self) -> None:
        """Load settings from API and populate fields."""
        try:
            self._settings = self.admin.api.get_machine_settings()
            self._populate_fields(self._settings)
            seeded = self._settings.get("seeded_from_env", False)
            self._seed_status_var.set(
                f"{'Seeded from env' if seeded else 'User-edited'} | v{self._settings.get('version', 1)}"
            )
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load settings: {exc}")

    def _populate_fields(self, data: dict, prefix: str = "") -> None:
        """Recursively populate field vars from nested dict."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._populate_fields(value, full_key)
            elif full_key in self._field_vars:
                var = self._field_vars[full_key]
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))
                else:
                    var.set(str(value) if value is not None else "")

    def _collect_fields(self) -> dict:
        """Collect field values into nested dict matching API schema."""
        result: dict = {}
        for key, var in self._field_vars.items():
            parts = key.split(".")
            d = result
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            val = var.get()
            # Try to parse as int/float/bool
            if isinstance(var, tk.BooleanVar):
                d[parts[-1]] = bool(val)
            else:
                try:
                    d[parts[-1]] = int(val)
                except (ValueError, TypeError):
                    try:
                        d[parts[-1]] = float(val)
                    except (ValueError, TypeError):
                        d[parts[-1]] = str(val)
        return result

    def _save_settings(self) -> None:
        """Save settings to API."""
        payload = self._collect_fields()
        try:
            self._settings = self.admin.api.update_machine_settings(payload)
            self._seed_status_var.set("Saved (user-edited)")
            messagebox.showinfo("Success", "Machine settings saved.")
        except Exception as exc:
            messagebox.showerror("Error", f"Save failed: {exc}")

    def _reseed_from_env(self) -> None:
        """Re-seed settings from env vars (with confirmation)."""
        if not messagebox.askyesno(
            "Confirm Re-seed",
            "This will overwrite current settings with values from .env file.\n\nContinue?",
        ):
            return
        try:
            data = self.admin.api.seed_machine_settings(force=True)
            self._settings = data.get("settings", {})
            self._populate_fields(self._settings)
            self._seed_status_var.set("Re-seeded from env")
            messagebox.showinfo("Success", "Settings re-seeded from .env")
        except Exception as exc:
            messagebox.showerror("Error", f"Re-seed failed: {exc}")

    # ── Diagnostics ───────────────────────────────────────────────────

    def _refresh_diagnostics(self) -> None:
        """Refresh PLC diagnostics from API."""
        try:
            data = self.admin.api.get_plc_diagnostics()
            if data.get("enabled"):
                state = data.get("state", "?")
                strategy = data.get("strategy", "?")
                connected = data.get("connected", False)
                self._diag_status_var.set(
                    f"State={state} | Strategy={strategy} | Connected={connected}"
                )
                inputs = data.get("last_input_snapshot", [])
                self._diag_inputs_var.set(str(inputs) if inputs else "No data")
            else:
                self._diag_status_var.set("PLC disabled")
                self._diag_inputs_var.set("N/A")
        except Exception as exc:
            self._diag_status_var.set(f"Error: {exc}")

    def _test_coil(self) -> None:
        """Pulse a coil for wiring test."""
        try:
            addr = int(self._test_coil_addr.get())
            duration = int(self._test_coil_duration.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid address or duration")
            return

        if not messagebox.askyesno(
            "Confirm Coil Test",
            f"Pulse coil address {addr} for {duration}ms?\n\n"
            "This will fire a real relay if dry_run=False!",
        ):
            return

        try:
            self.admin.api.test_plc_coil(addr, duration, confirm=True)
            messagebox.showinfo("Success", f"Coil {addr} pulsed for {duration}ms")
        except Exception as exc:
            messagebox.showerror("Error", f"Test failed: {exc}")

    def _emergency_all_off(self) -> None:
        """Emergency all-off."""
        if not messagebox.askyesno("EMERGENCY ALL OFF", "Turn off ALL coils immediately?"):
            return
        try:
            self.admin.api.plc_all_off()
            messagebox.showinfo("Success", "All coils OFF")
        except Exception as exc:
            messagebox.showerror("Error", f"All-off failed: {exc}")

from __future__ import annotations

import datetime
import secrets
import string
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

from client_tk.app.components.async_bridge import run_async
from client_tk.app.components.scrollable_frame import AutoHideScrollbar, ScrollableFrame
from client_tk.app.components.template_forms import LabeledValuePanel
from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    APP_BG,
    BORDER,
    PANEL_ALT_BG,
    PANEL_BG,
    SHELL_BG,
    SUCCESS,
    SUCCESS_HOVER,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


RESPONSIVE_BREAKPOINT = 1180


def _safe_text(value: object, fallback: str = "-") -> str:
    text = str(value or "").strip()
    return text or fallback


def _format_timestamp(value: object) -> str:
    text = _safe_text(value)
    if text == "-":
        return text
    return text.replace("T", " ")[:19]


def _format_status(value: object) -> str:
    return "Active" if bool(value) else "Inactive"


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _random_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(max(16, int(length))))


class CompactStatCard(ctk.CTkFrame):
    def __init__(self, master, title: str, *, background: str, foreground: str):
        super().__init__(master, fg_color=background, corner_radius=8, border_width=1, border_color=BORDER)
        self.columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text=title, text_color=foreground, font=("Segoe UI", 9, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(8, 0),
        )
        self.value_label = ctk.CTkLabel(self, text="0", text_color=foreground, font=("Segoe UI", 16, "bold"))
        self.value_label.grid(row=1, column=0, sticky="w", padx=10)
        self.note_label = ctk.CTkLabel(self, text="", text_color=foreground, font=("Segoe UI", 8))
        self.note_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))

    def set_value(self, value: object, note: str = "") -> None:
        self.value_label.configure(text=str(value))
        self.note_label.configure(text=str(note or ""))


class AdminScreen(ctk.CTkFrame):
    """Production-facing Admin workspace.

    Admin only handles preset deployment, RFID operator onboarding, and monitoring.
    Full template/model/debug controls stay in the Engineer screen.
    """

    def __init__(self, master, api_client, session_state):
        super().__init__(master, fg_color=APP_BG, corner_radius=0)
        self.api = api_client
        self.state = session_state

        self._layout_compact: bool | None = None
        self._overview_cards_visible = True
        self._last_refresh: dict[str, str] = {}

        self._deployments_cache: list[dict] = []
        self._templates_cache: list[dict] = []
        self._models_cache: list[dict] = []
        self._model_lookup: dict[str, dict] = {}
        self._users_cache: list[dict] = []
        self._results_cache: list[dict] = []

        self.current_template_id: int | None = None
        self.current_template_version_id: int | None = None
        self._editing_deployment_id: int | None = None

        self.status_var = tk.StringVar(value="Admin ready.")
        self.refresh_time_var = tk.StringVar(value="")

        self.preset_name_var = tk.StringVar()
        self.preset_description_var = tk.StringVar()
        self.preset_line_var = tk.StringVar()
        self.preset_station_var = tk.StringVar()
        self.preset_model_choice_var = tk.StringVar()
        self.preset_model_path_var = tk.StringVar()
        self.preset_model_meta_path_var = tk.StringVar()
        self.preset_expected_code_var = tk.StringVar()
        self.part_ready_roi_x_var = tk.StringVar(value="0.2")
        self.part_ready_roi_y_var = tk.StringVar(value="0.2")
        self.part_ready_roi_w_var = tk.StringVar(value="0.25")
        self.part_ready_roi_h_var = tk.StringVar(value="0.25")
        self.sticker_roi_x_var = tk.StringVar(value="0.2")
        self.sticker_roi_y_var = tk.StringVar(value="0.2")
        self.sticker_roi_w_var = tk.StringVar(value="0.6")
        self.sticker_roi_h_var = tk.StringVar(value="0.6")

        self.operator_username_var = tk.StringVar()
        self.operator_rfid_var = tk.StringVar()

        self.monitor_line_var = tk.StringVar()
        self.monitor_station_var = tk.StringVar()
        self.monitor_context_var = tk.StringVar(value="Recent production activity.")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabs()
        self._build_status_bar()
        self.bind("<Configure>", self._on_resize)
        self.after_idle(self._apply_responsive_layout)

        self.refresh_all()

    # ------------------------------------------------------------------
    # Layout
    def _build_header(self) -> None:
        self.header = ctk.CTkFrame(self, fg_color=SHELL_BG, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 6))
        self.header.columnconfigure(0, weight=1)

        user = self.state.user or {}
        identity = f"{_safe_text(user.get('username'))} ({_safe_text(user.get('role'), 'admin')})"
        ctk.CTkLabel(
            self.header,
            text="Admin Production Setup",
            font=("Segoe UI", 14, "bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(
            self.header,
            text=f"{identity} | Presets, operators, and production monitor.",
            text_color=TEXT_SECONDARY,
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))
        ctk.CTkButton(
            self.header,
            text="Refresh",
            command=self.refresh_all,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            height=30,
            corner_radius=6,
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=12, pady=10)

        self.overview_cards_frame = ctk.CTkFrame(self.header, fg_color="transparent", corner_radius=0)
        self.overview_cards_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))
        for index in range(4):
            self.overview_cards_frame.columnconfigure(index, weight=1)
        self.admin_cards = {
            "presets": CompactStatCard(self.overview_cards_frame, "Active Presets", background="#0f172a", foreground="#f8fafc"),
            "operators": CompactStatCard(self.overview_cards_frame, "Operators", background="#134e4a", foreground="#ecfdf5"),
            "accept": CompactStatCard(self.overview_cards_frame, "Accept", background="#166534", foreground="#f0fdf4"),
            "reject": CompactStatCard(self.overview_cards_frame, "Reject", background="#991b1b", foreground="#fef2f2"),
        }
        self._layout_overview_cards(compact=False)

    def _build_tabs(self) -> None:
        notebook = ctk.CTkTabview(self, fg_color=APP_BG, corner_radius=0)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        for tab_name in ("Presets", "Operators", "Monitor"):
            notebook.add(tab_name)

        original_tab = notebook.tab

        def _tabs() -> list[str]:
            return ["Presets", "Operators", "Monitor"]

        def _select(tab_id: str | None = None):
            if tab_id is None:
                return notebook.get()
            notebook.set(tab_id)
            return tab_id

        def _tab(tab_id: str, option: str | None = None):
            if option == "text":
                return tab_id
            return original_tab(tab_id)

        notebook.tabs = _tabs  # type: ignore[attr-defined]
        notebook.select = _select  # type: ignore[attr-defined]
        notebook.tab = _tab  # type: ignore[attr-defined]

        self._notebook = notebook
        self.presets_tab = notebook.tab("Presets")
        self.operators_tab = notebook.tab("Operators")
        self.monitor_tab = notebook.tab("Monitor")

        self._build_presets_tab()
        self._build_operators_tab()
        self._build_monitor_tab()

    def _build_status_bar(self) -> None:
        status_bar = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        status_bar.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        status_bar.columnconfigure(0, weight=1)
        ctk.CTkLabel(status_bar, textvariable=self.status_var, text_color=TEXT_SECONDARY).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(status_bar, textvariable=self.refresh_time_var, text_color=TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="e")

    def _build_presets_tab(self) -> None:
        self.presets_tab.columnconfigure(0, weight=3)
        self.presets_tab.columnconfigure(1, weight=2)
        self.presets_tab.rowconfigure(0, weight=1)

        self.presets_left = ttk.Frame(self.presets_tab, padding=8)
        self.presets_right = ttk.Frame(self.presets_tab, padding=8)
        self.presets_left.grid(row=0, column=0, sticky="nsew")
        self.presets_right.grid(row=0, column=1, sticky="nsew")

        listing = ctk.CTkFrame(self.presets_left, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        listing.pack(fill="both", expand=True)
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(2, weight=1)
        ctk.CTkLabel(listing, text="Active Production Presets", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(listing, text="One active preset per line and station.", text_color=TEXT_SECONDARY).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))
        self.preset_table = self._build_table(
            listing,
            [
                ("id", "ID", 55, "center"),
                ("line", "Line", 110, "w"),
                ("station", "Station", 100, "w"),
                ("preset", "Preset", 220, "w"),
                ("version", "Version", 80, "center"),
                ("status", "Status", 90, "center"),
            ],
            row=2,
            height=18,
        )
        self.preset_table.bind("<<TreeviewSelect>>", self._on_preset_selected)
        footer = self._build_action_row(
            listing,
            [
                ("Refresh", self.refresh_presets, "neutral", "left"),
                ("New Preset", self.reset_preset_wizard, "neutral", "right"),
                ("Deactivate Selected", self.deactivate_selected_preset, "neutral", "right"),
            ],
        )
        footer.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 10))

        wizard = ctk.CTkFrame(self.presets_right, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        wizard.pack(fill="both", expand=True)
        wizard.columnconfigure(1, weight=1)
        wizard.columnconfigure(3, weight=1)
        ctk.CTkLabel(wizard, text="Preset Wizard", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(wizard, text="Fill only production-critical values. Technical defaults are applied automatically.", text_color=TEXT_SECONDARY, wraplength=520, justify="left").grid(row=1, column=0, columnspan=4, sticky="w", padx=12, pady=(2, 10))

        self._entry(wizard, 2, 0, "Preset Name", self.preset_name_var, columnspan=3)
        self._entry(wizard, 3, 0, "Description", self.preset_description_var, columnspan=3)
        self._entry(wizard, 4, 0, "Line", self.preset_line_var)
        self._entry(wizard, 4, 2, "Station", self.preset_station_var)

        ttk.Label(wizard, text="Model").grid(row=5, column=0, sticky="w", padx=(12, 8), pady=5)
        self.preset_model_selector = ttk.Combobox(wizard, textvariable=self.preset_model_choice_var, state="readonly")
        self.preset_model_selector.grid(row=5, column=1, columnspan=3, sticky="ew", padx=(0, 12), pady=5)
        self.preset_model_selector.bind("<<ComboboxSelected>>", self._on_preset_model_selected)
        self._entry(wizard, 6, 0, "Sticker Code", self.preset_expected_code_var, columnspan=3)

        ctk.CTkLabel(wizard, text="Part Ready ROI", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(row=7, column=0, columnspan=4, sticky="w", padx=12, pady=(12, 2))
        self._roi_entries(wizard, 8, self.part_ready_roi_x_var, self.part_ready_roi_y_var, self.part_ready_roi_w_var, self.part_ready_roi_h_var)
        ctk.CTkLabel(wizard, text="Sticker ROI", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(row=9, column=0, columnspan=4, sticky="w", padx=12, pady=(12, 2))
        self._roi_entries(wizard, 10, self.sticker_roi_x_var, self.sticker_roi_y_var, self.sticker_roi_w_var, self.sticker_roi_h_var)

        ctk.CTkButton(
            wizard,
            text="Save & Deploy Preset",
            command=self.save_and_deploy_preset,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            height=34,
            corner_radius=6,
        ).grid(row=11, column=0, columnspan=4, sticky="ew", padx=12, pady=(16, 10))

    def _build_operators_tab(self) -> None:
        self.operators_tab.columnconfigure(0, weight=3)
        self.operators_tab.columnconfigure(1, weight=2)
        self.operators_tab.rowconfigure(0, weight=1)

        self.operators_left = ttk.Frame(self.operators_tab, padding=8)
        self.operators_right = ttk.Frame(self.operators_tab, padding=8)
        self.operators_left.grid(row=0, column=0, sticky="nsew")
        self.operators_right.grid(row=0, column=1, sticky="nsew")

        listing = ctk.CTkFrame(self.operators_left, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        listing.pack(fill="both", expand=True)
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(2, weight=1)
        ctk.CTkLabel(listing, text="Operators", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(listing, text="RFID login users for production operation.", text_color=TEXT_SECONDARY).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))
        self.users_table = self._build_table(
            listing,
            [
                ("id", "ID", 60, "center"),
                ("username", "Username", 190, "w"),
                ("role", "Role", 110, "center"),
                ("status", "Status", 100, "center"),
                ("rfid", "RFID", 100, "center"),
            ],
            row=2,
            height=18,
        )
        footer = self._build_action_row(listing, [("Refresh", self.refresh_operators, "neutral", "left")])
        footer.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 10))

        form = ctk.CTkFrame(self.operators_right, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        ctk.CTkLabel(form, text="RFID Quick Add", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(form, text="Create an operator and bind the scanned card in one step.", text_color=TEXT_SECONDARY, wraplength=420, justify="left").grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 10))
        self._entry(form, 2, 0, "Username", self.operator_username_var, columns=2)
        self._entry(form, 3, 0, "RFID UID", self.operator_rfid_var, columns=2)
        ctk.CTkButton(
            form,
            text="Create Operator",
            command=self.create_operator_from_rfid,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            height=32,
            corner_radius=6,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 10))

    def _build_monitor_tab(self) -> None:
        self.monitor_tab.columnconfigure(0, weight=1)
        self.monitor_tab.rowconfigure(2, weight=1)

        filters = ctk.CTkFrame(self.monitor_tab, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        filters.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        for index in range(6):
            filters.columnconfigure(index, weight=1 if index % 2 else 0)
        ctk.CTkLabel(filters, text="Production Monitor", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, columnspan=6, sticky="w", padx=10, pady=(10, 0))
        self._entry(filters, 1, 0, "Line", self.monitor_line_var)
        self._entry(filters, 1, 2, "Station", self.monitor_station_var)
        ctk.CTkButton(filters, text="Refresh", command=self.refresh_monitor, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT, height=28, corner_radius=6).grid(row=1, column=4, sticky="e", padx=(8, 4), pady=(4, 10))
        ctk.CTkButton(filters, text="Export CSV", command=self.export_monitor_csv, fg_color=SUCCESS_HOVER, hover_color=SUCCESS, text_color=TEXT_ON_ACCENT, height=28, corner_radius=6).grid(row=1, column=5, sticky="e", padx=(4, 10), pady=(4, 10))

        self.monitor_cards_frame = ttk.Frame(self.monitor_tab, padding=(8, 0, 8, 6))
        self.monitor_cards_frame.grid(row=1, column=0, sticky="ew")
        for index in range(5):
            self.monitor_cards_frame.columnconfigure(index, weight=1)
        self.monitor_cards = {
            "total": CompactStatCard(self.monitor_cards_frame, "Total", background="#0f172a", foreground="#f8fafc"),
            "accept": CompactStatCard(self.monitor_cards_frame, "Accept", background="#166534", foreground="#f0fdf4"),
            "reject": CompactStatCard(self.monitor_cards_frame, "Reject", background="#991b1b", foreground="#fef2f2"),
            "reject_rate": CompactStatCard(self.monitor_cards_frame, "Reject Rate", background="#7c2d12", foreground="#fff7ed"),
            "push": CompactStatCard(self.monitor_cards_frame, "Push Issues", background="#334155", foreground="#f8fafc"),
        }
        self._layout_monitor_cards(compact=False)

        body = ttk.Frame(self.monitor_tab, padding=8)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self.monitor_left = ttk.Frame(body)
        self.monitor_right = ttk.Frame(body)
        self.monitor_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.monitor_right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        recent = ctk.CTkFrame(self.monitor_left, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        recent.pack(fill="both", expand=True)
        recent.columnconfigure(0, weight=1)
        recent.rowconfigure(2, weight=1)
        ctk.CTkLabel(recent, text="Recent Inspections", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(recent, textvariable=self.monitor_context_var, text_color=TEXT_SECONDARY).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))
        self.results_table = self._build_table(
            recent,
            [
                ("id", "ID", 60, "center"),
                ("time", "Time", 145, "w"),
                ("decision", "Decision", 90, "center"),
                ("part", "Part", 150, "w"),
                ("line", "Line", 90, "w"),
                ("station", "Station", 90, "w"),
                ("push", "Push", 90, "center"),
                ("reason", "Reason", 130, "w"),
            ],
            row=2,
            height=16,
        )
        actions = self._build_action_row(recent, [("Retry Failed Visible", self.retry_visible_failed_pushes, "neutral", "left")])
        actions.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 10))

        side = ctk.CTkFrame(self.monitor_right, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        side.pack(fill="both", expand=True)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)
        ctk.CTkLabel(side, text="Recent Rejects", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 8))
        self.reject_table = self._build_table(
            side,
            [
                ("id", "ID", 60, "center"),
                ("part", "Part", 130, "w"),
                ("reason", "Reason", 150, "w"),
                ("time", "Time", 140, "w"),
            ],
            row=1,
            height=12,
        )
        self.monitor_summary = LabeledValuePanel(
            side,
            "Selected Summary",
            [
                ("decision", "Decision"),
                ("reason", "Reason"),
                ("push_status", "Push"),
                ("line_id", "Line"),
                ("station_id", "Station"),
            ],
            columns=1,
        )
        self.monitor_summary.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 10))
        self.results_table.bind("<<TreeviewSelect>>", self.open_monitor_result)

    # ------------------------------------------------------------------
    # Refresh and render
    def refresh_all(self) -> None:
        self.refresh_presets()
        self.refresh_template_options()
        self.refresh_model_options()
        self.refresh_operators()
        self.refresh_monitor()

    def refresh_presets(self) -> None:
        self._set_status("Loading presets...")

        def _done(items, error):
            if error:
                self._set_status(f"Preset load error: {error}")
                return
            self._deployments_cache = list(items or [])
            self._render_presets()
            self._update_overview_cards()
            self._record_refresh("Presets")
            self._set_status(f"Loaded {len(self._deployments_cache)} active presets.")

        run_async(self, self.api.list_deployments, callback=_done)

    def refresh_template_options(self) -> None:
        def _done(items, error):
            if error:
                return
            self._templates_cache = list(items or [])

        run_async(self, self.api.list_templates, callback=_done)

    def refresh_model_options(self) -> None:
        def _done(items, error):
            if error:
                self._model_lookup = {}
                self.preset_model_selector.configure(values=[])
                return
            self._models_cache = list(items or [])
            self._model_lookup = {}
            values: list[str] = []
            for item in self._models_cache:
                label = f"{item.get('id')} | {item.get('name') or item.get('path') or 'model'}"
                path = str(item.get("path") or "").strip()
                if path:
                    label = f"{label} | {path}"
                values.append(label)
                self._model_lookup[label] = dict(item)
            self.preset_model_selector.configure(values=values)
            if values and not self.preset_model_choice_var.get().strip():
                self.preset_model_choice_var.set(values[0])
                self._on_preset_model_selected()

        run_async(self, self.api.list_models, callback=_done)

    def refresh_operators(self) -> None:
        self._set_status("Loading operators...")

        def _done(items, error):
            if error:
                self._set_status(f"Operator load error: {error}")
                return
            self._users_cache = list(items or [])
            self._render_operators()
            self._update_overview_cards()
            self._record_refresh("Operators")

        run_async(self, self.api.list_users, callback=_done)

    def refresh_monitor(self) -> None:
        params = self._monitor_filters()
        self._set_status("Loading monitor...")

        def _load():
            return {
                "summary": self.api.dashboard_summary(params),
                "results": self.api.list_inspections(params),
            }

        def _done(result, error):
            if error:
                self._set_status(f"Monitor load error: {error}")
                return
            payload = result or {}
            self._results_cache = list(payload.get("results") or [])
            self._render_monitor_results()
            self._render_monitor_summary(payload.get("summary") or {})
            self._update_overview_cards()
            self._record_refresh("Monitor")
            self._set_status(f"Loaded {len(self._results_cache)} recent inspections.")

        run_async(self, _load, callback=_done)

    def _render_presets(self) -> None:
        self._clear_tree(self.preset_table)
        active_items = [item for item in self._deployments_cache if bool(item.get("is_active", True))]
        if not active_items:
            self.preset_table.insert("", "end", iid="__empty__", values=("-", "No active presets.", "", "", "", ""))
            return
        for item in active_items:
            self.preset_table.insert(
                "",
                "end",
                iid=str(item.get("id")),
                values=(
                    item.get("id"),
                    _safe_text(item.get("line_id")),
                    _safe_text(item.get("station_id")),
                    _safe_text(item.get("template_name")),
                    _safe_text(item.get("template_version_id")),
                    _format_status(item.get("is_active", True)),
                ),
            )

    def _render_operators(self) -> None:
        self._clear_tree(self.users_table)
        operators = [item for item in self._users_cache if str(item.get("role") or "").strip().lower() == "operator"]
        if not operators:
            self.users_table.insert("", "end", iid="__empty__", values=("-", "No operators.", "", "", ""))
            return
        for item in operators:
            rfid_status = "Bound" if item.get("rfid_bound") else "Unbound"
            if item.get("rfid_uid_last4"):
                rfid_status = f"*{_safe_text(item.get('rfid_uid_last4'))}"
            self.users_table.insert(
                "",
                "end",
                iid=str(item.get("id")),
                values=(
                    item.get("id"),
                    _safe_text(item.get("username")),
                    _safe_text(item.get("role")),
                    _format_status(item.get("is_active", True)),
                    rfid_status,
                ),
            )

    def _render_monitor_results(self) -> None:
        self._clear_tree(self.results_table)
        self._clear_tree(self.reject_table)
        if not self._results_cache:
            self.results_table.insert("", "end", iid="__empty__", values=("-", "No inspection data.", "", "", "", "", "", ""))
            self.monitor_context_var.set("No inspections found for current filter.")
            return
        for item in self._results_cache:
            decision = str(item.get("decision") or item.get("decision_code") or "").strip().upper()
            reason = item.get("reject_reason_code") or ("OK" if decision == "ACCEPT" else "-")
            self.results_table.insert(
                "",
                "end",
                iid=str(item.get("id")),
                values=(
                    item.get("id"),
                    _format_timestamp(item.get("inspected_at")),
                    _safe_text(decision),
                    _safe_text(item.get("part_name")),
                    _safe_text(item.get("line_id")),
                    _safe_text(item.get("station_id")),
                    _safe_text(item.get("push_status")),
                    _safe_text(reason),
                ),
            )
            if decision == "REJECT":
                self.reject_table.insert(
                    "",
                    "end",
                    iid=str(item.get("id")),
                    values=(
                        item.get("id"),
                        _safe_text(item.get("part_name")),
                        _safe_text(reason),
                        _format_timestamp(item.get("inspected_at")),
                    ),
                )
        self.monitor_context_var.set(f"Showing {len(self._results_cache)} recent inspections.")

    def _render_monitor_summary(self, summary: dict) -> None:
        total = int(summary.get("total") or summary.get("total_count") or summary.get("total_inspections") or len(self._results_cache) or 0)
        accept = int(summary.get("accept") or summary.get("accept_count") or summary.get("total_accept") or sum(1 for item in self._results_cache if str(item.get("decision") or "").upper() == "ACCEPT"))
        reject = int(summary.get("reject") or summary.get("reject_count") or summary.get("total_reject") or sum(1 for item in self._results_cache if str(item.get("decision") or "").upper() == "REJECT"))
        reject_rate = (reject / total * 100.0) if total else 0.0
        push_issues = sum(1 for item in self._results_cache if str(item.get("push_status") or "").lower() in {"failed", "pending"})
        self.monitor_cards["total"].set_value(total)
        self.monitor_cards["accept"].set_value(accept)
        self.monitor_cards["reject"].set_value(reject)
        self.monitor_cards["reject_rate"].set_value(f"{reject_rate:.1f}%")
        self.monitor_cards["push"].set_value(push_issues, "failed/pending")
        self.admin_cards["accept"].set_value(accept)
        self.admin_cards["reject"].set_value(reject)

    # ------------------------------------------------------------------
    # Preset behavior
    def reset_preset_wizard(self) -> None:
        self.current_template_id = None
        self.current_template_version_id = None
        self._editing_deployment_id = None
        self.preset_name_var.set("")
        self.preset_description_var.set("")
        self.preset_line_var.set("")
        self.preset_station_var.set("")
        self.preset_expected_code_var.set("")
        self.part_ready_roi_x_var.set("0.2")
        self.part_ready_roi_y_var.set("0.2")
        self.part_ready_roi_w_var.set("0.25")
        self.part_ready_roi_h_var.set("0.25")
        self.sticker_roi_x_var.set("0.2")
        self.sticker_roi_y_var.set("0.2")
        self.sticker_roi_w_var.set("0.6")
        self.sticker_roi_h_var.set("0.6")
        self._set_status("Preset wizard reset.")

    def _on_preset_selected(self, _event=None) -> None:
        deployment_id = self._selected_treeview_id(self.preset_table)
        if deployment_id is None:
            return
        deployment = next((item for item in self._deployments_cache if int(item.get("id") or 0) == deployment_id), None)
        if not deployment:
            return
        self._editing_deployment_id = deployment_id
        self.preset_line_var.set(str(deployment.get("line_id") or ""))
        self.preset_station_var.set(str(deployment.get("station_id") or ""))

        version_id = int(deployment.get("template_version_id") or 0)
        template_id = int(deployment.get("template_id") or 0)
        if version_id:
            try:
                detail = self.api.get_template_version(version_id)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Preset detail load failed: {exc}")
                return
        elif template_id:
            try:
                detail = self.api.get_template(template_id)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Preset detail load failed: {exc}")
                return
        else:
            return
        self._apply_preset_detail(detail, deployment=deployment)
        self._set_status(f"Loaded preset {_safe_text(deployment.get('template_name'))}.")

    def _apply_preset_detail(self, detail: dict, *, deployment: dict | None = None) -> None:
        self.current_template_id = int(detail.get("id") or (deployment or {}).get("template_id") or 0) or None
        self.current_template_version_id = int(detail.get("version_id") or (deployment or {}).get("template_version_id") or 0) or None
        self.preset_name_var.set(str(detail.get("name") or (deployment or {}).get("template_name") or ""))
        self.preset_description_var.set(str(detail.get("description") or ""))
        if deployment:
            self.preset_line_var.set(str(deployment.get("line_id") or ""))
            self.preset_station_var.set(str(deployment.get("station_id") or ""))
        sticker = detail.get("sticker") or {}
        self.preset_expected_code_var.set(str(sticker.get("ocr_expected_text") or sticker.get("expected_class") or ""))
        part_ready_roi = detail.get("part_ready_roi") or {}
        sticker_roi = detail.get("sticker_roi") or detail.get("roi") or {}
        self.part_ready_roi_x_var.set(str(part_ready_roi.get("x", 0.2)))
        self.part_ready_roi_y_var.set(str(part_ready_roi.get("y", 0.2)))
        self.part_ready_roi_w_var.set(str(part_ready_roi.get("w", 0.25)))
        self.part_ready_roi_h_var.set(str(part_ready_roi.get("h", 0.25)))
        self.sticker_roi_x_var.set(str(sticker_roi.get("x", 0.2)))
        self.sticker_roi_y_var.set(str(sticker_roi.get("y", 0.2)))
        self.sticker_roi_w_var.set(str(sticker_roi.get("w", 0.6)))
        self.sticker_roi_h_var.set(str(sticker_roi.get("h", 0.6)))

        vision = detail.get("vision") or {}
        model_path = str(vision.get("model_path") or "")
        self.preset_model_path_var.set(model_path)
        self.preset_model_meta_path_var.set(str(vision.get("model_meta_path") or ""))
        self._select_model_label_for_path(model_path)

    def _on_preset_model_selected(self, _event=None) -> None:
        item = self._model_lookup.get(self.preset_model_choice_var.get().strip())
        if not item:
            return
        self.preset_model_path_var.set(str(item.get("path") or ""))
        self.preset_model_meta_path_var.set(str(item.get("meta_path") or ""))

    def _select_model_label_for_path(self, model_path: str) -> None:
        normalized = str(model_path or "").strip().lower()
        if not normalized:
            return
        for label, item in self._model_lookup.items():
            if str(item.get("path") or "").strip().lower() == normalized:
                self.preset_model_choice_var.set(label)
                return

    def save_and_deploy_preset(self) -> None:
        try:
            payload = self._preset_payload()
        except ValueError as exc:
            messagebox.showerror("Preset", str(exc))
            return
        try:
            if self.current_template_id:
                saved = self.api.update_template(self.current_template_id, payload)
            else:
                saved = self.api.create_template(payload)
            template_id = int(saved.get("id") or self.current_template_id or 0)
            version_id = int(saved.get("version_id") or saved.get("current_version_id") or 0)
            if not template_id or not version_id:
                raise ValueError("Saved preset did not return template id and version id.")
            deployment = self.api.deploy_template(
                {
                    "template_id": template_id,
                    "template_version_id": version_id,
                    "line_id": self.preset_line_var.get().strip(),
                    "station_id": self.preset_station_var.get().strip(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Preset", str(exc))
            return
        self.current_template_id = template_id
        self.current_template_version_id = version_id
        self._editing_deployment_id = int(deployment.get("id") or 0) or self._editing_deployment_id
        self.refresh_presets()
        self._set_status(f"Preset deployed to {self.preset_line_var.get().strip()}/{self.preset_station_var.get().strip()}.")
        messagebox.showinfo("Preset", "Preset saved and deployed.")

    def _preset_payload(self) -> dict:
        name = self.preset_name_var.get().strip()
        line = self.preset_line_var.get().strip()
        station = self.preset_station_var.get().strip()
        expected_code = self.preset_expected_code_var.get().strip()
        model_path = self.preset_model_path_var.get().strip()
        if not name:
            raise ValueError("Preset name is required.")
        if not line or not station:
            raise ValueError("Line and station are required.")
        if not model_path:
            raise ValueError("Model is required.")
        if not expected_code:
            raise ValueError("Sticker code is required.")

        return {
            "id": self.current_template_id,
            "version_id": self.current_template_version_id,
            "version_number": 1,
            "name": name,
            "description": self.preset_description_var.get().strip(),
            "is_active": True,
            "camera": {"camera_index": 0, "width": None, "height": None, "fps": None},
            "part_ready_roi": {
                "x": _float_or_default(self.part_ready_roi_x_var.get(), 0.2),
                "y": _float_or_default(self.part_ready_roi_y_var.get(), 0.2),
                "w": _float_or_default(self.part_ready_roi_w_var.get(), 0.25),
                "h": _float_or_default(self.part_ready_roi_h_var.get(), 0.25),
            },
            "sticker_roi": {
                "x": _float_or_default(self.sticker_roi_x_var.get(), 0.2),
                "y": _float_or_default(self.sticker_roi_y_var.get(), 0.2),
                "w": _float_or_default(self.sticker_roi_w_var.get(), 0.6),
                "h": _float_or_default(self.sticker_roi_h_var.get(), 0.6),
            },
            "vision": {
                "model_path": model_path,
                "model_meta_path": self.preset_model_meta_path_var.get().strip() or None,
                "runtime": "ultralytics",
                "conf_threshold": 0.25,
                "stream_fps": 10.0,
                "inference_fps": 4.0,
                "imgsz": 640,
                "classes": [expected_code],
                "enable_ergonomic_check": False,
                "ergonomic_pose_model_path": None,
                "ergonomic_min_keypoint_conf": 0.35,
                "ocr_engine": "default",
                "ocr_language": "eng",
                "ocr_psm": 7,
                "ocr_allowlist": "",
                "text_anchor_class": "text_anchor",
                "center_dot_class": "center_dot",
                "anchor_crop_padding_ratio": 0.08,
                "anchor_crop_scale": 2.0,
            },
            "part_ready": {
                "enabled": True,
                "color_profile_id": None,
                "colorspace": "LAB",
                "distance_threshold": None,
                "min_match_ratio": 0.75,
            },
            "sticker": {
                "part_name": expected_code,
                "expected_class": expected_code,
                "line": line,
                "station": station,
                "enabled": True,
                "validator_mode": "ml_detection",
                "min_roi_confidence": 0.0,
                "min_class_confidence": None,
                "max_offset_x": 80,
                "max_offset_y": 80,
                "expected_center_x": None,
                "expected_center_y": None,
                "expected_tilt_degrees": 0.0,
                "max_tilt_degrees": None,
                "ocr_mode": None,
                "ocr_expected_text": expected_code,
                "ocr_min_confidence": None,
                "ocr_regex": None,
                "ocr_canonical_map": {},
                "anchor_min_confidence": None,
                "dot_min_confidence": None,
                "expected_dot_x": None,
                "expected_dot_y": None,
                "max_anchor_offset_x": None,
                "max_anchor_offset_y": None,
                "tilt_gate_enabled": False,
                "commit_stable_frames": 1,
                "part_ready_settle_ms": None,
            },
            "persistence": {"write_to_db": True},
            "metadata": {"preset_ui": "admin_simple"},
        }

    def deactivate_selected_preset(self) -> None:
        deployment_id = self._selected_treeview_id(self.preset_table)
        if deployment_id is None:
            return
        if not self._confirm_action("Deactivate Preset", f"Deactivate preset deployment #{deployment_id}?"):
            return
        try:
            self.api.deactivate_deployment(deployment_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Preset", str(exc))
            return
        self.refresh_presets()
        self._set_status(f"Preset deployment #{deployment_id} deactivated.")

    # ------------------------------------------------------------------
    # Operator behavior
    def create_operator_from_rfid(self) -> None:
        username = self.operator_username_var.get().strip()
        rfid_uid = self.operator_rfid_var.get().strip()
        if not username:
            messagebox.showerror("Operators", "Username is required.")
            return
        if not rfid_uid:
            messagebox.showerror("Operators", "Scan RFID first.")
            return
        try:
            created = self.api.create_user(
                {
                    "username": username,
                    "password": _random_password(),
                    "role": "operator",
                }
            )
            user_id = int(created.get("id") or 0)
            if user_id <= 0:
                raise ValueError("User API did not return a valid id.")
            self.api.bind_user_rfid(user_id, rfid_uid)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Operators", str(exc))
            return
        self.operator_username_var.set("")
        self.operator_rfid_var.set("")
        self.refresh_operators()
        self._set_status(f"Operator {username} created and RFID bound.")

    # ------------------------------------------------------------------
    # Monitor behavior
    def _monitor_filters(self) -> dict[str, object]:
        params: dict[str, object] = {"limit": 100}
        if self.monitor_line_var.get().strip():
            params["line_id"] = self.monitor_line_var.get().strip()
        if self.monitor_station_var.get().strip():
            params["station_id"] = self.monitor_station_var.get().strip()
        return params

    def export_monitor_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile="inspections.csv",
            title="Export Inspection Results",
        )
        if not path:
            return
        try:
            csv_text = self.api.export_inspections_csv(self._monitor_filters())
            with open(path, "w", encoding="utf-8", newline="") as file_handle:
                file_handle.write(csv_text)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export CSV", str(exc))
            return
        self._set_status(f"CSV export saved to {path}.")

    def retry_visible_failed_pushes(self) -> None:
        retry_ids = [
            int(item.get("id") or 0)
            for item in self._results_cache
            if str(item.get("push_status") or "").lower() in {"failed", "pending"} and int(item.get("id") or 0) > 0
        ]
        if not retry_ids:
            messagebox.showinfo("Monitor", "No failed or pending pushes are visible.")
            return
        try:
            result = self.api.retry_failed_inspection_pushes(result_ids=retry_ids, limit=len(retry_ids))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Monitor", str(exc))
            return
        self.refresh_monitor()
        self._set_status(
            f"Retry attempted={result.get('attempted', len(retry_ids))}, succeeded={result.get('succeeded', 0)}."
        )

    def open_monitor_result(self, _event=None) -> None:
        result_id = self._selected_treeview_id(self.results_table)
        if result_id is None:
            return
        item = next((entry for entry in self._results_cache if int(entry.get("id") or 0) == result_id), None)
        if not item:
            return
        decision = str(item.get("decision") or item.get("decision_code") or "").strip().upper()
        self.monitor_summary.set_values(
            {
                "decision": decision,
                "reason": item.get("reject_reason_code") or ("OK" if decision == "ACCEPT" else "-"),
                "push_status": item.get("push_status"),
                "line_id": item.get("line_id"),
                "station_id": item.get("station_id"),
            }
        )

    # ------------------------------------------------------------------
    # Shared helpers
    def _entry(self, master, row: int, column: int, label: str, variable: tk.StringVar, *, columnspan: int = 1, columns: int = 4) -> ttk.Entry:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(12, 8), pady=5)
        entry = ttk.Entry(master, textvariable=variable)
        entry.grid(row=row, column=column + 1, columnspan=columnspan, sticky="ew", padx=(0, 12), pady=5)
        if columns:
            for index in range(columns):
                master.columnconfigure(index, weight=1 if index % 2 else 0)
        return entry

    def _roi_entries(self, master, row: int, x_var: tk.StringVar, y_var: tk.StringVar, w_var: tk.StringVar, h_var: tk.StringVar) -> None:
        labels = (("x", x_var), ("y", y_var), ("w", w_var), ("h", h_var))
        for offset, (label, variable) in enumerate(labels):
            column = offset * 2
            ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(12 if offset == 0 else 8, 4), pady=5)
            ttk.Entry(master, textvariable=variable, width=8).grid(row=row, column=column + 1, sticky="ew", padx=(0, 8), pady=5)

    def _build_table(self, master, columns: list[tuple[str, str, int, str]], *, row: int | None = None, height: int = 14) -> ttk.Treeview:
        shell = ttk.Frame(master)
        if row is None:
            shell.pack(fill="both", expand=True)
        else:
            shell.grid(row=row, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        column_names = [item[0] for item in columns]
        tree = ttk.Treeview(shell, columns=column_names, show="headings", height=height, selectmode="browse")
        for name, heading, width, anchor in columns:
            tree.heading(name, text=heading)
            tree.column(name, width=width, minwidth=60, stretch=True, anchor=anchor)
        y_scroll = AutoHideScrollbar(shell, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        return tree

    def _build_action_row(self, master, buttons: list[tuple]) -> ctk.CTkFrame:
        row = ctk.CTkFrame(master, fg_color="transparent", corner_radius=0)
        left = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
        right = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
        left.pack(side="left")
        right.pack(side="right")
        for label, command, tone, side in buttons:
            target = left if side == "left" else right
            color = ACCENT if tone == "primary" else PANEL_ALT_BG
            hover = ACCENT_HOVER if tone == "primary" else BORDER
            ctk.CTkButton(
                target,
                text=label,
                command=command,
                fg_color=color,
                hover_color=hover,
                text_color=TEXT_ON_ACCENT if tone == "primary" else TEXT_PRIMARY,
                height=28,
                corner_radius=6,
            ).pack(side="left", padx=(0, 6))
        return row

    def _layout_overview_cards(self, *, compact: bool) -> None:
        try:
            if not self.winfo_exists() or not self.overview_cards_frame.winfo_exists():
                return
            slaves = self.overview_cards_frame.grid_slaves()
        except tk.TclError:
            return
        for widget in slaves:
            widget.grid_forget()
        columns = 2 if compact else 4
        for column in range(columns):
            self.overview_cards_frame.columnconfigure(column, weight=1)
        for index, key in enumerate(("presets", "operators", "accept", "reject")):
            self.admin_cards[key].grid(row=index // columns, column=index % columns, sticky="ew", padx=4, pady=4)

    def _layout_monitor_cards(self, *, compact: bool) -> None:
        try:
            if not self.winfo_exists() or not self.monitor_cards_frame.winfo_exists():
                return
            slaves = self.monitor_cards_frame.grid_slaves()
        except tk.TclError:
            return
        for widget in slaves:
            widget.grid_forget()
        columns = 2 if compact else 5
        for column in range(columns):
            self.monitor_cards_frame.columnconfigure(column, weight=1)
        for index, key in enumerate(("total", "accept", "reject", "reject_rate", "push")):
            self.monitor_cards[key].grid(row=index // columns, column=index % columns, sticky="ew", padx=4, pady=4)

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _selected_treeview_id(self, tree: ttk.Treeview) -> int | None:
        selection = tree.selection()
        if not selection:
            focus = tree.focus()
            selection = (focus,) if focus else ()
        if not selection:
            return None
        raw = str(selection[0])
        if raw.startswith("__"):
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _confirm_action(self, title: str, message: str) -> bool:
        return bool(messagebox.askyesno(title, message))

    def _record_refresh(self, key: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._last_refresh[key] = timestamp
        self.refresh_time_var.set(f"Last {key}: {timestamp}")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _update_overview_cards(self) -> None:
        active_presets = sum(1 for item in self._deployments_cache if bool(item.get("is_active", True)))
        operators = sum(1 for item in self._users_cache if str(item.get("role") or "").lower() == "operator")
        self.admin_cards["presets"].set_value(active_presets, "active")
        self.admin_cards["operators"].set_value(operators, "RFID users")

    def _on_resize(self, _event=None) -> None:
        self.after_idle(self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        try:
            if not self.winfo_exists():
                return
            width = max(self.winfo_width(), self.winfo_toplevel().winfo_width())
            height = self.winfo_height()
        except tk.TclError:
            return
        compact = width < RESPONSIVE_BREAKPOINT
        if compact != self._layout_compact:
            self._layout_compact = compact
            self._layout_overview_cards(compact=compact)
            self._layout_monitor_cards(compact=compact)

            for left, right in (
                (self.presets_left, self.presets_right),
                (self.operators_left, self.operators_right),
            ):
                left.grid_forget()
                right.grid_forget()
                if compact:
                    left.grid(row=0, column=0, columnspan=2, sticky="nsew")
                    right.grid(row=1, column=0, columnspan=2, sticky="nsew")
                else:
                    left.grid(row=0, column=0, sticky="nsew")
                    right.grid(row=0, column=1, sticky="nsew")

        self._overview_cards_visible = height >= 760
        try:
            if self._overview_cards_visible:
                self.overview_cards_frame.grid()
            else:
                self.overview_cards_frame.grid_remove()
        except tk.TclError:
            return

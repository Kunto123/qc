from __future__ import annotations

import platform
import threading
import tkinter as tk
import uuid
from tkinter import messagebox, ttk

import customtkinter as ctk
import cv2

from client_tk.app.components.async_bridge import run_async
from client_tk.app.components.counter_panel import BREAKDOWN_ORDER, CounterPanel
from client_tk.app.components.live_view import LiveView
from client_tk.app.components.result_panel import ResultPanel
from client_tk.app.components.scrollable_frame import ScrollableFrame
from client_tk.app.config import DEFAULT_UPLOAD_INTERVAL_MS
from client_tk.app.services.camera_capture import CameraCaptureService
from client_tk.app.services.frame_upload import FrameUploadService
from client_tk.app.theme import APP_BG, ACCENT_SOFT, BORDER, PANEL_ALT_BG, PANEL_BG, SHELL_BG, TEXT_PRIMARY, TEXT_SECONDARY, ACCENT, ACCENT_HOVER, TEXT_ON_ACCENT, SUCCESS, SUCCESS_HOVER


BADGE_COLORS = {
    "neutral": ("#475569", "#f8fafc"),
    "success": ("#166534", "#f0fdf4"),
    "danger": ("#991b1b", "#fef2f2"),
    "warning": ("#b45309", "#fffbeb"),
    "info": ("#1d4ed8", "#eff6ff"),
}
RESPONSIVE_BREAKPOINT = 1240
HEARTBEAT_INTERVAL_MS = 20_000
PLC_POLL_INTERVAL_MS = 8_000
AUTO_START_FRAME_WAIT_MS = 150
AUTO_START_MAX_ATTEMPTS = 40


class OperatorScreen(ctk.CTkFrame):
    def __init__(self, master, api_client, session_state):
        super().__init__(master, fg_color=APP_BG, corner_radius=0)
        self.api = api_client
        self.state = session_state
        self.capture = CameraCaptureService()
        self.uploader = FrameUploadService()  # Local frame pump via API client bridge.
        self._latest_payload: dict | None = None
        self._latest_error: str | None = None
        self._lock = threading.Lock()
        self._after_id: str | None = None
        self._heartbeat_after_id: str | None = None
        self._plc_poll_after_id: str | None = None
        self._closed = False
        self._machine_id = f"{platform.node() or 'workstation'}-{uuid.getnode():012x}"
        self._settings_window: tk.Toplevel | None = None
        self._plc_release_window: tk.Toplevel | None = None
        self._template_lookup: dict[str, dict] = {}
        self._template_detail_lookup: dict[int, dict] = {}
        self._template_version_detail_lookup: dict[int, dict] = {}
        self._is_compact_layout: bool | None = None
        self._is_preview_compact: bool | None = None
        self._auth_error_notified = False
        self._auto_start_done = False
        self._auto_start_after_id: str | None = None

        self.line_value = tk.StringVar()
        self.station_value = tk.StringVar()
        self.camera_value = tk.StringVar(value="0")
        self.template_version_value = tk.StringVar()
        self.part_ready_roi_x_value = tk.StringVar(value="0.2")
        self.part_ready_roi_y_value = tk.StringVar(value="0.2")
        self.part_ready_roi_w_value = tk.StringVar(value="0.25")
        self.part_ready_roi_h_value = tk.StringVar(value="0.25")
        self.sticker_roi_x_value = tk.StringVar(value="0.2")
        self.sticker_roi_y_value = tk.StringVar(value="0.2")
        self.sticker_roi_w_value = tk.StringVar(value="0.6")
        self.sticker_roi_h_value = tk.StringVar(value="0.6")
        self.template_choice = tk.StringVar()
        self.display_source = tk.StringVar(value="Right View: Live Camera + Sticker ROI")

        self.operator_context = tk.StringVar(value=f"Operator: {self.state.user.get('username') if self.state.user else '-'}")
        self.line_context = tk.StringVar(value="Line: -")
        self.station_context = tk.StringVar(value="Station: -")
        self.template_context = tk.StringVar(value="Template: -")
        self.info_var = tk.StringVar(value="Idle. Pilih template atau deployment, lalu start camera.")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self._build_top_bar()
        self._build_context_bar()
        self._build_status_strip()
        self._build_content()
        self._load_template_choices()
        self._refresh_context_summary()
        self._update_status_badges()
        self.bind("<Configure>", self._on_resize)
        self.after_idle(self._apply_responsive_layout)
        self._schedule_poll()
        self._schedule_heartbeat(delay_ms=1_000)
        self._schedule_plc_poll(delay_ms=4_000)
        self.after_idle(self._auto_start_first_template)

    def _build_top_bar(self) -> None:
        self.top_bar = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        self.top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.top_bar.columnconfigure(0, weight=1)

        self.action_bar = ctk.CTkFrame(self.top_bar, fg_color=APP_BG, corner_radius=0)
        self.action_bar.grid(row=0, column=0, sticky="ew")
        self.action_buttons = [
            ctk.CTkButton(self.action_bar, text="\u2699 Settings", command=self._open_settings, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT),
            ctk.CTkButton(self.action_bar, text="Load Deployment", command=self._load_deployment, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT),
            ctk.CTkButton(self.action_bar, text="Start Camera", command=self._start_camera, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT),
            ctk.CTkButton(self.action_bar, text="Stop Camera", command=self._stop_camera, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT),
            ctk.CTkButton(self.action_bar, text="Start Session", command=self._start_session, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT),
            ctk.CTkButton(self.action_bar, text="Stop Session", command=self._stop_session, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT),
        ]

        self.template_box = ctk.CTkFrame(self.top_bar, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self.template_box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.template_box, text="Template", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 6))
        self.template_selector = ttk.Combobox(self.template_box, textvariable=self.template_choice, width=32, state="readonly")
        self.template_selector.grid(row=1, column=0, padx=(10, 6), pady=(0, 10), sticky="ew")
        self.template_selector.bind("<<ComboboxSelected>>", self._on_template_selected)
        ctk.CTkButton(self.template_box, text="Refresh", command=self._load_template_choices, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT).grid(row=1, column=1, padx=(0, 10), pady=(0, 10))

        self._layout_top_bar(compact=False)

    def _build_context_bar(self) -> None:
        self.context_bar = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self.context_bar.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.context_labels = {
            "operator": ctk.CTkLabel(self.context_bar, textvariable=self.operator_context, font=("Segoe UI", 13, "bold"), text_color=TEXT_PRIMARY),
            "line": ctk.CTkLabel(self.context_bar, textvariable=self.line_context, font=("Segoe UI", 11), text_color=TEXT_SECONDARY),
            "station": ctk.CTkLabel(self.context_bar, textvariable=self.station_context, font=("Segoe UI", 11), text_color=TEXT_SECONDARY),
            "template": ctk.CTkLabel(self.context_bar, textvariable=self.template_context, font=("Segoe UI", 11), text_color=TEXT_SECONDARY),
        }
        self._layout_context_bar(compact=False)

    def _build_status_strip(self) -> None:
        self.status_frame = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self.status_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(self.status_frame, text="System Status", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w",
            padx=10,
            pady=(10, 6),
        )
        self.status_badges_container = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        self.status_badges_container.pack(fill="x", padx=10, pady=(0, 10))
        self.badges: dict[str, ctk.CTkLabel] = {}
        for key in ("SERVER", "CAMERA", "SESSION", "DB", "EVENT", "PLC"):
            label = ctk.CTkLabel(
                self.status_badges_container,
                text=f"{key}: -",
                fg_color=BADGE_COLORS["neutral"][0],
                text_color=BADGE_COLORS["neutral"][1],
                font=("Segoe UI", 10, "bold"),
                corner_radius=999,
                height=28,
                padx=12,
            )
            self.badges[key] = label
        self._layout_status_strip(compact=False)

    def _build_content(self) -> None:
        self.content_scroller = ScrollableFrame(self)
        self.content_scroller.grid(row=3, column=0, sticky="nsew")
        self.content = self.content_scroller.body
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=0)

        self.live_container = ttk.Frame(self.content)
        self.live_container.columnconfigure(0, weight=1)
        self.live_container.rowconfigure(0, weight=1)
        self.live_container.rowconfigure(1, weight=0)

        self.preview_strip = ttk.Frame(self.live_container)
        self.preview_strip.grid(row=0, column=0, sticky="nsew")
        self.preview_strip.columnconfigure(0, weight=2)
        self.preview_strip.columnconfigure(1, weight=3)
        self.preview_strip.rowconfigure(0, weight=1)

        self.part_ready_preview = LiveView(self.preview_strip, "Part Ready ROI", size=(420, 560))
        self.main_view = LiveView(self.preview_strip, "Sticker ROI / ML Overlay", size=(900, 560))

        live_footer = ttk.Frame(self.live_container)
        live_footer.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        live_footer.columnconfigure(0, weight=1)
        ttk.Label(live_footer, textvariable=self.display_source, foreground="#475569").grid(row=0, column=0, sticky="w")
        ttk.Label(live_footer, textvariable=self.info_var, foreground="#475569", wraplength=700, justify="left").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )

        self.sidebar_container = ttk.Frame(self.content, width=360)
        self.sidebar_container.grid_propagate(False)
        self.sidebar_container.columnconfigure(0, weight=1)
        self.sidebar_container.rowconfigure(0, weight=0)  # counter: fixed
        self.sidebar_container.rowconfigure(1, weight=0)  # decision banner: fixed
        self.sidebar_container.rowconfigure(2, weight=1)  # scroll: fills rest

        # Fixed counter strip — always visible, outside scroll area
        self.counter_panel = CounterPanel(self.sidebar_container)
        self.counter_panel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        # Decision status banner — fixed below counter, outside scroll area
        self.decision_status_frame = ctk.CTkFrame(
            self.sidebar_container, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER
        )
        self.decision_status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.decision_banner = ctk.CTkLabel(
            self.decision_status_frame,
            text="WAITING",
            fg_color="#334155",
            text_color="#f8fafc",
            font=("Segoe UI", 24, "bold"),
            corner_radius=14,
            anchor="center",
        )
        self.decision_banner.pack(fill="x", padx=12, pady=(12, 6))
        self.decision_subtitle = ctk.CTkLabel(
            self.decision_status_frame,
            text="Menunggu event inspeksi pertama.",
            font=("Segoe UI", 10),
            wraplength=320,
            justify="left",
            text_color=TEXT_SECONDARY,
        )
        self.decision_subtitle.pack(anchor="w", padx=12, pady=(0, 6))

        self.stiker_terpasang_btn = ctk.CTkButton(
            self.decision_status_frame,
            text="✓ Stiker Terpasang",
            command=self._on_stiker_terpasang,
            fg_color=SUCCESS,
            hover_color=SUCCESS_HOVER,
            text_color=TEXT_ON_ACCENT,
            font=("Segoe UI", 11, "bold"),
            corner_radius=10,
            height=34,
            state="disabled",
        )
        self.stiker_terpasang_btn.pack(fill="x", padx=12, pady=(0, 4))

        self.plc_release_btn = ctk.CTkButton(
            self.decision_status_frame,
            text="Release Clamp",
            command=self._open_plc_release_dialog,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            text_color="#fef2f2",
            font=("Segoe UI", 11, "bold"),
            corner_radius=10,
            height=34,
        )
        self.plc_release_btn.pack(fill="x", padx=12, pady=(0, 4))
        self.plc_status_label = ctk.CTkLabel(
            self.decision_status_frame,
            text="",
            font=("Segoe UI", 9),
            text_color=TEXT_SECONDARY,
            wraplength=320,
            justify="left",
        )
        self.plc_status_label.pack(anchor="w", padx=12, pady=(0, 10))

        self.sidebar_canvas = tk.Canvas(
            self.sidebar_container,
            highlightthickness=0,
            bg=self._resolve_canvas_background(),
        )
        self.sidebar_scrollbar = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_canvas.grid(row=2, column=0, sticky="nsew")
        self.sidebar_scrollbar.grid(row=2, column=1, sticky="ns")

        self.sidebar_inner = ttk.Frame(self.sidebar_canvas)
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar_inner, anchor="nw")

        self.sidebar_inner.columnconfigure(0, weight=1)
        self.sidebar_inner.bind("<Configure>", self._sync_sidebar_scroll)
        self.sidebar_canvas.bind("<Configure>", self._resize_sidebar_inner)
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Reject breakdown — at top of scrollable area
        breakdown_frame = ctk.CTkFrame(
            self.sidebar_inner, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER
        )
        breakdown_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(breakdown_frame, text="Reject Breakdown", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w", padx=12, pady=(12, 6)
        )
        breakdown_body = ctk.CTkFrame(breakdown_frame, fg_color=PANEL_ALT_BG, corner_radius=12, border_width=1, border_color=BORDER)
        breakdown_body.pack(fill="x", padx=12, pady=(0, 12))
        breakdown_body.grid_columnconfigure(1, weight=1)
        self.breakdown_labels: dict[str, ctk.CTkLabel] = {}
        for _idx, _key in enumerate(BREAKDOWN_ORDER):
            ctk.CTkLabel(breakdown_body, text=f"{_key}:", text_color=TEXT_PRIMARY).grid(
                row=_idx, column=0, sticky="w", padx=10, pady=3
            )
            _lbl = ctk.CTkLabel(breakdown_body, text="0", text_color=TEXT_SECONDARY)
            _lbl.grid(row=_idx, column=1, sticky="e", padx=10, pady=3)
            self.breakdown_labels[_key] = _lbl

        self.result_panel = ResultPanel(self.sidebar_inner)
        self.result_panel.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        events_frame = ctk.CTkFrame(self.sidebar_inner, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER)
        events_frame.grid(row=2, column=0, sticky="nsew")
        self.sidebar_inner.rowconfigure(2, weight=1)
        ctk.CTkLabel(events_frame, text="Recent Events", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w", padx=12, pady=(12, 6)
        )
        list_wrapper = ctk.CTkFrame(events_frame, fg_color=PANEL_ALT_BG, corner_radius=10, border_width=1, border_color=BORDER)
        list_wrapper.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.recent_list = tk.Listbox(
            list_wrapper,
            height=12,
            bg=PANEL_ALT_BG,
            fg=TEXT_PRIMARY,
            selectbackground=ACCENT_SOFT,
            selectforeground=TEXT_ON_ACCENT,
            borderwidth=0,
            relief="flat",
            font=("Segoe UI", 9),
            highlightthickness=0,
        )
        self.recent_list.pack(fill="both", expand=True, padx=4, pady=4)

    def _resolve_canvas_background(self) -> str:
        style = ttk.Style(self)
        background = style.lookup("TFrame", "background")
        if background:
            return str(background)
        return APP_BG

    def _sync_sidebar_scroll(self, _event=None) -> None:
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _resize_sidebar_inner(self, event) -> None:
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if self._closed or not self.winfo_exists():
            return
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return
        if widget == self.sidebar_canvas or str(widget).startswith(str(self.sidebar_inner)):
            self.sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _apply_responsive_layout(self) -> None:
        width = max(self.winfo_width(), self.winfo_toplevel().winfo_width())
        compact = width < RESPONSIVE_BREAKPOINT
        self._layout_top_bar(compact=compact)
        self._layout_context_bar(compact=compact)
        self._layout_status_strip(compact=compact)
        if compact != self._is_compact_layout:
            self._is_compact_layout = compact

            self.live_container.grid_forget()
            self.sidebar_container.grid_forget()
            self.content.columnconfigure(0, weight=1)
            self.content.columnconfigure(1, weight=0)
            self.content.rowconfigure(0, weight=1)
            self.content.rowconfigure(1, weight=0)

            if compact:
                self.live_container.grid(row=0, column=0, sticky="nsew")
                self.sidebar_container.grid(row=1, column=0, sticky="ew", pady=(10, 0))
                self.sidebar_container.configure(width=0, height=300)
            else:
                self.content.columnconfigure(0, weight=1)
                self.content.columnconfigure(1, weight=0)
                self.live_container.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
                self.sidebar_container.grid(row=0, column=1, sticky="ns")
                self.sidebar_container.configure(width=380, height=1)

        self._apply_preview_layout()

    def _apply_preview_layout(self) -> None:
        preview_width = max(self.live_container.winfo_width(), self.winfo_width())
        compact_preview = preview_width < 980
        if compact_preview == self._is_preview_compact:
            return
        self._is_preview_compact = compact_preview

        self.part_ready_preview.grid_forget()
        self.main_view.grid_forget()

        if compact_preview:
            self.preview_strip.rowconfigure(0, weight=2)
            self.preview_strip.rowconfigure(1, weight=3)
            self.preview_strip.columnconfigure(0, weight=1)
            self.preview_strip.columnconfigure(1, weight=0)
            self.part_ready_preview.grid(row=0, column=0, sticky="nw", pady=(0, 8))
            self.main_view.grid(row=1, column=0, sticky="nw")
        else:
            self.preview_strip.rowconfigure(0, weight=1)
            self.preview_strip.rowconfigure(1, weight=0)
            self.preview_strip.columnconfigure(0, weight=2)
            self.preview_strip.columnconfigure(1, weight=3)
            self.part_ready_preview.grid(row=0, column=0, sticky="nw", padx=(0, 8))
            self.main_view.grid(row=0, column=1, sticky="nw", padx=(8, 0))

    def _on_resize(self, _event=None) -> None:
        self.after_idle(self._apply_responsive_layout)

    def _layout_top_bar(self, *, compact: bool) -> None:
        for widget in self.action_bar.grid_slaves():
            widget.grid_forget()

        for column in range(6):
            self.action_bar.columnconfigure(column, weight=1 if compact else 0)
        self.action_bar.rowconfigure(0, weight=1)
        self.action_bar.rowconfigure(1, weight=1)

        if compact:
            self.action_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            self.template_box.grid(row=1, column=0, sticky="ew")
            self.template_box.columnconfigure(0, weight=1)
            self.template_box.columnconfigure(1, weight=0)
            for index, button in enumerate(self.action_buttons):
                row = 0 if index < 3 else 1
                column = index if index < 3 else index - 3
                button.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
        else:
            self.action_bar.grid(row=0, column=0, sticky="w")
            self.template_box.grid(row=0, column=1, sticky="e")
            for index, button in enumerate(self.action_buttons):
                button.grid(row=0, column=index, sticky="w", padx=(0 if index == 0 else 6, 0))

    def _layout_context_bar(self, *, compact: bool) -> None:
        for widget in self.context_bar.grid_slaves():
            widget.grid_forget()

        if compact:
            self.context_bar.columnconfigure(0, weight=1)
            self.context_bar.columnconfigure(1, weight=1)
            self.context_labels["operator"].grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
            self.context_labels["line"].grid(row=1, column=0, sticky="w", padx=(0, 8))
            self.context_labels["station"].grid(row=1, column=1, sticky="w")
            self.context_labels["template"].grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        else:
            for index in range(4):
                self.context_bar.columnconfigure(index, weight=1)
            self.context_labels["operator"].grid(row=0, column=0, sticky="w")
            self.context_labels["line"].grid(row=0, column=1, sticky="w", padx=8)
            self.context_labels["station"].grid(row=0, column=2, sticky="w", padx=8)
            self.context_labels["template"].grid(row=0, column=3, sticky="w", padx=8)

    def _layout_status_strip(self, *, compact: bool) -> None:
        for widget in self.status_badges_container.grid_slaves():
            widget.grid_forget()

        keys = ["SERVER", "CAMERA", "SESSION", "DB", "EVENT", "PLC"]
        columns = 3 if compact else 6
        rows = 2 if compact else 1
        for column in range(columns):
            self.status_badges_container.columnconfigure(column, weight=1)
        for row in range(rows):
            self.status_badges_container.rowconfigure(row, weight=1)

        for index, key in enumerate(keys):
            row = index // columns
            column = index % columns
            self.badges[key].grid(row=row, column=column, sticky="ew", padx=4, pady=3)

    def _open_settings(self) -> None:
        if self._settings_window and self._settings_window.winfo_exists():
            self._settings_window.lift()
            self._settings_window.focus_force()
            return

        window = tk.Toplevel(self)
        window.title("Operator Settings")
        window.geometry("700x520")
        window.transient(self.winfo_toplevel())
        window.resizable(True, True)
        self._settings_window = window
        window.protocol("WM_DELETE_WINDOW", self._close_settings)

        shell = ttk.Frame(window, padding=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        scroller = ScrollableFrame(shell)
        scroller.grid(row=0, column=0, sticky="nsew")
        body = scroller.body
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        general = ttk.LabelFrame(body, text="Runtime Context", padding=10)
        general.grid(row=0, column=0, columnspan=2, sticky="ew")
        general.columnconfigure(1, weight=1)
        general.columnconfigure(3, weight=1)
        self._settings_entry(general, 0, 0, "Line", self.line_value)
        self._settings_entry(general, 0, 2, "Station", self.station_value)
        self._settings_entry(general, 1, 0, "Camera", self.camera_value)
        self._settings_entry(general, 1, 2, "Template Ver", self.template_version_value)

        part_ready_roi = ttk.LabelFrame(body, text="Part Ready ROI", padding=10)
        part_ready_roi.grid(row=1, column=0, sticky="nsew", pady=(12, 0), padx=(0, 6))
        for index in range(8):
            part_ready_roi.columnconfigure(index, weight=1)
        ttk.Label(
            part_ready_roi,
            text="Isi urutan: x = kiri, y = atas, w = lebar, h = tinggi.",
            foreground="#475569",
            wraplength=280,
            justify="left",
        ).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 8))
        self._settings_roi_entry(part_ready_roi, 1, 0, "x (left)", self.part_ready_roi_x_value)
        self._settings_roi_entry(part_ready_roi, 1, 2, "y (top)", self.part_ready_roi_y_value)
        self._settings_roi_entry(part_ready_roi, 1, 4, "w (width)", self.part_ready_roi_w_value)
        self._settings_roi_entry(part_ready_roi, 1, 6, "h (height)", self.part_ready_roi_h_value)

        sticker_roi = ttk.LabelFrame(body, text="Sticker ROI", padding=10)
        sticker_roi.grid(row=1, column=1, sticky="nsew", pady=(12, 0), padx=(6, 0))
        for index in range(8):
            sticker_roi.columnconfigure(index, weight=1)
        ttk.Label(
            sticker_roi,
            text="Isi urutan: x = kiri, y = atas, w = lebar, h = tinggi.",
            foreground="#475569",
            wraplength=280,
            justify="left",
        ).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 8))
        self._settings_roi_entry(sticker_roi, 1, 0, "x (left)", self.sticker_roi_x_value)
        self._settings_roi_entry(sticker_roi, 1, 2, "y (top)", self.sticker_roi_y_value)
        self._settings_roi_entry(sticker_roi, 1, 4, "w (width)", self.sticker_roi_w_value)
        self._settings_roi_entry(sticker_roi, 1, 6, "h (height)", self.sticker_roi_h_value)

        ttk.Label(
            body,
            text="ROI disimpan dalam format rasio 0-1 terhadap frame kamera. Gunakan part-ready ROI untuk gate warna, dan sticker ROI untuk inferensi model. Right view akan menampilkan frame penuh dengan ROI sticker atau overlay hasil machine learning.",
            foreground="#475569",
            wraplength=620,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))

        footer = ttk.Frame(shell)
        footer.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(footer, text="Load Deployment", command=self._load_deployment).pack(side="left")
        ttk.Button(footer, text="Use Template ROI", command=self._sync_selected_template_detail).pack(side="left", padx=8)
        ttk.Button(footer, text="Apply ROI", command=self._apply_roi).pack(side="left")
        ttk.Button(footer, text="Close", command=self._close_settings).pack(side="right")

    def _close_settings(self) -> None:
        if self._settings_window and self._settings_window.winfo_exists():
            self._settings_window.destroy()
        self._settings_window = None
        self._refresh_context_summary()

    def _settings_entry(self, master, row: int, column: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(master, textvariable=variable).grid(row=row, column=column + 1, sticky="ew", padx=(0, 12), pady=6)

    def _settings_roi_entry(self, master, row: int, column: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=variable, width=10).grid(row=row, column=column + 1, sticky="ew", padx=4, pady=4)

    def _load_template_choices(self) -> None:
        try:
            items = self.api.list_templates()
        except Exception as exc:  # noqa: BLE001
            self.info_var.set(f"Failed to load template list: {exc}")
            return
        values: list[str] = []
        self._template_lookup = {}
        self._template_detail_lookup = {}
        self._template_version_detail_lookup = {}
        for item in items:
            label = f"{item['name']} | v{item.get('version_number')} | version_id={item.get('version_id')}"
            values.append(label)
            self._template_lookup[label] = item
        self.template_selector.configure(values=values)
        self._sync_template_selector()

    def _auto_start_first_template(self) -> None:
        if self._closed or self._auto_start_done:
            return
        self._auto_start_done = True
        values = list(self.template_selector.cget("values") or [])
        if not values:
            self.info_var.set("Tidak ada template tersedia untuk auto start.")
            return
        first_label = str(values[0])
        self.template_choice.set(first_label)
        self.info_var.set("Auto start: memilih template pertama.")
        self._on_template_selected()
        if not self._start_camera(show_errors=False):
            self.info_var.set("Auto start: kamera gagal dibuka. Start Camera bisa dilakukan manual.")
            return
        self._wait_for_auto_start_frame(attempt=0)

    def _wait_for_auto_start_frame(self, *, attempt: int) -> None:
        if self._closed or self.state.active_session:
            return
        if self.capture.get_latest_frame() is not None:
            self.info_var.set("Auto start: frame kamera siap, memulai session.")
            self._start_session()
            return
        if attempt >= AUTO_START_MAX_ATTEMPTS:
            self.info_var.set("Auto start: kamera belum menghasilkan frame. Start Session bisa dilakukan manual.")
            return
        self._auto_start_after_id = self.after(
            AUTO_START_FRAME_WAIT_MS,
            lambda: self._wait_for_auto_start_frame(attempt=attempt + 1),
        )

    def _fetch_template_detail(self, template_id: int) -> dict:
        template_id = int(template_id)
        cached = self._template_detail_lookup.get(template_id)
        if cached:
            return cached
        detail = self.api.get_template(template_id)
        self._template_detail_lookup[template_id] = detail
        return detail

    def _fetch_template_version_detail(self, version_id: int) -> dict:
        version_id = int(version_id)
        cached = self._template_version_detail_lookup.get(version_id)
        if cached:
            return cached
        detail = self.api.get_template_version(version_id)
        self._template_version_detail_lookup[version_id] = detail
        return detail

    def _roi_vars(self, kind: str) -> dict[str, tk.StringVar]:
        if kind == "part_ready":
            return {
                "x": self.part_ready_roi_x_value,
                "y": self.part_ready_roi_y_value,
                "w": self.part_ready_roi_w_value,
                "h": self.part_ready_roi_h_value,
            }
        return {
            "x": self.sticker_roi_x_value,
            "y": self.sticker_roi_y_value,
            "w": self.sticker_roi_w_value,
            "h": self.sticker_roi_h_value,
        }

    def _format_roi_value(self, value) -> str:
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value or "")

    def _set_roi_values(self, kind: str, roi: dict | None) -> None:
        values = self._roi_vars(kind)
        payload = roi or {}
        for key, variable in values.items():
            if key in payload:
                variable.set(self._format_roi_value(payload.get(key)))

    def _float_value(self, value: str, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_upload_interval_ms(self) -> int:
        fallback = max(50, int(DEFAULT_UPLOAD_INTERVAL_MS))
        detail = self.state.cache.get("selected_template_detail") if isinstance(self.state.cache, dict) else None
        if not isinstance(detail, dict):
            return fallback
        vision = detail.get("vision") if isinstance(detail.get("vision"), dict) else {}
        try:
            inference_fps = float(vision.get("inference_fps") or 0.0)
        except (TypeError, ValueError):
            return fallback
        if inference_fps <= 0.0:
            return fallback
        return max(50, int(round(1000.0 / inference_fps)))

    def _read_roi_payload(self, kind: str) -> dict[str, float]:
        variables = self._roi_vars(kind)
        return {
            key: self._float_value(variable.get(), default=(1.0 if key in {"w", "h"} else 0.0))
            for key, variable in variables.items()
        }

    def _validated_roi_payload(self, kind: str) -> dict[str, float]:
        variables = self._roi_vars(kind)
        payload: dict[str, float] = {}
        label = "Part Ready ROI" if kind == "part_ready" else "Sticker ROI"
        for key, variable in variables.items():
            raw = variable.get().strip()
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} field '{key}' harus numerik.") from exc
            if key in {"x", "y"} and not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} field '{key}' harus di rentang 0 sampai 1.")
            if key in {"w", "h"} and not 0.0 < value <= 1.0:
                raise ValueError(f"{label} field '{key}' harus lebih besar dari 0 dan maksimal 1.")
            payload[key] = value
        return payload

    def _crop_local_roi(self, frame, roi_payload: dict[str, float]):
        if frame is None:
            return None
        height, width = frame.shape[:2]
        x = max(0, min(width - 1, int(float(roi_payload.get("x", 0.0)) * width)))
        y = max(0, min(height - 1, int(float(roi_payload.get("y", 0.0)) * height)))
        roi_w = max(1, int(float(roi_payload.get("w", 1.0)) * width))
        roi_h = max(1, int(float(roi_payload.get("h", 1.0)) * height))
        x2 = min(width, x + roi_w)
        y2 = min(height, y + roi_h)
        cropped = frame[y:y2, x:x2]
        if cropped.size == 0:
            return None
        return cropped

    def _build_full_frame_with_roi(self, frame, kind: str, *, label: str, color: tuple[int, int, int]):
        if frame is None:
            return None
        payload = self._read_roi_payload(kind)
        height, width = frame.shape[:2]
        x = max(0, min(width - 1, int(float(payload.get("x", 0.0)) * width)))
        y = max(0, min(height - 1, int(float(payload.get("y", 0.0)) * height)))
        roi_w = max(1, int(float(payload.get("w", 1.0)) * width))
        roi_h = max(1, int(float(payload.get("h", 1.0)) * height))
        x2 = min(width, x + roi_w)
        y2 = min(height, y + roi_h)
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} | x={x} y={y} w={x2 - x} h={y2 - y}",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _build_local_detection_overlay(self, frame, payload: dict):
        """Render detection bboxes onto the live camera frame using payload data.

        Replicates the backend overlay composition locally so that the network
        round-trip cost of JPEG-encoding and transmitting the overlay image is
        eliminated entirely (stream response mode).
        """
        if frame is None:
            return None
        overlay = frame.copy()
        validation = payload.get("validation") or {}
        part_ready = payload.get("part_ready") or {}
        sticker_roi = payload.get("sticker_roi_meta") or {}
        part_ready_roi = payload.get("part_ready_roi_meta") or {}
        detections = payload.get("detections") or []
        event_state = payload.get("event_state") or "idle"

        # Backend computes roi_meta pixel coords from the (possibly downscaled) frame
        # that was uploaded. The display overlay is drawn on the full-resolution live
        # camera frame. Scale all backend pixel coordinates accordingly.
        client_timings = payload.get("client_timings") or {}
        sent_w = client_timings.get("frame_width")
        sent_h = client_timings.get("frame_height")
        disp_h, disp_w = frame.shape[:2]
        scale_x = disp_w / sent_w if sent_w else 1.0
        scale_y = disp_h / sent_h if sent_h else 1.0

        from shared.contracts.enums import DecisionCode
        decision = validation.get("decision")
        reject_reason = validation.get("reject_reason_code") or "OK"
        decision_color = (0, 180, 0) if decision == DecisionCode.ACCEPT.value else (0, 0, 220)

        # Part ready ROI box (blue).
        if part_ready_roi.get("width") and part_ready_roi.get("height"):
            px = int(part_ready_roi["x"] * scale_x)
            py = int(part_ready_roi["y"] * scale_y)
            pw = int(part_ready_roi["width"] * scale_x)
            ph = int(part_ready_roi["height"] * scale_y)
            cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (50, 180, 255), 2)

        # Sticker ROI box (yellow).
        sx = int(sticker_roi.get("x", 0) * scale_x)
        sy = int(sticker_roi.get("y", 0) * scale_y)
        sw = int(sticker_roi.get("width", 0) * scale_x)
        sh = int(sticker_roi.get("height", 0) * scale_y)
        if sw and sh:
            cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (255, 200, 0), 2)

        # Detection bounding boxes (coordinates are relative to sticker ROI, in backend frame space).
        for det in detections:
            pos = det.get("position") or {}
            x1 = int(sx + float(pos.get("x1", 0)) * scale_x)
            y1 = int(sy + float(pos.get("y1", 0)) * scale_y)
            x2 = int(sx + float(pos.get("x2", 0)) * scale_x)
            y2 = int(sy + float(pos.get("y2", 0)) * scale_y)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            lbl = str(det.get("label") or "")
            conf = float(det.get("confidence") or 0.0)
            cv2.putText(overlay, f"{lbl} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Expected center crosshair.
        if sw and sh:
            template_detail = self.state.cache.get("selected_template_detail") if isinstance(self.state.cache, dict) else None
            sticker_cfg = (template_detail or {}).get("sticker") or {}
            cx_ratio = float(sticker_cfg.get("expected_center_x") or 0.5)
            cy_ratio = float(sticker_cfg.get("expected_center_y") or 0.5)
            exp_x = int(sx + cx_ratio * sw)
            exp_y = int(sy + cy_ratio * sh)
            arm = 18
            cv2.line(overlay, (exp_x - arm, exp_y), (exp_x + arm, exp_y), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(overlay, (exp_x, exp_y - arm), (exp_x, exp_y + arm), (0, 220, 255), 2, cv2.LINE_AA)

        # Decision and status text.
        cv2.putText(overlay, f"{decision} / {reject_reason}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, decision_color, 2, cv2.LINE_AA)
        pr_val = part_ready.get("part_ready")
        pr_ratio = part_ready.get("match_ratio", part_ready.get("part_ready_confidence", "-"))
        cv2.putText(overlay, f"part_ready={pr_val} ratio={pr_ratio}", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"state={event_state}", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return overlay

    def _update_local_roi_previews(self, frame) -> None:
        if frame is None:
            self.part_ready_preview.reset()
            self.main_view.reset()
            return
        part_ready_crop = self._crop_local_roi(frame, self._read_roi_payload("part_ready"))
        sticker_scene = self._build_full_frame_with_roi(frame, "sticker", label="Sticker ROI", color=(255, 214, 10))
        if part_ready_crop is not None:
            self.part_ready_preview.update_bgr(part_ready_crop)
        else:
            self.part_ready_preview.reset()
        if sticker_scene is not None:
            self.main_view.update_bgr(sticker_scene)
            self.display_source.set("Right View: Live Camera + Sticker ROI")
        else:
            self.main_view.reset()

    def _current_template_id(self) -> int | None:
        selected = self._template_lookup.get(self.template_choice.get().strip())
        if selected and selected.get("id"):
            return int(selected["id"])
        deployment = self.state.active_deployment or {}
        if deployment.get("template_id"):
            return int(deployment["template_id"])
        current_version = self.template_version_value.get().strip()
        for item in self._template_lookup.values():
            if str(item.get("version_id")) == current_version and item.get("id"):
                return int(item["id"])
        return None

    def _apply_template_detail(
        self,
        detail: dict | None,
        *,
        lock_version_id: int | None = None,
        keep_line_station: bool = False,
    ) -> None:
        if not detail:
            return
        version_id = lock_version_id or detail.get("version_id") or detail.get("current_version_id")
        if version_id:
            self.template_version_value.set(str(version_id))
        self._set_roi_values("part_ready", detail.get("part_ready_roi") or detail.get("roi") or {})
        self._set_roi_values("sticker", detail.get("sticker_roi") or detail.get("roi") or {})
        camera_config = detail.get("camera") or {}
        if camera_config.get("camera_index") is not None:
            self.camera_value.set(str(camera_config["camera_index"]))
        sticker_config = detail.get("sticker") or {}
        if not keep_line_station:
            if sticker_config.get("line"):
                self.line_value.set(str(sticker_config["line"]))
            if sticker_config.get("station"):
                self.station_value.set(str(sticker_config["station"]))
        self.state.cache["selected_template_detail"] = detail
        self._refresh_context_summary()

    def _sync_selected_template_detail(self) -> None:
        template_id = self._current_template_id()
        if not template_id:
            self.info_var.set("Belum ada template aktif untuk disinkronkan.")
            return
        try:
            detail = self._fetch_template_detail(template_id)
        except Exception as exc:  # noqa: BLE001
            self.info_var.set(f"Failed to sync template detail: {exc}")
            return
        self._apply_template_detail(detail)
        self.info_var.set(f"Template ROI synced: {detail.get('name')} v{detail.get('version_number')}")

    def _sync_template_selector(self) -> None:
        current_version = str(self.template_version_value.get().strip() or "")
        if not current_version:
            return
        for label, item in self._template_lookup.items():
            if str(item.get("version_id")) == current_version:
                self.template_choice.set(label)
                return

    def _on_template_selected(self, _event=None) -> None:
        label = self.template_choice.get().strip()
        item = self._template_lookup.get(label)
        if not item:
            return
        session_was_running = self.state.active_session is not None
        previous_camera_index = self.camera_value.get().strip()
        self.state.active_deployment = None
        try:
            detail = self._fetch_template_detail(int(item["id"]))
        except Exception as exc:  # noqa: BLE001
            self.template_version_value.set(str(item.get("version_id") or ""))
            self.info_var.set(f"Failed to load template detail: {exc}")
        else:
            if session_was_running:
                self._stop_session()
            self._apply_template_detail(detail)
            camera_changed = self.camera_value.get().strip() != previous_camera_index
            if camera_changed:
                self._restart_camera_for_template_change()
            if session_was_running:
                self.info_var.set(f"Template changed: restarting session with {item.get('name')} v{item.get('version_number')}")
                self._restart_session_after_template_change()
            else:
                self.info_var.set(f"Template selected manually: {item.get('name')} v{item.get('version_number')}")
        self._refresh_context_summary()

    def _restart_camera_for_template_change(self) -> bool:
        self.capture.stop()
        self.main_view.reset()
        self.part_ready_preview.reset()
        return self._start_camera(show_errors=False)

    def _restart_session_after_template_change(self) -> None:
        if self.capture.get_latest_frame() is not None:
            self._start_session()
            return
        if not self._start_camera(show_errors=False):
            self.info_var.set("Template changed, tapi kamera belum siap. Start Camera/Session bisa dilakukan manual.")
            return
        self._wait_for_auto_start_frame(attempt=0)

    def _selected_template_name(self) -> str | None:
        selected = self._template_lookup.get(self.template_choice.get().strip())
        if not selected:
            return None
        return str(selected.get("name") or "").strip() or None

    def _set_badge(self, key: str, value: str, tone: str = "neutral") -> None:
        bg, fg = BADGE_COLORS.get(tone, BADGE_COLORS["neutral"])
        self.badges[key].configure(text=f"{key}: {value}", fg_color=bg, text_color=fg)

    def _on_stiker_terpasang(self) -> None:
        self.stiker_terpasang_btn.configure(state="disabled")

        def _work():
            return self.api.plc_sticker_done()

        def _on_done(result, error):
            if error:
                self.plc_status_label.configure(text=f"Gagal release: {error}")
                self.stiker_terpasang_btn.configure(state="normal")
            else:
                self.plc_status_label.configure(text="Stiker terpasang — clamp dilepas.")
                self._set_badge("PLC", "READY", "success")

        run_async(self, _work, callback=_on_done)

    def _open_plc_release_dialog(self) -> None:
        if self._plc_release_window and self._plc_release_window.winfo_exists():
            self._plc_release_window.lift()
            self._plc_release_window.focus_force()
            return

        dialog = tk.Toplevel(self)
        dialog.title("Admin Authorization — Release Clamp")
        dialog.geometry("400x290")
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        self._plc_release_window = dialog
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._close_plc_dialog(dialog))

        shell = ttk.Frame(dialog, padding=16)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(1, weight=1)

        ttk.Label(
            shell,
            text="Tindakan ini mengirim sinyal release clamp ke PLC.\nHanya admin yang dapat melakukan ini.",
            foreground="#991b1b",
            wraplength=340,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(shell, text="Username Admin:").grid(row=1, column=0, sticky="w", pady=4, padx=(0, 8))
        username_var = tk.StringVar()
        ttk.Entry(shell, textvariable=username_var, width=28).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(shell, text="Password Admin:").grid(row=2, column=0, sticky="w", pady=4, padx=(0, 8))
        password_var = tk.StringVar()
        ttk.Entry(shell, textvariable=password_var, show="*", width=28).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(shell, text="Alasan:").grid(row=3, column=0, sticky="w", pady=4, padx=(0, 8))
        reason_var = tk.StringVar(value="manual_admin")
        ttk.Entry(shell, textvariable=reason_var, width=28).grid(row=3, column=1, sticky="ew", pady=4)

        status_var = tk.StringVar(value="")
        status_lbl = ttk.Label(shell, textvariable=status_var, foreground="#1d4ed8", wraplength=340, justify="left")
        status_lbl.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 4))

        btn_frame = ttk.Frame(shell)
        btn_frame.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))

        cancel_btn = ttk.Button(btn_frame, text="Batal", command=lambda: self._close_plc_dialog(dialog))
        cancel_btn.pack(side="left", padx=(0, 8))

        confirm_btn = ctk.CTkButton(
            btn_frame,
            text="Release Clamp ▶",
            fg_color="#991b1b",
            hover_color="#7f1d1d",
            text_color="#fef2f2",
            font=("Segoe UI", 11, "bold"),
            command=lambda: self._execute_plc_release(
                username_var.get().strip(),
                password_var.get(),
                reason_var.get().strip() or "manual_admin",
                status_var,
                status_lbl,
                confirm_btn,
                cancel_btn,
                dialog,
            ),
        )
        confirm_btn.pack(side="left")

    def _close_plc_dialog(self, dialog: tk.Toplevel) -> None:
        if dialog.winfo_exists():
            dialog.destroy()
        self._plc_release_window = None

    def _execute_plc_release(
        self,
        username: str,
        password: str,
        reason: str,
        status_var: tk.StringVar,
        status_lbl,
        confirm_btn,
        cancel_btn,
        dialog: tk.Toplevel,
    ) -> None:
        if not username or not password:
            status_var.set("Username dan password tidak boleh kosong.")
            status_lbl.configure(foreground="#991b1b")
            return

        status_var.set("Mengautentikasi admin...")
        status_lbl.configure(foreground="#1d4ed8")
        confirm_btn.configure(state="disabled")
        cancel_btn.configure(state="disabled")

        def _work():
            from client_tk.app.api_client import ApiClient
            temp = ApiClient(self.api.base_url)
            login_result = temp.login(username, password)
            token = login_result.get("token")
            if not token:
                raise RuntimeError("Login berhasil tapi token tidak ditemukan.")
            user_info = login_result.get("user") or {}
            role = str(user_info.get("role") or "").strip().lower()
            if role != "admin":
                raise RuntimeError(f"Akun '{username}' bukan admin (role={role or '-'}).")
            temp.set_token(token)
            return temp.plc_manual_release(reason)

        def _on_done(result, error):
            confirm_btn.configure(state="normal")
            cancel_btn.configure(state="normal")
            if error:
                msg = str(error)
                if "503" in msg or "disabled" in msg.lower():
                    status_var.set("PLC dinonaktifkan di backend (QC_SUITE_PLC_ENABLED=0).")
                    self._set_badge("PLC", "DISABLED", "neutral")
                    self.plc_status_label.configure(text="PLC dinonaktifkan di backend.")
                elif "403" in msg or "forbidden" in msg.lower():
                    status_var.set("Akses ditolak. Pastikan akun yang dimasukkan adalah admin.")
                    self._set_badge("PLC", "DENIED", "danger")
                elif "401" in msg or "invalid" in msg.lower() or "credentials" in msg.lower():
                    status_var.set("Login gagal: username atau password salah.")
                    self._set_badge("PLC", "AUTH ERR", "danger")
                else:
                    status_var.set(f"Error: {msg}")
                    self._set_badge("PLC", "ERROR", "danger")
                    self.plc_status_label.configure(text=f"PLC error: {msg[:60]}")
                status_lbl.configure(foreground="#991b1b")
            else:
                status_var.set("Clamp berhasil dirilis.")
                status_lbl.configure(foreground="#166534")
                self._set_badge("PLC", "RELEASED", "success")
                self.plc_status_label.configure(text=f"Clamp dirilis oleh admin '{username}'.")
                dialog.after(1500, lambda: self._close_plc_dialog(dialog))

        run_async(dialog, _work, callback=_on_done)

    def _refresh_context_summary(self) -> None:
        username = self.state.user.get("username") if self.state.user else "-"
        self.operator_context.set(f"Operator: {username}")
        line = self.line_value.get().strip() or (self.state.active_session or {}).get("line_id") or "-"
        station = self.station_value.get().strip() or (self.state.active_session or {}).get("station_id") or "-"
        selected_detail = self.state.cache.get("selected_template_detail") if isinstance(self.state.cache, dict) else None
        template_name = (
            (self.state.active_session or {}).get("template_name")
            or (self.state.active_deployment or {}).get("template_name")
            or (selected_detail or {}).get("name")
            or self._selected_template_name()
            or "-"
        )
        template_version = self.template_version_value.get().strip() or (self.state.active_session or {}).get("template_version_id") or "-"
        self.line_context.set(f"Line: {line}")
        self.station_context.set(f"Station: {station}")
        self.template_context.set(f"Template: {template_name} v{template_version}")
        self._sync_template_selector()

    def _update_status_badges(self, payload: dict | None = None) -> None:
        token = getattr(self.state, "token", None)
        latest_error = getattr(self.state, "latest_error", None)
        server_tone = "success" if token and not latest_error else "danger" if latest_error else "info"
        self._set_badge("SERVER", "ONLINE" if token and not latest_error else "ISSUE", server_tone)

        camera_ready = self.capture.get_latest_frame() is not None
        self._set_badge("CAMERA", "READY" if camera_ready else "STOPPED", "success" if camera_ready else "neutral")

        session_running = self.state.active_session is not None
        self._set_badge("SESSION", "RUNNING" if session_running else "IDLE", "info" if session_running else "neutral")

        db_write = ((payload or {}).get("last_committed_result") or {}).get("db_write") or (payload or {}).get("db_write") or {}
        if db_write.get("written"):
            self._set_badge("DB", "WRITTEN", "success")
        elif db_write.get("reason") in {"disabled", "not_committed"}:
            self._set_badge("DB", "WAITING", "neutral")
        else:
            self._set_badge("DB", str(db_write.get("reason") or "ISSUE").upper(), "warning")

        event_state = str((payload or {}).get("event_state") or "idle").upper()
        event_tone = "success" if event_state == "DECISION_COMMITTED" else "info" if event_state not in {"IDLE", "COOLDOWN"} else "neutral"
        self._set_badge("EVENT", event_state, event_tone)

    def _sync_recent_events(self, payload: dict) -> None:
        items = payload.get("recent_events") or []
        self.recent_list.delete(0, "end")
        for item in items:
            timestamp = str(item.get("committed_at") or "-").replace("T", " ")[:19]
            decision = item.get("decision") or "-"
            reason = item.get("reject_reason_code") or "OK"
            part_name = item.get("part_name") or "-"
            self.recent_list.insert("end", f"{timestamp} | {decision} | {part_name} | {reason}")

    def _load_deployment(self) -> None:
        line_id = self.line_value.get().strip()
        station_id = self.station_value.get().strip()
        if not line_id or not station_id:
            messagebox.showerror("Deployment", "Line dan Station wajib diisi.")
            return
        response = self.api.get_active_deployment(line_id, station_id)
        deployment = response.get("deployment") if isinstance(response, dict) else None
        if not deployment:
            messagebox.showwarning("Deployment", "Tidak ada deployment aktif.")
            return
        self.state.active_deployment = deployment
        self.line_value.set(str(deployment.get("line_id") or line_id))
        self.station_value.set(str(deployment.get("station_id") or station_id))
        deployment_version_id = int(deployment.get("template_version_id") or 0)
        self.template_version_value.set(str(deployment_version_id or ""))
        detail = None
        if deployment_version_id:
            try:
                detail = self._fetch_template_version_detail(deployment_version_id)
            except Exception as exc:  # noqa: BLE001
                self.info_var.set(f"Deployment loaded, tapi detail template version gagal dibaca: {exc}")

        if detail is None and deployment.get("template_id"):
            try:
                detail = self._fetch_template_detail(int(deployment["template_id"]))
            except Exception as exc:  # noqa: BLE001
                self.info_var.set(f"Deployment loaded, tapi detail template gagal dibaca: {exc}")

        if detail is not None:
            self._apply_template_detail(
                detail,
                lock_version_id=deployment_version_id or None,
                keep_line_station=True,
            )
        self.info_var.set(f"Deployment loaded: {deployment.get('template_name')}")
        self._refresh_context_summary()
        self._update_status_badges()

    def _start_camera(self, *, show_errors: bool = True) -> bool:
        try:
            self.capture.start(int(self.camera_value.get() or 0))
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                messagebox.showerror("Camera", str(exc))
            self.info_var.set(f"Camera failed: {exc}")
            self._update_status_badges()
            return False
        self.info_var.set("Camera started. Menunggu frame pertama.")
        self._update_status_badges()
        return True

    def _stop_camera(self) -> None:
        self.capture.stop()
        self.main_view.reset()
        self.part_ready_preview.reset()
        self.info_var.set("Camera stopped.")
        self.display_source.set("Right View: Live Camera + Sticker ROI")
        self._update_status_badges()

    def _start_session(self) -> None:
        if self.capture.get_latest_frame() is None:
            messagebox.showwarning("Session", "Start camera dan tunggu frame pertama dulu.")
            return
        template_version_id = int(self.template_version_value.get() or 0)
        if not template_version_id:
            messagebox.showerror("Session", "Template version wajib diisi.")
            return
        try:
            part_ready_roi = self._validated_roi_payload("part_ready")
            sticker_roi = self._validated_roi_payload("sticker")
        except ValueError as exc:
            messagebox.showerror("Session", str(exc))
            return
        payload = self.api.create_session(
            {
                "client_id": str(self.state.user.get("id") if self.state.user else "client"),
                "camera_index": int(self.camera_value.get() or 0),
                "template_version_id": template_version_id,
                "line_id": self.line_value.get().strip(),
                "station_id": self.station_value.get().strip(),
            }
        )
        self.state.active_session = payload
        self.api.update_rois(
            payload["session_id"],
            part_ready_roi=part_ready_roi,
            sticker_roi=sticker_roi,
        )
        self.state.latest_result = None
        self.state.latest_error = None
        self.state.cache["part_ready"] = None
        self.state.cache["sticker_detection"] = None
        self.state.cache["last_committed_result"] = None
        self._auth_error_notified = False
        with self._lock:
            self._latest_payload = None
            self._latest_error = None
        self.result_panel.reset()
        self.counter_panel.reset()
        self.decision_banner.configure(fg_color="#334155", text_color="#f8fafc", text="WAITING")
        self.decision_subtitle.configure(text="Menunggu event inspeksi pertama.")
        self.plc_status_label.configure(text="")
        for _lbl in self.breakdown_labels.values():
            _lbl.configure(text="0")
        self.recent_list.delete(0, "end")
        # Restore raw preview immediately so camera stays visible during first backend round-trip.
        _current_frame = self.capture.get_latest_frame()
        if _current_frame is not None:
            self._update_local_roi_previews(_current_frame)
        else:
            self.part_ready_preview.reset()
            self.main_view.reset()
        upload_interval_ms = self._resolve_upload_interval_ms()
        self.uploader.start(
            interval_ms=upload_interval_ms,
            get_frame=self.capture.get_latest_frame,
            send_frame=lambda image_b64: self.api.push_frame(
                payload["session_id"],
                image_b64,
                # "stream" skips backend overlay compose+encode (~10-50ms saved per frame).
                # Client renders detection boxes locally from bbox data in the payload.
                response_mode="stream",
            ),
            on_result=self._set_result,
            on_error=self._set_error,
        )
        infer_fps_actual = round(1000.0 / upload_interval_ms, 1)
        self.info_var.set(
            f"Session running: {payload['session_id']} "
            f"(infer @ {upload_interval_ms} ms / {infer_fps_actual} fps | preview @ 10 fps | JPEG q=75)"
        )
        self._refresh_context_summary()
        self._update_status_badges()

    def _stop_session(self) -> None:
        stop_message = "Session stopped."
        if self.state.active_session:
            try:
                self.api.stop_session(self.state.active_session["session_id"])
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if self._is_auth_error(message):
                    stop_message = "Session lokal dihentikan, tapi stop di backend gagal (401). Silakan login ulang."
                    if not self._auth_error_notified:
                        self._auth_error_notified = True
                        messagebox.showwarning("Session", "Sesi backend sudah tidak terotorisasi (401). Silakan login ulang.")
                else:
                    stop_message = f"Session lokal dihentikan. Warning backend: {message}"
        self.uploader.stop()
        self.state.active_session = None
        self.state.cache["part_ready"] = None
        self.state.cache["sticker_detection"] = None
        self.info_var.set(stop_message)
        self._refresh_context_summary()
        self._update_status_badges()

    def _apply_roi(self) -> None:
        try:
            part_ready_roi = self._validated_roi_payload("part_ready")
            sticker_roi = self._validated_roi_payload("sticker")
        except ValueError as exc:
            messagebox.showerror("ROI", str(exc))
            return
        if not self.state.active_session:
            self.info_var.set("Dua ROI disimpan lokal. Akan diterapkan saat session aktif.")
            self._refresh_context_summary()
            return
        self.api.update_rois(
            self.state.active_session["session_id"],
            part_ready_roi=part_ready_roi,
            sticker_roi=sticker_roi,
        )
        self.info_var.set("Part-ready ROI dan sticker ROI updated.")

    def _set_result(self, payload: dict) -> None:
        with self._lock:
            self._latest_payload = payload
            self._latest_error = None
        self._auth_error_notified = False

    @staticmethod
    def _is_auth_error(message: str | None) -> bool:
        text = str(message or "").strip().lower()
        return "401" in text or "unauthorized" in text

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._latest_error = message

    def _schedule_poll(self) -> None:
        if self._closed:
            return
        self._after_id = self.after(100, self._poll_ui)

    def _schedule_heartbeat(self, *, delay_ms: int | None = None) -> None:
        if self._closed:
            return
        if self._heartbeat_after_id:
            try:
                self.after_cancel(self._heartbeat_after_id)
            except tk.TclError:
                pass
        wait = HEARTBEAT_INTERVAL_MS if delay_ms is None else max(500, int(delay_ms))
        self._heartbeat_after_id = self.after(wait, self._send_heartbeat)

    def _send_heartbeat(self) -> None:
        self._heartbeat_after_id = None
        if self._closed:
            return
        if not getattr(self.state, "token", None):
            self._schedule_heartbeat()
            return

        line_id = self.line_value.get().strip() or None
        station_id = self.station_value.get().strip() or None

        def _load():
            return self.api.heartbeat(
                self._machine_id,
                client_version="client_tk",
                line_id=line_id,
                station_id=station_id,
            )

        run_async(self, _load, callback=lambda _result, _error: None)
        self._schedule_heartbeat()

    def _schedule_plc_poll(self, *, delay_ms: int | None = None) -> None:
        if self._closed:
            return
        if self._plc_poll_after_id:
            try:
                self.after_cancel(self._plc_poll_after_id)
            except tk.TclError:
                pass
        wait = PLC_POLL_INTERVAL_MS if delay_ms is None else max(500, int(delay_ms))
        self._plc_poll_after_id = self.after(wait, self._poll_plc_status)

    def _poll_plc_status(self) -> None:
        self._plc_poll_after_id = None
        if self._closed:
            return
        if not getattr(self.state, "token", None):
            self._schedule_plc_poll()
            return

        def _load():
            return self.api.plc_status()

        def _on_done(result, error):
            if self._closed:
                return
            if result and not error:
                self._update_plc_badge_from_status(result)

        run_async(self, _load, callback=_on_done)
        self._schedule_plc_poll()

    def _update_plc_badge_from_status(self, status: dict) -> None:
        if not status.get("enabled", True):
            self._set_badge("PLC", "DISABLED", "neutral")
            self.stiker_terpasang_btn.configure(state="disabled")
            return
        if not status.get("running"):
            self._set_badge("PLC", "STOPPED", "danger")
            self.stiker_terpasang_btn.configure(state="disabled")
            return
        connected = status.get("connected", False)
        clamp_engaged = status.get("clamp_engaged", False)
        if not connected:
            self._set_badge("PLC", "DISCONN", "warning")
            self.stiker_terpasang_btn.configure(state="disabled")
        elif clamp_engaged:
            self._set_badge("PLC", "ENGAGED", "warning")
            self.stiker_terpasang_btn.configure(state="normal")
        else:
            self._set_badge("PLC", "READY", "success")
            self.stiker_terpasang_btn.configure(state="disabled")

    def _poll_ui(self) -> None:
        if self._closed:
            return
        try:
            frame = self.capture.get_latest_frame()

            with self._lock:
                payload = self._latest_payload
                error = self._latest_error

            if payload:
                self.state.latest_result = payload
                self.state.latest_error = None
                self.state.cache["part_ready"] = payload.get("part_ready")
                self.state.cache["sticker_detection"] = payload.get("sticker_detection")
                self.state.cache["last_committed_result"] = payload.get("last_committed_result")
                self.result_panel.update_payload(payload)
                self.counter_panel.update_payload(payload)
                # Update decision banner (fixed area below counter)
                _live_val = payload.get("validation") or {}
                _committed = payload.get("last_committed_result") or {}
                _committed_val = _committed.get("validation") or {}
                _disp_val = _committed_val or _live_val
                _decision = _disp_val.get("decision") or "WAITING"
                _d_palette = {"ACCEPT": ("#166534", "#f0fdf4"), "REJECT": ("#991b1b", "#fef2f2"), "WAITING": ("#334155", "#f8fafc")}
                _d_bg, _d_fg = _d_palette.get(_decision, ("#334155", "#f8fafc"))
                self.decision_banner.configure(fg_color=_d_bg, text_color=_d_fg, text=_decision)
                self.decision_subtitle.configure(
                    text=(
                        "Banner menampilkan hasil committed terakhir."
                        if _committed_val
                        else "Belum ada event committed. Banner mengikuti hasil live terbaru."
                    )
                )
                # Update reject breakdown (scrollable area)
                _breakdown = (payload.get("counters") or {}).get("session_reject_breakdown") or {}
                for _key, _lbl in self.breakdown_labels.items():
                    _lbl.configure(text=str(_breakdown.get(_key, 0)))
                self._sync_recent_events(payload)
                self._refresh_context_summary()
                self._update_status_badges(payload)
                display_frame = frame
                timings = payload.get("timings") or {}
                total_ms = timings.get("total_ms")
                inference_ms = timings.get("inference_ms")
                client_timings = payload.get("client_timings") or {}
                timing_suffix = ""
                if isinstance(total_ms, (int, float)):
                    timing_suffix = f" | backend={float(total_ms):.0f}ms"
                    if isinstance(inference_ms, (int, float)):
                        timing_suffix += f" infer={float(inference_ms):.0f}ms"
                enc_ms = client_timings.get("encode_ms")
                req_ms = client_timings.get("request_ms")
                if isinstance(enc_ms, (int, float)):
                    timing_suffix += f" enc={float(enc_ms):.0f}ms"
                if isinstance(req_ms, (int, float)):
                    timing_suffix += f" req={float(req_ms):.0f}ms"
                # Always prefer live camera frame for part_ready preview (fast, no decode overhead).
                if display_frame is not None:
                    local_part_ready = self._crop_local_roi(display_frame, self._read_roi_payload("part_ready"))
                    if local_part_ready is not None:
                        self.part_ready_preview.update_bgr(local_part_ready)
                elif payload.get("part_ready_preview_image_b64"):
                    self.part_ready_preview.update_b64(payload.get("part_ready_preview_image_b64"))
                if payload.get("overlay_image_b64"):
                    self.main_view.update_b64(payload.get("overlay_image_b64"))
                    self.display_source.set("Right View: ML Overlay (backend)")
                elif display_frame is not None and payload.get("sticker_roi_meta"):
                    # Stream mode: render detection boxes locally (no backend overlay round-trip).
                    local_overlay = self._build_local_detection_overlay(display_frame, payload)
                    if local_overlay is not None:
                        self.main_view.update_bgr(local_overlay)
                        self.display_source.set("Right View: ML Overlay (local)")
                elif display_frame is not None:
                    local_scene = self._build_full_frame_with_roi(display_frame, "sticker", label="Sticker ROI", color=(255, 214, 10))
                    if local_scene is not None:
                        self.main_view.update_bgr(local_scene)
                        self.display_source.set("Right View: Live Camera + Sticker ROI")
                if payload.get("count_committed"):
                    committed = payload.get("last_committed_result") or {}
                    validation = committed.get("validation") or {}
                    part_ready = committed.get("part_ready") or {}
                    detection = committed.get("sticker_detection") or {}
                    self.info_var.set(
                        f"Committed {validation.get('decision')} | "
                        f"Part gate={'READY' if part_ready.get('part_ready') else 'BLOCK'} | "
                        f"{validation.get('part_name') or '-'} | "
                        f"{validation.get('reject_reason_code') or 'OK'} | "
                        f"backend={detection.get('backend') or '-'} raw={detection.get('raw_detection_count') if detection.get('raw_detection_count') is not None else '-'}"
                        f"{timing_suffix}."
                    )
                else:
                    live_validation = payload.get("validation") or {}
                    live_part_ready = payload.get("part_ready") or {}
                    live_detection = payload.get("sticker_detection") or {}
                    self.info_var.set(
                        f"Gate={'READY' if live_part_ready.get('part_ready') else 'BLOCK'} "
                        f"(ratio {live_part_ready.get('match_ratio') if live_part_ready.get('match_ratio') is not None else '-'}) | "
                        f"Live decision: {live_validation.get('decision') or '-'} | "
                        f"Detected: {live_validation.get('detected_class') or '-'} | "
                        f"Reject: {live_validation.get('reject_reason_code') or 'OK'} | "
                        f"backend={live_detection.get('backend') or '-'} raw={live_detection.get('raw_detection_count') if live_detection.get('raw_detection_count') is not None else '-'}"
                        f"{timing_suffix}"
                    )
            elif error:
                self.state.latest_error = error
                if self._is_auth_error(error):
                    self.uploader.stop()
                    self.state.active_session = None
                    self.state.cache["part_ready"] = None
                    self.state.cache["sticker_detection"] = None
                    self._refresh_context_summary()
                    if not self._auth_error_notified:
                        self._auth_error_notified = True
                        messagebox.showwarning("Session", "Akses sesi ditolak (401). Silakan login ulang.")
                    self.info_var.set("Session dihentikan karena otorisasi gagal (401). Silakan login ulang.")
                    with self._lock:
                        self._latest_error = None
                self._update_status_badges()
                if not self._is_auth_error(error):
                    self.info_var.set(f"Frame pump error: {error}")
                if frame is not None:
                    self._update_local_roi_previews(frame)
            else:
                self._update_status_badges()
                if frame is not None:
                    self._update_local_roi_previews(frame)
                elif self.state.active_session:
                    self.info_var.set("Session aktif — menunggu frame kamera.")
        except tk.TclError as exc:
            self.state.latest_error = str(exc)
            self.info_var.set(f"UI render warning: {exc}")
            self._update_status_badges()
        finally:
            self._schedule_poll()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._heartbeat_after_id:
            try:
                self.after_cancel(self._heartbeat_after_id)
            except tk.TclError:
                pass
            self._heartbeat_after_id = None
        if self._plc_poll_after_id:
            try:
                self.after_cancel(self._plc_poll_after_id)
            except tk.TclError:
                pass
            self._plc_poll_after_id = None
        if self._auto_start_after_id:
            try:
                self.after_cancel(self._auto_start_after_id)
            except tk.TclError:
                pass
            self._auto_start_after_id = None
        if hasattr(self, "sidebar_canvas"):
            self.sidebar_canvas.unbind_all("<MouseWheel>")
        self._close_settings()
        if self._plc_release_window and self._plc_release_window.winfo_exists():
            self._plc_release_window.destroy()
        self._stop_session()
        self.capture.stop()

    def destroy(self) -> None:
        self.shutdown()
        super().destroy()

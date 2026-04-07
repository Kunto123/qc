from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import cv2

from client_tk.app.components.counter_panel import CounterPanel
from client_tk.app.components.live_view import LiveView
from client_tk.app.components.result_panel import ResultPanel
from client_tk.app.config import DEFAULT_UPLOAD_INTERVAL_MS
from client_tk.app.services.camera_capture import CameraCaptureService
from client_tk.app.services.frame_upload import FrameUploadService


BADGE_COLORS = {
    "neutral": ("#475569", "#f8fafc"),
    "success": ("#166534", "#f0fdf4"),
    "danger": ("#991b1b", "#fef2f2"),
    "warning": ("#b45309", "#fffbeb"),
    "info": ("#1d4ed8", "#eff6ff"),
}
RESPONSIVE_BREAKPOINT = 1240


class OperatorScreen(ttk.Frame):
    def __init__(self, master, api_client, session_state):
        super().__init__(master, padding=10)
        self.api = api_client
        self.state = session_state
        self.capture = CameraCaptureService()
        self.uploader = FrameUploadService()
        self._latest_payload: dict | None = None
        self._latest_error: str | None = None
        self._lock = threading.Lock()
        self._after_id: str | None = None
        self._closed = False
        self._settings_window: tk.Toplevel | None = None
        self._template_lookup: dict[str, dict] = {}
        self._template_detail_lookup: dict[int, dict] = {}
        self._is_compact_layout: bool | None = None
        self._is_preview_compact: bool | None = None

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

    def _build_top_bar(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)

        action_bar = ttk.Frame(frame)
        action_bar.grid(row=0, column=0, sticky="w")
        ttk.Button(action_bar, text="\u2699 Settings", command=self._open_settings).pack(side="left", padx=(0, 6))
        ttk.Button(action_bar, text="Load Deployment", command=self._load_deployment).pack(side="left", padx=6)
        ttk.Button(action_bar, text="Start Camera", command=self._start_camera).pack(side="left", padx=6)
        ttk.Button(action_bar, text="Stop Camera", command=self._stop_camera).pack(side="left", padx=6)
        ttk.Button(action_bar, text="Start Session", command=self._start_session).pack(side="left", padx=6)
        ttk.Button(action_bar, text="Stop Session", command=self._stop_session).pack(side="left", padx=6)

        template_box = ttk.LabelFrame(frame, text="Template", padding=8)
        template_box.grid(row=0, column=1, sticky="e")
        self.template_selector = ttk.Combobox(template_box, textvariable=self.template_choice, width=32, state="readonly")
        self.template_selector.grid(row=0, column=0, padx=(0, 6))
        self.template_selector.bind("<<ComboboxSelected>>", self._on_template_selected)
        ttk.Button(template_box, text="Refresh", command=self._load_template_choices).grid(row=0, column=1)

    def _build_context_bar(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        for index in range(4):
            frame.columnconfigure(index, weight=1)
        ttk.Label(frame, textvariable=self.operator_context, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.line_context).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(frame, textvariable=self.station_context).grid(row=0, column=2, sticky="w", padx=8)
        ttk.Label(frame, textvariable=self.template_context).grid(row=0, column=3, sticky="w", padx=8)

    def _build_status_strip(self) -> None:
        frame = ttk.LabelFrame(self, text="System Status", padding=8)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.badges: dict[str, tk.Label] = {}
        for key in ("SERVER", "CAMERA", "SESSION", "DB", "EVENT"):
            label = tk.Label(
                frame,
                text=f"{key}: -",
                bg=BADGE_COLORS["neutral"][0],
                fg=BADGE_COLORS["neutral"][1],
                font=("Segoe UI", 10, "bold"),
                padx=10,
                pady=6,
            )
            label.pack(side="left", padx=4, pady=2)
            self.badges[key] = label

    def _build_content(self) -> None:
        self.content = ttk.Frame(self)
        self.content.grid(row=3, column=0, sticky="nsew")
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
        self.sidebar_container.rowconfigure(1, weight=1)  # scroll: fills rest

        # Fixed counter strip — always visible, outside scroll area
        self.counter_panel = CounterPanel(self.sidebar_container)
        self.counter_panel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        self.sidebar_canvas = tk.Canvas(
            self.sidebar_container,
            highlightthickness=0,
            bg=self._resolve_canvas_background(),
        )
        self.sidebar_scrollbar = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_canvas.grid(row=1, column=0, sticky="nsew")
        self.sidebar_scrollbar.grid(row=1, column=1, sticky="ns")

        self.sidebar_inner = ttk.Frame(self.sidebar_canvas)
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar_inner, anchor="nw")

        self.sidebar_inner.columnconfigure(0, weight=1)
        self.sidebar_inner.bind("<Configure>", self._sync_sidebar_scroll)
        self.sidebar_canvas.bind("<Configure>", self._resize_sidebar_inner)
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.result_panel = ResultPanel(self.sidebar_inner)
        self.result_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        events = ttk.LabelFrame(self.sidebar_inner, text="Recent Events", padding=8)
        events.grid(row=1, column=0, sticky="nsew")
        self.sidebar_inner.rowconfigure(1, weight=1)
        self.recent_list = tk.Listbox(events, height=12)
        self.recent_list.pack(fill="both", expand=True)

    def _resolve_canvas_background(self) -> str:
        style = ttk.Style(self)
        background = style.lookup("TFrame", "background")
        if background:
            return str(background)
        root_bg = self.winfo_toplevel().cget("bg")
        return str(root_bg or "#f0f0f0")

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
            self.part_ready_preview.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
            self.main_view.grid(row=1, column=0, sticky="nsew")
        else:
            self.preview_strip.rowconfigure(0, weight=1)
            self.preview_strip.rowconfigure(1, weight=0)
            self.preview_strip.columnconfigure(0, weight=2)
            self.preview_strip.columnconfigure(1, weight=3)
            self.part_ready_preview.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            self.main_view.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    def _on_resize(self, _event=None) -> None:
        self.after_idle(self._apply_responsive_layout)

    def _open_settings(self) -> None:
        if self._settings_window and self._settings_window.winfo_exists():
            self._settings_window.lift()
            self._settings_window.focus_force()
            return

        window = tk.Toplevel(self)
        window.title("Operator Settings")
        window.geometry("660x430")
        window.transient(self.winfo_toplevel())
        window.resizable(True, False)
        self._settings_window = window
        window.protocol("WM_DELETE_WINDOW", self._close_settings)

        body = ttk.Frame(window, padding=14)
        body.pack(fill="both", expand=True)
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

        footer = ttk.Frame(body)
        footer.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(14, 0))
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
        for item in items:
            label = f"{item['name']} | v{item.get('version_number')} | version_id={item.get('version_id')}"
            values.append(label)
            self._template_lookup[label] = item
        self.template_selector.configure(values=values)
        self._sync_template_selector()

    def _fetch_template_detail(self, template_id: int) -> dict:
        template_id = int(template_id)
        cached = self._template_detail_lookup.get(template_id)
        if cached:
            return cached
        detail = self.api.get_template(template_id)
        self._template_detail_lookup[template_id] = detail
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

    def _apply_template_detail(self, detail: dict | None) -> None:
        if not detail:
            return
        version_id = detail.get("version_id") or detail.get("current_version_id")
        if version_id:
            self.template_version_value.set(str(version_id))
        self._set_roi_values("part_ready", detail.get("part_ready_roi") or detail.get("roi") or {})
        self._set_roi_values("sticker", detail.get("sticker_roi") or detail.get("roi") or {})
        camera_config = detail.get("camera") or {}
        if camera_config.get("camera_index") is not None:
            self.camera_value.set(str(camera_config["camera_index"]))
        sticker_config = detail.get("sticker") or {}
        if sticker_config.get("line"):
            self.line_value.set(str(sticker_config["line"]))
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
        self.state.active_deployment = None
        try:
            detail = self._fetch_template_detail(int(item["id"]))
        except Exception as exc:  # noqa: BLE001
            self.template_version_value.set(str(item.get("version_id") or ""))
            self.info_var.set(f"Failed to load template detail: {exc}")
        else:
            self._apply_template_detail(detail)
            self.info_var.set(f"Template selected manually: {item.get('name')} v{item.get('version_number')}")
        self._refresh_context_summary()

    def _selected_template_name(self) -> str | None:
        selected = self._template_lookup.get(self.template_choice.get().strip())
        if not selected:
            return None
        return str(selected.get("name") or "").strip() or None

    def _set_badge(self, key: str, value: str, tone: str = "neutral") -> None:
        bg, fg = BADGE_COLORS.get(tone, BADGE_COLORS["neutral"])
        self.badges[key].configure(text=f"{key}: {value}", bg=bg, fg=fg)

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
        server_tone = "success" if self.state.token and not self.state.latest_error else "danger" if self.state.latest_error else "info"
        self._set_badge("SERVER", "ONLINE" if self.state.token and not self.state.latest_error else "ISSUE", server_tone)

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
        self.template_version_value.set(str(deployment.get("template_version_id") or ""))
        if deployment.get("template_id"):
            try:
                detail = self._fetch_template_detail(int(deployment["template_id"]))
            except Exception as exc:  # noqa: BLE001
                self.info_var.set(f"Deployment loaded, tapi detail template gagal dibaca: {exc}")
            else:
                self._apply_template_detail(detail)
        self.info_var.set(f"Deployment loaded: {deployment.get('template_name')}")
        self._refresh_context_summary()
        self._update_status_badges()

    def _start_camera(self) -> None:
        try:
            self.capture.start(int(self.camera_value.get() or 0))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Camera", str(exc))
            return
        self.info_var.set("Camera started. Menunggu frame pertama.")
        self._update_status_badges()

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
        with self._lock:
            self._latest_payload = None
            self._latest_error = None
        self.result_panel.reset()
        self.counter_panel.reset()
        self.recent_list.delete(0, "end")
        self.part_ready_preview.reset()
        self.main_view.reset()
        self.uploader.start(
            interval_ms=DEFAULT_UPLOAD_INTERVAL_MS,
            get_frame=self.capture.get_latest_frame,
            send_frame=lambda image_b64: self.api.push_frame(payload["session_id"], image_b64),
            on_result=self._set_result,
            on_error=self._set_error,
        )
        self.info_var.set(f"Session running: {payload['session_id']}")
        self._refresh_context_summary()
        self._update_status_badges()

    def _stop_session(self) -> None:
        if self.state.active_session:
            try:
                self.api.stop_session(self.state.active_session["session_id"])
            except Exception:  # noqa: BLE001
                pass
        self.uploader.stop()
        self.state.active_session = None
        self.state.cache["part_ready"] = None
        self.state.cache["sticker_detection"] = None
        self.info_var.set("Session stopped.")
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

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._latest_error = message

    def _schedule_poll(self) -> None:
        if self._closed:
            return
        self._after_id = self.after(100, self._poll_ui)

    def _poll_ui(self) -> None:
        if self._closed:
            return
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
            self._sync_recent_events(payload)
            self._refresh_context_summary()
            self._update_status_badges(payload)
            if payload.get("part_ready_preview_image_b64"):
                self.part_ready_preview.update_b64(payload.get("part_ready_preview_image_b64"))
            elif frame is not None:
                local_part_ready = self._crop_local_roi(frame, self._read_roi_payload("part_ready"))
                if local_part_ready is not None:
                    self.part_ready_preview.update_bgr(local_part_ready)
            if payload.get("overlay_image_b64"):
                self.main_view.update_b64(payload.get("overlay_image_b64"))
                self.display_source.set("Right View: Server ML Overlay")
            elif frame is not None:
                local_scene = self._build_full_frame_with_roi(frame, "sticker", label="Sticker ROI", color=(255, 214, 10))
                if local_scene is not None:
                    self.main_view.update_bgr(local_scene)
                    self.display_source.set("Right View: Live Camera + Sticker ROI")
            if payload.get("count_committed"):
                committed = payload.get("last_committed_result") or {}
                validation = committed.get("validation") or {}
                part_ready = committed.get("part_ready") or {}
                self.info_var.set(
                    f"Committed {validation.get('decision')} | "
                    f"Part gate={'READY' if part_ready.get('part_ready') else 'BLOCK'} | "
                    f"{validation.get('part_name') or '-'} | "
                    f"{validation.get('reject_reason_code') or 'OK'}."
                )
            else:
                live_validation = payload.get("validation") or {}
                live_part_ready = payload.get("part_ready") or {}
                self.info_var.set(
                    f"Gate={'READY' if live_part_ready.get('part_ready') else 'BLOCK'} "
                    f"(ratio {live_part_ready.get('match_ratio') if live_part_ready.get('match_ratio') is not None else '-'}) | "
                    f"Live decision: {live_validation.get('decision') or '-'} | "
                    f"Detected: {live_validation.get('detected_class') or '-'}"
                )
        elif error:
            self.state.latest_error = error
            self._update_status_badges()
            self.info_var.set(f"Upload error: {error}")
            if frame is not None:
                self._update_local_roi_previews(frame)
        else:
            self._update_status_badges()
            if frame is not None:
                self._update_local_roi_previews(frame)

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
        if hasattr(self, "sidebar_canvas"):
            self.sidebar_canvas.unbind_all("<MouseWheel>")
        self._close_settings()
        self._stop_session()
        self.capture.stop()

    def destroy(self) -> None:
        self.shutdown()
        super().destroy()

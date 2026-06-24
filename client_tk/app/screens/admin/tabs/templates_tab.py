"""Templates tab -- preset library, preset wizard, ROI picker, gap reference."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk

from client_tk.app.components.roi_picker_canvas import RoiPickerCanvas
from client_tk.app.components.scrollable_frame import ScrollableFrame


def _float_or_default(value, default):
    """Parse a string value to float, returning default if empty or invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    BORDER,
    PANEL_ALT_BG,
    PANEL_BG,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class TemplatesTab:
    """Preset library + preset wizard extracted from AdminScreen."""

    def __init__(self, admin, tab_frame):
        self.admin = admin
        self.frame = tab_frame
        self._build()

    # ------------------------------------------------------------------
    # Build
    def _build(self) -> None:
        a = self.admin
        self.frame.columnconfigure(0, weight=3)
        self.frame.columnconfigure(1, weight=2)
        self.frame.rowconfigure(0, weight=1)

        body = a._make_scrollable_body(self.frame, "Templates")

        a.presets_left = ttk.Frame(body, padding=8)
        a.presets_right = ttk.Frame(body, padding=8)
        a.presets_left.grid(row=0, column=0, sticky="nsew")
        a.presets_right.grid(row=0, column=1, sticky="nsew")

        self._build_library(a, a.presets_left)
        self._build_wizard(a, a.presets_right)

    def _build_library(self, a, parent) -> None:
        listing = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        listing.pack(fill="both", expand=True)

        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(2, weight=1)

        ctk.CTkLabel(listing, text="Preset Library", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 0),
        )
        ctk.CTkLabel(
            listing,
            text="Shows templates and active deployments together; ACTIVE marks deployed records.",
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))

        a.preset_table = a._build_table(
            listing,
            [
                ("id", "ID", 55, "center"),
                ("preset", "Preset", 220, "w"),
                ("version", "Version", 80, "center"),
                ("status", "Status", 90, "center"),
            ],
            row=2,
            height=18,
        )
        a.preset_table.bind("<<TreeviewSelect>>", a._on_preset_selected)

        footer = a._build_action_row(
            listing,
            [
                ("Refresh", a.refresh_presets, "neutral", "left"),
                ("New Preset", a.reset_preset_wizard, "neutral", "right"),
                ("Delete/Deactivate Selected", a.deactivate_selected_preset, "neutral", "right"),
            ],
        )
        footer.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 10))

    def _build_wizard(self, a, parent) -> None:
        wizard = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        wizard.pack(fill="both", expand=True)
        wizard.columnconfigure(1, weight=1)
        wizard.columnconfigure(3, weight=1)

        ctk.CTkLabel(wizard, text="Preset Wizard", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 0),
        )
        ctk.CTkLabel(
            wizard,
            text="Fill only production-critical values. Technical defaults are applied automatically.",
            text_color=TEXT_SECONDARY, wraplength=520, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=12, pady=(2, 10))

        # Mode selector
        mode_frame = ctk.CTkFrame(wizard, fg_color=PANEL_ALT_BG, corner_radius=8, border_width=1, border_color=BORDER)
        mode_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 8))
        ctk.CTkLabel(mode_frame, text="Validation Mode:", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=10, pady=6
        )
        ctk.CTkRadioButton(mode_frame, text="QC Sticker", variable=a.preset_validator_mode_var, value="sticker",
                           text_color=TEXT_PRIMARY, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="w", padx=(0, 10), pady=6)
        ctk.CTkRadioButton(mode_frame, text="Component Counter", variable=a.preset_validator_mode_var, value="component_count",
                           text_color=TEXT_PRIMARY, font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=(0, 10), pady=6)

        a._entry(wizard, 3, 0, "Preset Name", a.preset_name_var, columnspan=3)
        a._entry(wizard, 4, 0, "Description", a.preset_description_var, columnspan=3)

        a._entry(wizard, 5, 0, "Camera Index", a.preset_camera_index_var, columnspan=1)

        ttk.Label(wizard, text="Model").grid(row=6, column=0, sticky="w", padx=(12, 8), pady=5)
        a.preset_model_selector = ttk.Combobox(wizard, textvariable=a.preset_model_choice_var, state="readonly")
        a.preset_model_selector.grid(row=6, column=1, columnspan=3, sticky="ew", padx=(0, 12), pady=5)
        a.preset_model_selector.bind("<<ComboboxSelected>>", a._on_preset_model_selected)

        ttk.Label(wizard, text="Runtime").grid(row=7, column=0, sticky="w", padx=(12, 8), pady=5)
        runtime_combo = ttk.Combobox(
            wizard,
            textvariable=a.preset_runtime_var,
            values=["auto", "ultralytics", "tflite", "onnx", "openvino"],
            width=14,
            state="readonly",
        )
        runtime_combo.grid(row=7, column=1, columnspan=3, sticky="w", padx=(0, 12), pady=5)

        a._entry(wizard, 8, 0, "Confidence Threshold", a.preset_conf_threshold_var, columnspan=3)

        # Mean-Std threshold variables (initialized lazily on first method change)
        a.preset_mean_max_var = tk.StringVar(value="105.0")
        a.preset_std_max_var = tk.StringVar(value="35.0")

        # Sticker-specific fields (shown in sticker mode, hidden in component mode)
        a._sticker_fields_start = 9
        a._entry(wizard, 9, 0, "Expected Class", a.preset_expected_class_var, columnspan=3)

        a._entry(wizard, 10, 0, "Max Tilt Degrees", a.preset_max_tilt_var, columnspan=2)
        ttk.Checkbutton(wizard, text="Aktifkan cek miring", variable=a.preset_tilt_gate_var).grid(
            row=10, column=2, sticky="w", padx=(0, 12), pady=5,
        )

        a._entry(wizard, 11, 0, "Gap Threshold (0-1)", a.preset_gap_threshold_var, columnspan=2)
        # Track gap threshold widgets for show/hide based on method
        a._gap_threshold_widgets = []
        for _r in (11,):
            try:
                for w in wizard.grid_slaves(row=_r):
                    a._gap_threshold_widgets.append(w)
            except Exception:
                pass

        # Part-ready method selector
        ttk.Label(wizard, text="Part Ready Method").grid(row=12, column=0, sticky="w", padx=(12, 8), pady=5)
        a.preset_part_ready_method_var = tk.StringVar(value="gap_template_match")
        method_combo = ttk.Combobox(
            wizard,
            textvariable=a.preset_part_ready_method_var,
            values=["gap_template_match", "mean_std_threshold"],
            width=20,
            state="readonly",
        )
        method_combo.grid(row=12, column=1, columnspan=2, sticky="w", padx=(0, 12), pady=5)
        a.preset_part_ready_method_var.trace_add("write", lambda *_: self._on_part_ready_method_changed(a))
        a.preset_part_ready_method_var.trace_add("write", lambda *_: self._on_method_or_mode_changed(a))
        a.preset_validator_mode_var.trace_add("write", lambda *_: self._on_method_or_mode_changed(a))
        # Mean-Std threshold fields (shown only when method=mean_std_threshold)
        a._mean_std_fields_start = 13
        a._entry(wizard, 13, 0, "MEAN_MAX", a.preset_mean_max_var, columnspan=2)
        a._entry(wizard, 14, 0, "STD_MAX", a.preset_std_max_var, columnspan=2)
        a._mean_std_field_rows = [13, 14]

        # Reference patch buttons — shift down to row 15
        ref_btn_row = ttk.Frame(wizard)
        ref_btn_row.grid(row=15, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 4))
        ttk.Button(ref_btn_row, text="Capture Reference", command=a._capture_part_ready_ref).pack(side="left", padx=(0, 6))
        ttk.Button(ref_btn_row, text="Upload Reference", command=a._upload_part_ready_ref).pack(side="left")
        a.gap_ref_status_label = ttk.Label(wizard, text="Referensi: belum dikonfigurasi", foreground="gray")
        a.gap_ref_status_label.grid(row=16, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 6))

        ttk.Label(wizard, text="Rotation\\xB0\\n(0/90/180/270)", foreground="gray").grid(
            row=11, column=2, sticky="w", padx=(12, 4), pady=5,
        )
        rot_entry = ttk.Entry(wizard, textvariable=a.preset_camera_rotation_var, width=8)
        rot_entry.grid(row=11, column=3, sticky="w", padx=(0, 12), pady=5)

        # Visual ROI picker
        self._build_roi_picker(a, wizard)

        # Component ROI editor (only visible in component_count mode)
        self._build_component_roi_editor(a, wizard)

        # Action buttons
        btn_row = ttk.Frame(wizard)
        btn_row.grid(row=22, column=0, columnspan=4, sticky="ew", padx=12, pady=(16, 6))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(0, weight=1)

        a._preset_action_btn = ctk.CTkButton(
            btn_row,
            text="Save & Deploy Preset",
            command=a.save_and_deploy_preset,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            height=34,
            corner_radius=6,
        )
        a._preset_action_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            btn_row,
            text="Export template.json",
            command=a.export_runtime_template,
            fg_color=PANEL_ALT_BG,
            hover_color=BORDER,
            text_color=TEXT_PRIMARY,
            height=34,
            corner_radius=6,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    # Keep references to sticker field widgets for show/hide
        a._sticker_field_widgets = []
        for _r in range(9, 13):
            try:
                for w in wizard.grid_slaves(row=_r):
                    a._sticker_field_widgets.append(w)
            except Exception:
                pass

        # Keep references to part ready reference widgets (capture/upload ref, status label)
        # These are hidden in component_counter mode since part ready uses Modbus sensor only
        # Now at rows 15 (ref buttons) and 16 (status label) — shifted for method selector
        a._part_ready_ref_widgets = []
        for _r in (15, 16):
            try:
                for w in wizard.grid_slaves(row=_r):
                    a._part_ready_ref_widgets.append(w)
            except Exception:
                pass

        # Keep references to mean-std threshold fields (shown only when method=mean_std_threshold)
        a._mean_std_field_widgets = []
        for _r in (13, 14):
            try:
                for w in wizard.grid_slaves(row=_r):
                    a._mean_std_field_widgets.append(w)
            except Exception:
                pass

        # Initial mode sync
        a.preset_validator_mode_var.trace_add("write", lambda *_: self._on_mode_changed(a))

    def _on_method_or_mode_changed(self, a) -> None:
        """Show/hide fields based on method and mode."""
        method = a.preset_part_ready_method_var.get()
        show_mean_std = (method == "mean_std_threshold")
        # Show/hide mean_std threshold fields
        for w in getattr(a, "_mean_std_field_widgets", []):
            w.grid() if show_mean_std else w.grid_remove()
        # Show/hide gap threshold field (only for gap_template_match)
        show_gap = (method == "gap_template_match")
        for w in getattr(a, "_gap_threshold_widgets", []):
            w.grid() if show_gap else w.grid_remove()
        # Show/hide calibration section based on method
        if show_mean_std:
            a._calib_mean_std_frame.grid()
        else:
            a._calib_mean_std_frame.grid_remove()

    def _on_part_ready_method_changed(self, a) -> None:
        """Show/hide mean-std threshold fields based on part ready method."""
        method = a.preset_part_ready_method_var.get()
        show_mean_std = (method == "mean_std_threshold")
        for w in getattr(a, "_mean_std_field_widgets", []):
            if show_mean_std:
                w.grid()
            else:
                w.grid_remove()
        # Also show/hide reference buttons & gap threshold based on method
        show_ref = (method == "gap_template_match")
        for w in getattr(a, "_part_ready_ref_widgets", []):
            if show_ref:
                w.grid()
            else:
                w.grid_remove()
        # Show/hide gap threshold entry (row 11) — only relevant for gap_template_match
        for w in getattr(a, "_gap_threshold_widgets", []):
            if show_ref:
                w.grid()
            else:
                w.grid_remove()

    def _on_mode_changed(self, a) -> None:
        """Show/hide fields based on validation mode."""
        mode = a.preset_validator_mode_var.get()
        # Show/hide sticker fields
        for w in getattr(a, "_sticker_field_widgets", []):
            if mode == "component_count":
                w.grid_remove()
            else:
                w.grid()
        # Show/hide component editor
        if mode == "component_count":
            a._comp_editor_frame.grid()
            a._add_comp_roi_btn.grid()
        else:
            a._comp_editor_frame.grid_remove()
            a._add_comp_roi_btn.grid_remove()
        # Show/hide sticker and part-ready ROI on canvas based on mode
        if hasattr(a, "preset_roi_picker"):
            a.preset_roi_picker.set_sticker_visible(mode != "component_count")
            a.preset_roi_picker.set_part_ready_visible(mode != "component_count")
        # ROI picker panel stays visible in both modes (used for component ROIs in counter mode)
        # But update its selector to show component ROIs vs sticker/part-ready ROIs
        self._update_roi_selector_dropdown(a)
        # Show/hide part ready reference buttons and mean-std fields
        # In component_count mode, part ready = Modbus sensor only -- hide all
        self._on_part_ready_method_changed(a)
        _is_component = (mode == "component_count")
        for w in getattr(a, "_part_ready_ref_widgets", []):
            w.grid_remove() if _is_component else w.grid()
        for w in getattr(a, "_mean_std_field_widgets", []):
            w.grid_remove() if _is_component else w.grid()

    # ------------------------------------------------------------------
    # ROI Picker
    def _build_roi_picker(self, a, wizard) -> None:
        roi_panel = ctk.CTkFrame(wizard, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        roi_panel.grid(row=17, column=0, columnspan=4, sticky="ew", padx=12, pady=(12, 2))
        roi_panel.columnconfigure(0, weight=1)
        a._roi_picker_panel = roi_panel

        ctk.CTkLabel(roi_panel, text="Visual ROI Picker", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 4),
        )

        roi_toolbar = ctk.CTkFrame(roi_panel, fg_color="transparent")
        roi_toolbar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        roi_toolbar.columnconfigure(1, weight=1)

        ttk.Label(roi_toolbar, text="ROI").grid(row=0, column=0, sticky="w", padx=(0, 6))
        a.preset_roi_selector = ttk.Combobox(
            roi_toolbar,
            textvariable=a.preset_roi_choice_var,
            values=["Part Ready ROI", "Sticker ROI"],
            state="readonly",
            width=18,
        )
        a.preset_roi_selector.grid(row=0, column=1, sticky="w", padx=(0, 8))
        a.preset_roi_selector.bind("<<ComboboxSelected>>", a._on_preset_roi_selected)

        a._add_comp_roi_btn = ttk.Button(roi_toolbar, text="+ Add ROI", command=lambda: self._on_add_comp_roi(a))
        a._add_comp_roi_btn.grid(row=0, column=2, padx=(0, 4))
        a._add_comp_roi_btn.grid_remove()

        ttk.Button(roi_toolbar, text="Pick Image", command=a._pick_preset_roi_image).grid(row=0, column=3, padx=(0, 4))
        a._live_cam_btn = ttk.Button(roi_toolbar, text="Start Live Camera", command=lambda: a._toggle_live_camera())
        a._live_cam_btn.grid(row=0, column=4, padx=(0, 4))
        ttk.Button(roi_toolbar, text="Reset", command=a._reset_preset_roi).grid(row=0, column=5)

        a.preset_roi_picker = RoiPickerCanvas(roi_panel, "Drag/resize selected ROI on the image", size=(520, 292))
        a.preset_roi_picker.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        a.preset_roi_picker.on_roi_changed = lambda kind, roi: self._on_comp_roi_picker_changed(a, kind, roi)
        rois = []
        for _cr in a.preset_component_rois:
            _roi = _cr.get("roi", {})
            _roi["name"] = _cr.get("name", "ROI")
            rois.append(_roi)
        a.preset_roi_picker.set_component_rois(rois)
        a._on_preset_roi_selected()

        # Mean Std Calibration section (collapsible)
        a._calib_mean_std_frame = ctk.CTkFrame(roi_panel, fg_color=PANEL_ALT_BG, corner_radius=6, border_width=1, border_color=BORDER)
        a._calib_mean_std_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        a._calib_mean_std_frame.columnconfigure(1, weight=1)

        calib_header = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        calib_header.grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(4, 2))
        ctk.CTkLabel(calib_header, text="Mean-Std Calibration", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w")

        # Step 1: Empty
        self._calib_step1_frame = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        self._calib_step1_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=8, pady=1)
        ctk.CTkButton(self._calib_step1_frame, text="1. Capture Empty (no part)", width=160, height=26,
                      command=lambda: self._calib_capture(a, "empty")).grid(row=0, column=0, padx=(0, 4))
        a._calib_empty_result = ctk.CTkLabel(self._calib_step1_frame, text="—", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a._calib_empty_result.grid(row=0, column=1, sticky="w")

        # Step 2: Part
        self._calib_step2_frame = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        self._calib_step2_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=1)
        ctk.CTkButton(self._calib_step2_frame, text="2. Capture Part (black)", width=160, height=26,
                      command=lambda: self._calib_capture(a, "part")).grid(row=0, column=0, padx=(0, 4))
        a._calib_part_result = ctk.CTkLabel(self._calib_step2_frame, text="—", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a._calib_part_result.grid(row=0, column=1, sticky="w")

        # Step 3: Sticker
        self._calib_step3_frame = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        self._calib_step3_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=8, pady=1)
        ctk.CTkButton(self._calib_step3_frame, text="3. Capture Sticker", width=160, height=26,
                      command=lambda: self._calib_capture(a, "sticker")).grid(row=0, column=0, padx=(0, 4))
        a._calib_sticker_result = ctk.CTkLabel(self._calib_step3_frame, text="—", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a._calib_sticker_result.grid(row=0, column=1, sticky="w")

        # Result
        self._calib_result_frame = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        self._calib_result_frame.grid(row=4, column=0, columnspan=4, sticky="ew", padx=8, pady=(4, 2))
        a._calib_computed = ctk.CTkLabel(self._calib_result_frame, text="Capture all 3 to compute thresholds", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a._calib_computed.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(self._calib_result_frame, text="Apply", width=60, height=24, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      text_color=TEXT_ON_ACCENT, command=lambda: self._calib_apply(a)).grid(row=0, column=3, sticky="e", padx=(8, 0))

        for var in (
            a.part_ready_roi_x_var,
            a.part_ready_roi_y_var,
            a.part_ready_roi_w_var,
            a.part_ready_roi_h_var,
            a.sticker_roi_x_var,
            a.sticker_roi_y_var,
            a.sticker_roi_w_var,
            a.sticker_roi_h_var,
        ):
            var.trace_add("write", lambda *_: a._sync_preset_roi_picker())

    # ------------------------------------------------------------------
    # Component ROI Editor
    def _build_component_roi_editor(self, a, wizard) -> None:
        a._comp_editor_frame = ctk.CTkFrame(wizard, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        a._comp_editor_frame.grid(row=19, column=0, columnspan=4, sticky="ew", padx=12, pady=(8, 4))
        a._comp_editor_frame.columnconfigure(0, weight=1)
        a._comp_editor_frame.grid_remove()

        header = ctk.CTkFrame(a._comp_editor_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        ctk.CTkLabel(header, text="Component ROIs", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkButton(header, text="+ Add ROI", command=lambda: self._on_add_comp_roi(a),
                       fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT,
                       font=("Segoe UI", 9, "bold"), width=80, height=28).pack(side="right")

        a._comp_roi_list_frame = ctk.CTkFrame(a._comp_editor_frame, fg_color="transparent")
        a._comp_roi_list_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        a._comp_roi_list_frame.columnconfigure(0, weight=1)

        a.preset_validator_mode_var.trace_add("write", lambda *_: self._refresh_comp_roi_editor(a))
        self._refresh_comp_roi_editor(a)

    def _on_add_comp_roi(self, a) -> None:
        idx = a.preset_roi_picker.add_component_roi(f"ROI {chr(65 + len(a.preset_component_rois))}")
        picker_rois = a.preset_roi_picker.get_component_rois()
        while len(a.preset_component_rois) < len(picker_rois):
            a.preset_component_rois.append({
                "name": "", "roi": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3},
                "classes": [], "strict_foreign_class": False,
            })
        for i, pr in enumerate(picker_rois):
            if i < len(a.preset_component_rois):
                a.preset_component_rois[i]["name"] = pr.get("name", "")
                a.preset_component_rois[i]["roi"] = {
                    "x": pr.get("x", 0.1), "y": pr.get("y", 0.1),
                    "w": pr.get("w", 0.3), "h": pr.get("h", 0.3),
                    "rotation": pr.get("rotation", 0.0),
                }
        # Add the single new ROI widget without destroying all existing ones
        self._build_single_comp_roi(a, a.preset_component_rois[idx], idx)
        self._update_roi_selector_dropdown(a)
        comp_name = a.preset_component_rois[idx].get("name", f"ROI {chr(65 + idx)}") if idx < len(a.preset_component_rois) else f"ROI {chr(65 + idx)}"
        a.preset_roi_choice_var.set(f"Component: {comp_name}")
        a.preset_roi_picker.set_active_roi(f"component:{idx}")

    def _on_remove_comp_roi(self, a, roi_idx: int) -> None:
        if roi_idx < len(a.preset_component_rois):
            a.preset_component_rois.pop(roi_idx)
            picker_rois = a.preset_roi_picker.get_component_rois()
            if roi_idx < len(picker_rois):
                a.preset_roi_picker.remove_component_roi(roi_idx)
            self._refresh_comp_roi_editor(a)
            self._update_roi_selector_dropdown(a)

    def _refresh_comp_roi_editor(self, a) -> None:
        mode = a.preset_validator_mode_var.get()
        if mode != "component_count":
            a._comp_editor_frame.grid_remove()
            a._add_comp_roi_btn.grid_remove()
            return
        a._comp_editor_frame.grid()
        a._add_comp_roi_btn.grid()
        # Sync data model to picker
        rois = []
        for _cr in a.preset_component_rois:
            _roi = _cr.get("roi", {})
            _roi["name"] = _cr.get("name", "ROI")
            rois.append(_roi)
        a.preset_roi_picker.set_component_rois(rois)
        # Only rebuild widgets if count changed
        existing_count = len(a._comp_roi_list_frame.winfo_children())
        if existing_count != len(a.preset_component_rois):
            for widget in a._comp_roi_list_frame.winfo_children():
                widget.destroy()
            for roi_idx, roi_data in enumerate(a.preset_component_rois):
                self._build_single_comp_roi(a, roi_data, roi_idx)
        self._update_roi_selector_dropdown(a)
        # Set active ROI to first component so user can immediately interact
        if a.preset_component_rois:
            a.preset_roi_picker.set_active_roi("component:0")

    def _build_single_comp_roi(self, a, roi_data: dict, roi_idx: int) -> None:
        row_frame = ctk.CTkFrame(a._comp_roi_list_frame, fg_color=PANEL_ALT_BG, corner_radius=6, border_width=1, border_color=BORDER)
        row_frame.grid(row=roi_idx, column=0, sticky="ew", pady=2)
        row_frame.columnconfigure(1, weight=1)
        name_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        name_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 2))
        name_var = tk.StringVar(value=roi_data.get("name", f"ROI {chr(65 + roi_idx)}"))
        ctk.CTkEntry(name_frame, textvariable=name_var, width=100, height=24).pack(side="left")
        name_var.trace_add("write", lambda *a2, idx=roi_idx, v=name_var: self._on_comp_roi_name_changed(idx, v))
        ctk.CTkButton(name_frame, text="✕", width=24, height=24, fg_color=BORDER, hover_color=ACCENT_HOVER,
                      command=lambda idx=roi_idx: self._on_remove_comp_roi(a, idx)).pack(side="right")
        cls_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        cls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
        ctk.CTkLabel(cls_frame, text="Class", font=("Segoe UI", 8, "bold"), text_color=TEXT_SECONDARY).grid(row=0, column=0, padx=2)
        ctk.CTkLabel(cls_frame, text="Count", font=("Segoe UI", 8, "bold"), text_color=TEXT_SECONDARY).grid(row=0, column=1, padx=2)
        classes = roi_data.get("classes", [])
        for cls_idx, cls_target in enumerate(classes):
            class_var = tk.StringVar(value=cls_target.get("class_name", ""))
            count_var = tk.StringVar(value=str(cls_target.get("count", 1)))
            _class_names = [c.strip() for c in a.preset_model_classes_var.get().split(",") if c.strip()] if hasattr(a, "preset_model_classes_var") else []
            if _class_names:
                ctk.CTkComboBox(cls_frame, variable=class_var, values=_class_names, width=100, height=24).grid(row=cls_idx+1, column=0, padx=2, pady=1)
            else:
                ctk.CTkEntry(cls_frame, textvariable=class_var, width=100, height=24).grid(row=cls_idx+1, column=0, padx=2, pady=1)
            ctk.CTkEntry(cls_frame, textvariable=count_var, width=50, height=24).grid(row=cls_idx+1, column=1, padx=2, pady=1)
            class_var.trace_add("write", lambda *a2, idx=roi_idx, cidx=cls_idx, v=class_var: self._on_comp_class_changed(idx, cidx, v))
            count_var.trace_add("write", lambda *a2, idx=roi_idx, cidx=cls_idx, v=count_var: self._on_comp_count_changed(idx, cidx, v))
        ctk.CTkButton(cls_frame, text="+", width=24, height=24, fg_color="transparent", hover_color=PANEL_BG,
                      command=lambda idx=roi_idx: self._on_add_comp_class(a, idx)).grid(row=len(classes)+1, column=0, pady=(2, 4))
        strict_var = tk.BooleanVar(value=roi_data.get("strict_foreign_class", False))
        ctk.CTkCheckBox(row_frame, text="Strict foreign class", variable=strict_var,
                        text_color=TEXT_SECONDARY, font=("Segoe UI", 8)).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4))
        strict_var.trace_add("write", lambda *a2, idx=roi_idx, v=strict_var: self._on_comp_strict_changed(idx, v))

    def _on_comp_roi_name_changed(self, roi_idx: int, var: tk.StringVar) -> None:
        if roi_idx < len(self.admin.preset_component_rois):
            self.admin.preset_component_rois[roi_idx]["name"] = var.get()
            # Sync name to canvas component ROIs so display name matches
            picker = getattr(self.admin, "preset_roi_picker", None)
            if picker is not None and roi_idx < len(picker._component_rois):
                picker._component_rois[roi_idx]["name"] = var.get()
                picker.redraw()
            # Sync dropdown text
            self._update_roi_selector_dropdown(self.admin)
            # Re-set active ROI to maintain selection
            kind = self.admin._preset_roi_kind()
            if kind and kind.startswith("component:"):
                self.admin.preset_roi_picker.set_active_roi(kind)

    def _on_comp_class_changed(self, roi_idx: int, cls_idx: int, var: tk.StringVar) -> None:
        if roi_idx < len(self.admin.preset_component_rois) and cls_idx < len(self.admin.preset_component_rois[roi_idx]["classes"]):
            self.admin.preset_component_rois[roi_idx]["classes"][cls_idx]["class_name"] = var.get()

    def _on_comp_count_changed(self, roi_idx: int, cls_idx: int, var: tk.StringVar) -> None:
        if roi_idx < len(self.admin.preset_component_rois) and cls_idx < len(self.admin.preset_component_rois[roi_idx]["classes"]):
            try:
                self.admin.preset_component_rois[roi_idx]["classes"][cls_idx]["count"] = int(var.get())
            except ValueError:
                self.admin.preset_component_rois[roi_idx]["classes"][cls_idx]["count"] = 1

    def _on_comp_strict_changed(self, roi_idx: int, var: tk.BooleanVar) -> None:
        if roi_idx < len(self.admin.preset_component_rois):
            self.admin.preset_component_rois[roi_idx]["strict_foreign_class"] = var.get()

    def _on_add_comp_class(self, a, roi_idx: int) -> None:
        if roi_idx < len(a.preset_component_rois):
            a.preset_component_rois[roi_idx]["classes"].append({"class_name": "", "count": 1})
            self._refresh_comp_roi_editor(a)

    def _update_roi_selector_dropdown(self, a) -> None:
        mode = a.preset_validator_mode_var.get()
        if mode == "component_count":
            values = []
            for i, cr in enumerate(a.preset_component_rois):
                name = cr.get("name", f"ROI {chr(65 + i)}")
                values.append(f"Component: {name}")
            # Only set choice to first if current choice is not in values
            current = a.preset_roi_choice_var.get()
            if current not in values and values:
                a.preset_roi_choice_var.set(values[0])
        else:
            values = ["Part Ready ROI", "Sticker ROI"]
            if a.preset_roi_choice_var.get() not in values:
                a.preset_roi_choice_var.set("Part Ready ROI")
        a.preset_roi_selector.configure(values=values)

    # ------------------------------------------------------------------
    # Logo Capture

    # ------------------------------------------------------------------
    # ROI changed callback
    def _on_comp_roi_picker_changed(self, a, kind: str, roi: dict) -> None:
        if kind.startswith("component:"):
            idx = int(kind.split(":")[1])
            if idx < len(a.preset_component_rois):
                a.preset_component_rois[idx]["roi"] = {
                    "x": roi.get("x", 0), "y": roi.get("y", 0),
                    "w": roi.get("w", 0), "h": roi.get("h", 0),
                    "rotation": roi.get("rotation", 0),
                }
        else:
            a._on_preset_roi_changed(kind, roi)

    # ------------------------------------------------------------------
    # Mean-Std Calibration
    # ------------------------------------------------------------------

    def _toggle_calib_mean_std(self, a) -> None:
        if self._calib_toggle_var.get():
            a._calib_mean_std_frame.grid()
        else:
            a._calib_mean_std_frame.grid_remove()

    def _calib_capture(self, a, step: str) -> None:
        """Capture a frame from camera and compute mean/std for the current ROI."""
        cam_idx = int(_float_or_default(a.preset_camera_index_var.get(), 0))
        try:
            from client_tk.app.services.camera_capture import CameraCaptureService
            cam = CameraCaptureService()
            cam.start(cam_idx)
            import time
            time.sleep(0.5)
            frame = cam.get_latest_frame()
            cam.stop()
            if frame is None:
                messagebox.showwarning("Calibration", "Camera returned no frame.")
                return
            # Apply camera rotation
            _rot = float(_float_or_default(a.preset_camera_rotation_var.get(), 0))
            if _rot != 0.0:
                import cv2
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -_rot, 1.0)
                cos_a = abs(M[0, 0])
                sin_a = abs(M[0, 1])
                new_w = int(h * sin_a + w * cos_a)
                new_h = int(h * cos_a + w * sin_a)
                M[0, 2] += (new_w - w) / 2
                M[1, 2] += (new_h - h) / 2
                frame = cv2.warpAffine(frame, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
        except Exception as exc:
            messagebox.showerror("Calibration", f"Failed to capture: {exc}")
            return

        # Crop to part_ready ROI
        roi_f = a._roi_payload_from_vars(
            a.part_ready_roi_x_var, a.part_ready_roi_y_var,
            a.part_ready_roi_w_var, a.part_ready_roi_h_var,
            defaults={"x": 0.2, "y": 0.2, "w": 0.25, "h": 0.25},
        )
        fh, fw = frame.shape[:2]
        x = max(0, int(roi_f["x"] * fw))
        y = max(0, int(roi_f["y"] * fh))
        w = max(1, int(roi_f["w"] * fw))
        h = max(1, int(roi_f["h"] * fh))
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        crop = frame[y:y2, x:x2]

        if crop.size == 0:
            messagebox.showwarning("Calibration", "ROI crop is empty. Check ROI position.")
            return

        import cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        mean_val = float(gray.mean())
        std_val = float(gray.std())

        if step == "empty":
            a._calib_empty_mean = mean_val
            a._calib_empty_result.configure(text=f"mean={mean_val:.1f}")
        elif step == "part":
            a._calib_part_mean = mean_val
            a._calib_part_std = std_val
            a._calib_part_result.configure(text=f"mean={mean_val:.1f}, std={std_val:.1f}")
        elif step == "sticker":
            a._calib_sticker_std = std_val
            a._calib_sticker_result.configure(text=f"std={std_val:.1f}")

        # Auto-compute if all 3 captured
        if a._calib_empty_mean > 0 and a._calib_part_mean > 0 and a._calib_sticker_std > 0:
            from backend.app.services.part_ready_detector import compute_mean_std_thresholds
            result = compute_mean_std_thresholds(
                a._calib_empty_mean, a._calib_part_mean,
                a._calib_part_std, a._calib_sticker_std,
            )
            a._calib_computed.configure(
                text=f"MEAN_MAX={result['mean_max']:.1f}, STD_MAX={result['std_max']:.1f} (gaps: mean={a._calib_empty_mean - a._calib_part_mean:.1f}, std={a._calib_sticker_std - a._calib_part_std:.1f})"
            )

    def _calib_apply(self, a) -> None:
        """Apply computed thresholds to the template."""
        if a._calib_empty_mean == 0 or a._calib_part_mean == 0 or a._calib_sticker_std == 0:
            messagebox.showwarning("Calibration", "Capture all 3 conditions first.")
            return
        from backend.app.services.part_ready_detector import compute_mean_std_thresholds
        result = compute_mean_std_thresholds(
            a._calib_empty_mean, a._calib_part_mean,
            a._calib_part_std, a._calib_sticker_std,
        )
        a.preset_mean_max_var.set(str(result["mean_max"]))
        a.preset_std_max_var.set(str(result["std_max"]))
        a._set_status(f"Applied mean_std thresholds: MEAN_MAX={result['mean_max']}, STD_MAX={result['std_max']}")

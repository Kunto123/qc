"""Templates tab -- preset library, preset wizard, ROI picker, gap reference."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

from client_tk.app.components.roi_picker_canvas import RoiPickerCanvas
from client_tk.app.components.scrollable_frame import ScrollableFrame
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

        # Sticker-specific fields (shown in sticker mode, hidden in component mode)
        a._sticker_fields_start = 9
        a._entry(wizard, 9, 0, "Expected Class", a.preset_expected_class_var, columnspan=3)
        a._entry(wizard, 10, 0, "Sticker Code", a.preset_expected_code_var, columnspan=3)

        a._entry(wizard, 11, 0, "Max Tilt Degrees", a.preset_max_tilt_var, columnspan=2)
        ttk.Checkbutton(wizard, text="Aktifkan cek miring", variable=a.preset_tilt_gate_var).grid(
            row=11, column=2, sticky="w", padx=(0, 12), pady=5,
        )

        a._entry(wizard, 12, 0, "Gap Threshold (0-1)", a.preset_gap_threshold_var, columnspan=2)

        # Reference patch buttons
        ref_btn_row = ttk.Frame(wizard)
        ref_btn_row.grid(row=13, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 4))
        ttk.Button(ref_btn_row, text="Capture Reference", command=a._capture_part_ready_ref).pack(side="left", padx=(0, 6))
        ttk.Button(ref_btn_row, text="Upload Reference", command=a._upload_part_ready_ref).pack(side="left")
        a.gap_ref_status_label = ttk.Label(wizard, text="Referensi: belum dikonfigurasi", foreground="gray")
        a.gap_ref_status_label.grid(row=14, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 6))

        ttk.Label(wizard, text="Rotation\\xB0\\n(0/90/180/270)", foreground="gray").grid(
            row=12, column=2, sticky="w", padx=(12, 4), pady=5,
        )
        rot_entry = ttk.Entry(wizard, textvariable=a.preset_camera_rotation_var, width=8)
        rot_entry.grid(row=12, column=3, sticky="w", padx=(0, 12), pady=5)

        # Visual ROI picker
        self._build_roi_picker(a, wizard)

        # Component ROI editor (only visible in component_count mode)
        self._build_component_roi_editor(a, wizard)

        # Logo reference capture (only visible in sticker mode)
        self._build_logo_capture(a, wizard)

        # Action buttons
        btn_row = ttk.Frame(wizard)
        btn_row.grid(row=21, column=0, columnspan=4, sticky="ew", padx=12, pady=(16, 6))
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
        a._part_ready_ref_widgets = []
        for _r in (13, 14):
            try:
                for w in wizard.grid_slaves(row=_r):
                    a._part_ready_ref_widgets.append(w)
            except Exception:
                pass

        # Initial mode sync
        a.preset_validator_mode_var.trace_add("write", lambda *_: self._on_mode_changed(a))

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
            a._logo_frame.grid_remove()
        else:
            a._comp_editor_frame.grid_remove()
            a._add_comp_roi_btn.grid_remove()
            a._logo_frame.grid()
        # Show/hide sticker and part-ready ROI on canvas based on mode
        if hasattr(a, "preset_roi_picker"):
            a.preset_roi_picker.set_sticker_visible(mode != "component_count")
            a.preset_roi_picker.set_part_ready_visible(mode != "component_count")
        # ROI picker panel stays visible in both modes (used for component ROIs in counter mode)
        # But update its selector to show component ROIs vs sticker/part-ready ROIs
        self._update_roi_selector_dropdown(a)
        # Show/hide part ready reference buttons (capture/upload ref, status label)
        # Not needed in component_counter mode since part ready = Modbus sensor only
        for w in getattr(a, "_part_ready_ref_widgets", []):
            if mode == "component_count":
                w.grid_remove()
            else:
                w.grid()

    # ------------------------------------------------------------------
    # ROI Picker
    def _build_roi_picker(self, a, wizard) -> None:
        roi_panel = ctk.CTkFrame(wizard, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        roi_panel.grid(row=16, column=0, columnspan=4, sticky="ew", padx=12, pady=(12, 2))
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
        ttk.Button(roi_toolbar, text="Capture from Camera", command=a._capture_preset_roi_from_camera).grid(row=0, column=4, padx=(0, 4))
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
        a._comp_editor_frame.grid(row=18, column=0, columnspan=4, sticky="ew", padx=12, pady=(8, 4))
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
        self._refresh_comp_roi_editor(a)
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
        rois = []
        for _cr in a.preset_component_rois:
            _roi = _cr.get("roi", {})
            _roi["name"] = _cr.get("name", "ROI")
            rois.append(_roi)
        a.preset_roi_picker.set_component_rois(rois)
        self._update_roi_selector_dropdown(a)
        for widget in a._comp_roi_list_frame.winfo_children():
            widget.destroy()
        for roi_idx, roi_data in enumerate(a.preset_component_rois):
            self._build_single_comp_roi(a, roi_data, roi_idx)

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
        values = ["Part Ready ROI"]
        mode = a.preset_validator_mode_var.get()
        if mode == "component_count":
            for i, cr in enumerate(a.preset_component_rois):
                name = cr.get("name", f"ROI {chr(65 + i)}")
                values.append(f"Component: {name}")
        else:
            values.append("Sticker ROI")
        a.preset_roi_selector.configure(values=values)

    # ------------------------------------------------------------------
    # Logo Capture
    def _build_logo_capture(self, a, wizard) -> None:
        a._logo_frame = ctk.CTkFrame(wizard, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        a._logo_frame.grid(row=19, column=0, columnspan=4, sticky="ew", padx=12, pady=(8, 4))
        a._logo_frame.columnconfigure(0, weight=1)

        logo_header = ctk.CTkFrame(a._logo_frame, fg_color="transparent")
        logo_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        ctk.CTkLabel(logo_header, text="Logo Reference", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkButton(logo_header, text="📸 Capture Logo", command=lambda: self._on_capture_logo(a),
                       fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT,
                       font=("Segoe UI", 9, "bold"), width=120, height=28).pack(side="right")

        a._logo_status_label = ctk.CTkLabel(a._logo_frame, text="No logo reference captured", text_color=TEXT_SECONDARY, font=("Segoe UI", 8))
        a._logo_status_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))

        a._logo_editor_frame = ctk.CTkFrame(a._logo_frame, fg_color="transparent")
        a._logo_editor_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        a._logo_editor_frame.grid_remove()

        _logo_fields = ctk.CTkFrame(a._logo_editor_frame, fg_color="transparent")
        _logo_fields.grid(row=0, column=0, sticky="ew")
        _logo_fields.columnconfigure(1, weight=1)
        _logo_fields.columnconfigure(3, weight=1)
        ctk.CTkLabel(_logo_fields, text="X:", text_color=TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=0, column=0, padx=(0, 4))
        a._logo_x_var = tk.StringVar(value="0.0")
        ctk.CTkEntry(_logo_fields, textvariable=a._logo_x_var, width=60, height=24).grid(row=0, column=1, padx=(0, 8))
        ctk.CTkLabel(_logo_fields, text="Y:", text_color=TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=0, column=2, padx=(0, 4))
        a._logo_y_var = tk.StringVar(value="0.0")
        ctk.CTkEntry(_logo_fields, textvariable=a._logo_y_var, width=60, height=24).grid(row=0, column=3, padx=(0, 8))
        ctk.CTkLabel(_logo_fields, text="W:", text_color=TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=1, column=0, padx=(0, 4), pady=(4, 0))
        a._logo_w_var = tk.StringVar(value="0.1")
        ctk.CTkEntry(_logo_fields, textvariable=a._logo_w_var, width=60, height=24).grid(row=1, column=1, padx=(0, 8), pady=(4, 0))
        ctk.CTkLabel(_logo_fields, text="H:", text_color=TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=1, column=2, padx=(0, 4), pady=(4, 0))
        a._logo_h_var = tk.StringVar(value="0.1")
        ctk.CTkEntry(_logo_fields, textvariable=a._logo_h_var, width=60, height=24).grid(row=1, column=3, padx=(0, 8), pady=(4, 0))
        _logo_btn_row = ctk.CTkFrame(a._logo_editor_frame, fg_color="transparent")
        _logo_btn_row.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ctk.CTkButton(_logo_btn_row, text="Save Logo", command=lambda: self._on_save_logo(a),
                       fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT,
                       font=("Segoe UI", 9, "bold"), width=80, height=28).pack(side="left", padx=(0, 8))
        ctk.CTkButton(_logo_btn_row, text="Cancel", command=lambda: self._on_cancel_logo(a),
                       fg_color=BORDER, hover_color=PANEL_ALT_BG, text_color=TEXT_PRIMARY,
                       font=("Segoe UI", 9), width=60, height=28).pack(side="left")

    def _on_capture_logo(self, a) -> None:
        frame = a._get_current_frame_for_logo()
        if frame is None:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Logo Capture", "No frame available. Please load an image first.")
            return
        a._logo_editor_frame.grid()
        a._logo_status_label.configure(text="Draw logo area and click Save", text_color=TEXT_PRIMARY)

    def _on_save_logo(self, a) -> None:
        try:
            import cv2, numpy as np
            from pathlib import Path
            from backend.app.core.config import PROJECT_ROOT as project_root
            x, y = float(a._logo_x_var.get()), float(a._logo_y_var.get())
            w, h = float(a._logo_w_var.get()), float(a._logo_h_var.get())
            frame = a._get_current_frame_for_logo()
            if frame is None:
                return
            fh, fw = frame.shape[:2]
            rx, ry = int(x * fw), int(y * fh)
            rw, rh = int(w * fw), int(h * fh)
            roi_frame = frame[ry:ry+rh, rx:rx+rw]
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = _clahe.apply(gray)
            median = float(np.median(gray))
            low, high = int(max(0, 0.5 * median)), int(min(255, 1.5 * median))
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edge_map = cv2.Canny(blurred, low, high)
            ref_dir = Path(project_root) / "backend/app/assets/part_ready_refs"
            ref_dir.mkdir(parents=True, exist_ok=True)
            template_id = a.current_template_id or 0
            save_path = ref_dir / f"{template_id}_logo.png"
            cv2.imwrite(str(save_path), edge_map)
            a._logo_editor_frame.grid_remove()
            a._logo_status_label.configure(text=f"Logo saved: {save_path.name}", text_color=TEXT_PRIMARY)
        except Exception as exc:
            import tkinter.messagebox as msgbox
            msgbox.showerror("Logo Capture", f"Failed to save logo: {exc}")

    def _on_cancel_logo(self, a) -> None:
        a._logo_editor_frame.grid_remove()
        a._logo_status_label.configure(text="Logo capture cancelled", text_color=TEXT_SECONDARY)

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

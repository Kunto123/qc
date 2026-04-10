from __future__ import annotations

import base64
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from backend.app.services.calibration import CalibrationService
from backend.app.core.model_catalog import list_base_models as catalog_list_base_models
from client_tk.app.components.annotation_canvas import AnnotationCanvas
from client_tk.app.components.live_view import LiveView
from client_tk.app.components.roi_picker_canvas import RoiPickerCanvas
from client_tk.app.components.scrollable_frame import ScrollableFrame
from client_tk.app.components.template_forms import JsonEditor, LabeledValuePanel


class EngineerScreen(ttk.Frame):
    def __init__(self, master, api_client, session_state):
        super().__init__(master, padding=8)
        self.api = api_client
        self.state = session_state
        self.upload_path: str | None = None
        self.upload_paths: list[str] = []
        self.selected_calibration_path: str | None = None
        self.computed_profile: dict | None = None
        self.calibration_image = None
        self._dataset_cache: list[dict] = []        
        self._base_model_cache: list[dict] = []
        self._base_model_lookup: dict[str, dict] = {}
        self._dataset_version_cache: list[dict] = []
        self._dataset_version_lookup: dict[str, dict] = {}
        self._training_jobs: list[dict] = []
        self._active_training_job_id: str | None = None
        self._model_cache: list[dict] = []
        self._profile_cache: list[dict] = []
        self._annotation_files: list[dict] = []
        self._annotation_index: int | None = None
        self._annotation_dataset_id: str | None = None
        self._annotation_dataset_name: str = ""
        self._annotation_class_name: str = "object"
        self._annotation_class_options: list[str] = ["object"]
        self._annotation_manual_classes: list[str] = []
        self._annotation_selected_label_index: int | None = None
        self._active_dataset_version_id: str | None = None
        self._tab_scrollers: dict[str, ScrollableFrame] = {}
        self._layout_compact: bool | None = None
        self._ignore_next_dataset_list_selection_event: bool = False

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.data_tab = ttk.Frame(notebook)
        self.training_tab = ttk.Frame(notebook)
        self.models_tab = ttk.Frame(notebook)
        self.calibration_tab = ttk.Frame(notebook)

        notebook.add(self.data_tab, text="Data")
        notebook.add(self.training_tab, text="Training")
        notebook.add(self.models_tab, text="Models")
        notebook.add(self.calibration_tab, text="Calibration")

        self.data_tab = self._make_scrollable_page(self.data_tab, "data")
        self.training_tab = self._make_scrollable_page(self.training_tab, "training")
        self.models_tab = self._make_scrollable_page(self.models_tab, "models")

        self._build_data_tab()
        self._build_training_tab()
        self._build_models_tab()
        self._build_calibration_tab()

        self.bind("<Configure>", self._on_resize)
        self.after_idle(self._apply_responsive_layout)

        self.refresh_datasets()
        self.refresh_base_models()
        self.refresh_augment_jobs()
        self.refresh_training_jobs()
        self.refresh_models()
        self.refresh_profiles()

    def _make_scrollable_page(self, tab: ttk.Frame, key: str) -> ttk.Frame:
        scroller = ScrollableFrame(tab)
        scroller.pack(fill="both", expand=True)
        self._tab_scrollers[key] = scroller
        scroller.body.columnconfigure(0, weight=1)
        return scroller.body

    def _layout_split_shell(
        self,
        shell: ttk.Frame,
        left: ttk.Frame,
        right: ttk.Frame,
        *,
        compact: bool,
        left_weight: int,
        right_weight: int,
    ) -> None:
        for widget in shell.grid_slaves():
            widget.grid_forget()

        if compact:
            shell.columnconfigure(0, weight=1)
            shell.rowconfigure(0, weight=1)
            shell.rowconfigure(1, weight=1)
            left.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
            right.grid(row=1, column=0, sticky="nsew")
        else:
            shell.columnconfigure(0, weight=left_weight)
            shell.columnconfigure(1, weight=right_weight)
            shell.rowconfigure(0, weight=1)
            left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            right.grid(row=0, column=1, sticky="nsew")

    def _on_resize(self, _event=None) -> None:
        self.after_idle(self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        width = max(self.winfo_width(), self.winfo_toplevel().winfo_width())
        compact = width < 1360
        if compact == self._layout_compact:
            return
        self._layout_compact = compact

        self._layout_split_shell(self.data_top_container, self.dataset_panel, self.upload_panel, compact=compact, left_weight=2, right_weight=2)
        self._layout_split_shell(self.training_container, self.augment_panel, self.train_panel, compact=compact, left_weight=1, right_weight=2)
        self._layout_split_shell(self.training_lower, self.training_jobs_panel, self.training_detail_panel, compact=compact, left_weight=2, right_weight=3)
        self._layout_split_shell(self.models_container, self.models_left_panel, self.models_right_panel, compact=compact, left_weight=2, right_weight=3)
        self._layout_split_shell(self.calibration_container, self.calibration_left_panel, self.calibration_right_outer, compact=compact, left_weight=2, right_weight=2)
        self._layout_data_annotation(compact=compact)

    def _layout_data_annotation(self, *, compact: bool) -> None:
        self.data_annotation_shell.columnconfigure(0, weight=1)
        self.data_annotation_shell.columnconfigure(1, weight=1)
        self.data_annotation_shell.rowconfigure(2, weight=1)
        if hasattr(self, "annotation_canvas"):
            self.annotation_canvas.request_redraw()

    def _build_data_tab(self) -> None:
        self.data_container = ttk.Frame(self.data_tab)
        self.data_container.pack(fill="both", expand=True, padx=6, pady=6)
        self.data_container.columnconfigure(0, weight=1)
        self.data_container.rowconfigure(0, weight=3)
        self.data_container.rowconfigure(1, weight=2)

        self.data_top_container = ttk.Frame(self.data_container)
        self.data_annotation_shell = ttk.Frame(self.data_container, padding=8)
        self.data_top_container.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.data_annotation_shell.grid(row=1, column=0, sticky="nsew")

        self.dataset_panel = ttk.Frame(self.data_top_container, padding=8)
        self.upload_panel = ttk.Frame(self.data_top_container, padding=8)

        ttk.Label(self.dataset_panel, text="Datasets", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            self.dataset_panel,
            text="Pilih dataset untuk sinkron ke upload, annotation, augment, dan training.",
            foreground="#475569",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))
        self.dataset_list = tk.Listbox(self.dataset_panel, height=14)
        self.dataset_list.pack(fill="both", expand=True)
        self.dataset_list.bind("<<ListboxSelect>>", lambda _event: self.on_dataset_selected())

        dataset_form = ttk.Frame(self.dataset_panel)
        dataset_form.pack(fill="x", pady=(8, 0))
        self.dataset_name = ttk.Entry(dataset_form)
        self.dataset_desc = ttk.Entry(dataset_form)
        self._grid_entry(dataset_form, 0, 0, "Name", self.dataset_name)
        self._grid_entry(dataset_form, 1, 0, "Description", self.dataset_desc)
        action_bar = ttk.Frame(self.dataset_panel)
        action_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(action_bar, text="Create", command=self.create_dataset).pack(side="left")
        ttk.Button(action_bar, text="Delete Selected", command=self.delete_dataset).pack(side="left", padx=6)
        ttk.Button(action_bar, text="Refresh", command=self.refresh_datasets).pack(side="left")

        self.dataset_summary = LabeledValuePanel(
            self.dataset_panel,
            "Dataset Summary",
            [
                ("image_count", "Images"),
                ("label_count", "Label Files"),
                ("annotated_image_count", "Annotated Images"),
                ("augmented_count", "Augmented"),
                ("annotation_coverage", "Coverage"),
            ],
            columns=2,
        )
        self.dataset_summary.pack(fill="x", pady=(8, 0))

        ttk.Label(self.upload_panel, text="Upload and Browse Files", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            self.upload_panel,
            text="Upload satu atau banyak file ke folder `images`, `labels`, atau `exports`, lalu lihat isi dataset aktif di browser file.",
            foreground="#475569",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))
        upload_form = ttk.Frame(self.upload_panel)
        upload_form.pack(fill="x")
        self.upload_dataset_id = ttk.Entry(upload_form)
        self.upload_target = ttk.Combobox(upload_form, values=["images", "labels", "exports"], state="readonly")
        self.upload_target.set("images")
        self.upload_file_label = ttk.Label(upload_form, text="No file selected", wraplength=380, justify="left")
        self._grid_entry(upload_form, 0, 0, "Dataset ID", self.upload_dataset_id)
        ttk.Label(upload_form, text="Target").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.upload_target.grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(upload_form, text="Choose Files", command=self._choose_upload_file).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(upload_form, text="Upload", command=self._upload_file).grid(row=2, column=1, sticky="e", pady=(6, 0))
        self.upload_file_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        upload_form.columnconfigure(1, weight=1)

        browser_frame = ttk.LabelFrame(self.upload_panel, text="Dataset Files", padding=8)
        browser_frame.pack(fill="both", expand=True, pady=(10, 0))
        toolbar = ttk.Frame(browser_frame)
        toolbar.pack(fill="x")
        self.browser_target = ttk.Combobox(toolbar, values=["images", "labels", "exports"], state="readonly")
        self.browser_target.set("images")
        self.browser_target.pack(side="left")
        ttk.Button(toolbar, text="Refresh Files", command=self.refresh_dataset_files).pack(side="left", padx=6)
        self.dataset_files = tk.Listbox(browser_frame)
        self.dataset_files.pack(fill="both", expand=True, pady=(8, 0))

        version_frame = ttk.LabelFrame(self.upload_panel, text="Dataset Versions", padding=8)
        version_frame.pack(fill="both", expand=True, pady=(10, 0))
        version_form = ttk.Frame(version_frame)
        version_form.pack(fill="x")
        version_form.columnconfigure(1, weight=1)
        version_form.columnconfigure(3, weight=1)
        self.version_name = ttk.Entry(version_form)
        self.version_description = ttk.Entry(version_form)
        self.version_train_ratio = ttk.Entry(version_form)
        self.version_valid_ratio = ttk.Entry(version_form)
        self.version_test_ratio = ttk.Entry(version_form)
        for entry, default in (
            (self.version_name, "Snapshot v1"),
            (self.version_description, "YOLO export snapshot"),
            (self.version_train_ratio, "0.7"),
            (self.version_valid_ratio, "0.2"),
            (self.version_test_ratio, "0.1"),
        ):
            entry.insert(0, default)
        self._grid_entry(version_form, 0, 0, "Name", self.version_name)
        self._grid_entry(version_form, 0, 2, "Description", self.version_description)
        self._grid_entry(version_form, 1, 0, "Train", self.version_train_ratio)
        self._grid_entry(version_form, 1, 2, "Valid", self.version_valid_ratio)
        self._grid_entry(version_form, 2, 0, "Test", self.version_test_ratio)
        version_btn_bar = ttk.Frame(version_form)
        version_btn_bar.grid(row=2, column=2, columnspan=2, sticky="e", pady=(4, 0))
        ttk.Button(version_btn_bar, text="Create Version", command=self.create_dataset_version).pack(side="left")
        ttk.Button(version_btn_bar, text="Rebuild Export", command=self.rebuild_dataset_version_export).pack(side="left", padx=6)
        ttk.Button(version_btn_bar, text="Refresh", command=self.refresh_dataset_versions).pack(side="left")
        self.dataset_versions = tk.Listbox(version_frame, height=5)
        self.dataset_versions.pack(fill="both", expand=True, pady=(8, 0))
        self.dataset_versions.bind("<<ListboxSelect>>", self.on_dataset_version_selected)
        self.dataset_version_summary = LabeledValuePanel(
            version_frame,
            "Version Summary",
            [
                ("version_number", "Version"),
                ("status", "Status"),
                ("export_format", "Export"),
                ("image_count", "Images"),
                ("annotated_image_count", "Annotated"),
                ("coverage_percent", "Coverage"),
            ],
            columns=2,
        )
        self.dataset_version_summary.pack(fill="x", pady=(8, 0))
        self.dataset_version_detail = JsonEditor(version_frame, "Version Detail", {}, text_height=8)
        self.dataset_version_detail.pack(fill="both", expand=True, pady=(8, 0))

        self.data_annotation_shell.columnconfigure(0, weight=1)
        self.data_annotation_shell.columnconfigure(1, weight=1)
        self.data_annotation_shell.rowconfigure(2, weight=1)
        ttk.Label(self.data_annotation_shell, text="Annotation Workflow", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.data_annotation_shell,
            text="Pilih dataset dan image, lalu anotasi langsung di atas gambar. Gunakan Previous / Next untuk berpindah image tanpa keluar dari alur kerja.",
            foreground="#475569",
            wraplength=900,
            justify="left",
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))

        annotation_toolbar = ttk.Frame(self.data_annotation_shell)
        annotation_toolbar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 8))
        annotation_toolbar.columnconfigure(1, weight=1)
        annotation_toolbar.columnconfigure(3, weight=1)
        annotation_toolbar.columnconfigure(5, weight=1)
        annotation_toolbar.columnconfigure(7, weight=1)
        self.annot_dataset_var = tk.StringVar(value="")
        self.annot_image_var = tk.StringVar(value="")
        self.annot_class_var = tk.StringVar(value="object")
        self.annot_dataset = ttk.Combobox(annotation_toolbar, textvariable=self.annot_dataset_var, values=[], state="readonly")
        self.annot_image = ttk.Entry(annotation_toolbar, textvariable=self.annot_image_var, state="readonly")
        self.annot_shape = ttk.Combobox(annotation_toolbar, values=["bbox", "polygon"], state="readonly")
        self.annot_class = ttk.Combobox(annotation_toolbar, textvariable=self.annot_class_var, values=["object"], state="normal")
        self.annot_shape.set("bbox")
        self._grid_entry(annotation_toolbar, 0, 0, "Dataset ID", self.annot_dataset)
        self._grid_entry(annotation_toolbar, 0, 2, "Image", self.annot_image)
        self._grid_entry(annotation_toolbar, 0, 4, "Type", self.annot_shape)
        self._grid_entry(annotation_toolbar, 0, 6, "Class", self.annot_class)
        annotation_actions = ttk.Frame(annotation_toolbar)
        annotation_actions.grid(row=1, column=0, columnspan=8, sticky="w", pady=(4, 0))
        self.annot_save_button = ttk.Button(
            annotation_actions,
            text="Save Annotation",
            command=self.save_current_annotation,
        )
        self.annot_save_button.pack(side="left")
        self.annot_apply_class_button = ttk.Button(
            annotation_actions,
            text="Apply Class to Selected",
            command=self._apply_class_to_selected_annotation,
            state="disabled",
        )
        self.annot_apply_class_button.pack(side="left", padx=(6, 0))
        self.annot_delete_label_button = ttk.Button(
            annotation_actions,
            text="Delete Selected Annotation",
            command=self._delete_selected_annotation,
            state="disabled",
        )
        self.annot_delete_label_button.pack(side="left", padx=(6, 0))

        self.annotation_canvas = AnnotationCanvas(self.data_annotation_shell, title="Image Annotation", size=(980, 560))
        self.annotation_canvas.grid(row=2, column=0, columnspan=2, sticky="nw")

        annotation_nav = ttk.Frame(self.data_annotation_shell)
        annotation_nav.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.annot_prev_button = ttk.Button(annotation_nav, text="Previous", command=self.previous_annotation_image)
        self.annot_prev_button.pack(side="left")
        self.annotation_status = tk.StringVar(value="Select a dataset to start annotating.")
        ttk.Label(annotation_nav, textvariable=self.annotation_status, foreground="#64748b").pack(side="left", padx=12)
        self.annot_next_button = ttk.Button(annotation_nav, text="Next", command=self.next_annotation_image)
        self.annot_next_button.pack(side="right")

        self.annot_shape.bind("<<ComboboxSelected>>", lambda _event: self._sync_annotation_mode())
        self.annot_dataset.bind("<<ComboboxSelected>>", self._on_annotation_dataset_selected)
        self.annot_class.bind("<<ComboboxSelected>>", self._on_annotation_class_input)
        self.annot_class.bind("<Return>", self._on_annotation_class_input)
        self.annot_class.bind("<FocusOut>", self._on_annotation_class_input)
        self.annotation_canvas.set_mode(self.annot_shape.get())
        self.annotation_canvas.on_labels_changed = self._on_annotation_labels_changed
        self.annotation_canvas.on_selection_changed = self._on_annotation_selection_changed
        data_scroller = self._tab_scrollers.get("data")
        if data_scroller is not None:
            data_scroller.bind("<<ScrollableFrameScrolled>>", self._on_data_tab_scrolled, add="+")
            data_scroller.canvas.bind("<Configure>", self._on_data_tab_scrolled, add="+")
        self._layout_split_shell(self.data_top_container, self.dataset_panel, self.upload_panel, compact=False, left_weight=2, right_weight=2)

    def _build_training_tab(self) -> None:
        self.training_container = ttk.Frame(self.training_tab)
        self.training_container.pack(fill="both", expand=True, padx=6, pady=6)
        self.training_container.columnconfigure(0, weight=1)
        self.training_container.rowconfigure(0, weight=1)

        self.augment_panel = ttk.Frame(self.training_container, padding=8)
        self.train_panel = ttk.Frame(self.training_container, padding=8)

        ttk.Label(self.augment_panel, text="Augment Jobs", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            self.augment_panel,
            text="MVP ini masih metadata-only, tapi panel ini tetap memisahkan pekerjaan augment dari training.",
            foreground="#475569",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))
        augment_form = ttk.LabelFrame(self.augment_panel, text="Recipe", padding=8)
        augment_form.pack(fill="x", pady=(8, 0))
        augment_form.columnconfigure(1, weight=1)
        augment_form.columnconfigure(3, weight=1)
        self.augment_dataset = ttk.Entry(augment_form)
        self.augment_transforms = ttk.Entry(augment_form)
        self.augment_multiplier = ttk.Spinbox(augment_form, from_=1, to=10, increment=1, width=8)
        self.augment_transforms.insert(0, "flip_h, brightness, blur")
        self.augment_multiplier.set(2)
        self._grid_entry(augment_form, 0, 0, "Dataset ID", self.augment_dataset)
        self._grid_entry(augment_form, 0, 2, "Transforms", self.augment_transforms)
        self._grid_entry(augment_form, 1, 0, "Multiplier", self.augment_multiplier)
        augment_btn_bar = ttk.Frame(augment_form)
        augment_btn_bar.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Button(augment_btn_bar, text="Create Augment Job", command=self.create_augment_job).pack(side="left")
        ttk.Button(augment_btn_bar, text="Refresh", command=self.refresh_augment_jobs).pack(side="left", padx=6)

        self.augment_jobs = tk.Listbox(self.augment_panel, height=12)
        self.augment_jobs.pack(fill="both", expand=True, pady=(8, 0))

        ttk.Label(self.train_panel, text="Training Jobs", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            self.train_panel,
            text="Training job untuk sticker detector. Pilih dataset, base model, dan device; backend akan memilih GPU dulu lalu fallback ke CPU bila GPU tidak tersedia.",
            foreground="#475569",
            wraplength=540,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        top = ttk.Frame(self.train_panel)
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)
        top.columnconfigure(5, weight=1)
        self.train_dataset = ttk.Entry(top)
        self.train_base_model = ttk.Combobox(top, state="readonly")
        self.train_device = ttk.Combobox(top, values=["auto", "gpu", "cpu"], state="readonly")
        self.train_device.set("auto")
        self._grid_entry(top, 0, 0, "Dataset ID", self.train_dataset)
        self._grid_entry(top, 0, 2, "Base Model", self.train_base_model)
        ttk.Label(top, text="Device").grid(row=0, column=4, sticky="w", padx=(0, 8), pady=4)
        self.train_device.grid(row=0, column=5, sticky="ew", pady=4)
        self.train_base_model_info = tk.StringVar(value="Pilih base model dari katalog YOLOv5 / YOLOv11.")
        ttk.Label(self.train_panel, textvariable=self.train_base_model_info, foreground="#475569", wraplength=540, justify="left").pack(anchor="w", pady=(4, 0))
        self.train_base_model.bind("<<ComboboxSelected>>", lambda _event: self._refresh_base_model_info())

        version_panel = ttk.LabelFrame(self.train_panel, text="Dataset Version", padding=8)
        version_panel.pack(fill="x", pady=(8, 0))
        version_panel.columnconfigure(1, weight=1)
        self.train_dataset_version = ttk.Combobox(version_panel, state="readonly")
        self.train_dataset_version_info = tk.StringVar(
            value="Pilih dataset version agar training berjalan di atas snapshot yang bisa direproduksi."
        )
        self._grid_entry(version_panel, 0, 0, "Version", self.train_dataset_version)
        ttk.Label(
            version_panel,
            textvariable=self.train_dataset_version_info,
            foreground="#475569",
            wraplength=540,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.train_dataset_version.bind("<<ComboboxSelected>>", self.on_dataset_version_selected)

        button_bar = ttk.Frame(self.train_panel)
        button_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(button_bar, text="Start Training", command=self.create_training_job).pack(side="left")
        ttk.Button(button_bar, text="Cancel Selected", command=self.cancel_training_job).pack(side="left", padx=6)
        ttk.Button(button_bar, text="Refresh", command=self.refresh_training_jobs).pack(side="left")

        self.training_lower = ttk.Frame(self.train_panel)
        self.training_lower.pack(fill="both", expand=True, pady=(10, 0))
        self.training_jobs_panel = ttk.Frame(self.training_lower)
        self.training_detail_panel = ttk.Frame(self.training_lower)

        self.train_jobs = tk.Listbox(self.training_jobs_panel)
        self.train_jobs.pack(fill="both", expand=True)
        self.train_jobs.bind("<<ListboxSelect>>", lambda _event: self.on_training_selected())

        self.training_summary = LabeledValuePanel(
            self.training_detail_panel,
            "Training Summary",
            [
                ("base_model", "Base Model"),
                ("status", "Status"),
                ("accuracy", "Accuracy"),
                ("map_score", "mAP"),
                ("r2_score", "R2"),
                ("error", "Error"),
            ],
            columns=2,
        )
        self.training_summary.pack(fill="x")
        self.training_detail = JsonEditor(self.training_detail_panel, "Training Job Detail", {})
        self.training_detail.pack(fill="both", expand=True, pady=(10, 0))
        self._layout_split_shell(self.training_container, self.augment_panel, self.train_panel, compact=False, left_weight=1, right_weight=2)
        self._layout_split_shell(self.training_lower, self.training_jobs_panel, self.training_detail_panel, compact=False, left_weight=2, right_weight=3)

    def _build_models_tab(self) -> None:
        self.models_container = ttk.Frame(self.models_tab)
        self.models_container.pack(fill="both", expand=True, padx=6, pady=6)
        self.models_container.columnconfigure(0, weight=2)
        self.models_container.columnconfigure(1, weight=3)
        self.models_container.rowconfigure(0, weight=1)

        self.models_left_panel = ttk.Frame(self.models_container, padding=8)
        self.models_right_panel = ttk.Frame(self.models_container, padding=8)

        ttk.Label(self.models_left_panel, text="Model Registry", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.models_list = tk.Listbox(self.models_left_panel)
        self.models_list.pack(fill="both", expand=True, pady=(8, 0))
        self.models_list.bind("<<ListboxSelect>>", lambda _event: self.on_model_selected())
        ttk.Button(self.models_left_panel, text="Refresh", command=self.refresh_models).pack(anchor="e", pady=(8, 0))

        ttk.Label(self.models_right_panel, text="Register Sticker Model", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.models_right_panel,
            text="Registry ini menjadi sumber resmi model sticker yang akan dipakai template. Simpan `path`, `meta_path`, runtime, task, dan daftar class.",
            foreground="#475569",
            wraplength=620,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 8))

        form = ttk.LabelFrame(self.models_right_panel, text="Model Form", padding=10)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        self.model_name = ttk.Entry(form)
        self.model_path = ttk.Entry(form)
        self.model_meta_path = ttk.Entry(form)
        self.model_runtime = ttk.Combobox(form, values=["ultralytics", "classic", "auto"], state="readonly")
        self.model_runtime.set("ultralytics")
        self.model_task = ttk.Combobox(form, values=["detection", "classification"], state="readonly")
        self.model_task.set("detection")
        self.model_class_names = ttk.Entry(form)
        self.model_arch_family = ttk.Entry(form)
        self.model_arch_variant = ttk.Entry(form)
        self._grid_entry(form, 0, 0, "Name", self.model_name)
        self._grid_entry(form, 0, 2, "Path", self.model_path)
        self._grid_entry(form, 1, 0, "Meta Path", self.model_meta_path)
        ttk.Label(form, text="Runtime").grid(row=1, column=2, sticky="w", padx=(0, 8), pady=4)
        self.model_runtime.grid(row=1, column=3, sticky="ew", pady=4)
        ttk.Label(form, text="Task").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.model_task.grid(row=2, column=1, sticky="ew", pady=4)
        self._grid_entry(form, 2, 2, "Classes CSV", self.model_class_names)
        self._grid_entry(form, 3, 0, "Architecture Family", self.model_arch_family)
        self._grid_entry(form, 3, 2, "Architecture Variant", self.model_arch_variant)
        btn_bar = ttk.Frame(form)
        btn_bar.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(btn_bar, text="Upload .pt File", command=self._upload_model_file).pack(side="left")
        ttk.Button(btn_bar, text="Register Model (path only)", command=self.create_model).pack(side="right")

        self.model_detail = JsonEditor(self.models_right_panel, "Model Detail", {})
        self.model_detail.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        self.models_right_panel.rowconfigure(3, weight=1)
        self.models_right_panel.columnconfigure(0, weight=1)
        self._layout_split_shell(self.models_container, self.models_left_panel, self.models_right_panel, compact=False, left_weight=2, right_weight=3)

    def _build_calibration_tab(self) -> None:
        self.calibration_scroller = ScrollableFrame(self.calibration_tab)
        self.calibration_scroller.pack(fill="both", expand=True, padx=6, pady=6)

        self.calibration_container = ttk.Frame(self.calibration_scroller.body)
        self.calibration_container.pack(fill="both", expand=True)

        self.calibration_container.columnconfigure(0, weight=2)
        self.calibration_container.columnconfigure(1, weight=2)
        self.calibration_container.rowconfigure(0, weight=1)

        self.calibration_left_panel = ttk.Frame(self.calibration_container, padding=8)
        self.calibration_right_outer = ttk.Frame(self.calibration_container)
        right_scroller = ScrollableFrame(self.calibration_right_outer)
        right_scroller.pack(fill="both", expand=True)
        self.calibration_right_panel = ttk.Frame(right_scroller.body, padding=8)
        self.calibration_right_panel.pack(fill="both", expand=True)

        ttk.Label(self.calibration_left_panel, text="Part Ready Color Calibration", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            self.calibration_left_panel,
            text="Flow: pilih image -> optional ROI -> compute profile -> save profile -> pakai `profile_id` itu di template part-ready.",
            foreground="#475569",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        control = ttk.LabelFrame(self.calibration_left_panel, text="Compute Profile", padding=10)
        control.pack(fill="x")
        control.columnconfigure(1, weight=1)
        control.columnconfigure(3, weight=1)
        ttk.Button(control, text="Choose Image", command=self.choose_calibration_image).grid(row=0, column=0, sticky="w")
        self.calibration_path_label = ttk.Label(control, text="No image selected")
        self.calibration_path_label.grid(row=0, column=1, columnspan=3, sticky="w")
        self.profile_name = ttk.Entry(control)
        self._grid_entry(control, 1, 0, "Profile Name", self.profile_name)
        self.calib_roi_x = ttk.Entry(control)
        self.calib_roi_y = ttk.Entry(control)
        self.calib_roi_w = ttk.Entry(control)
        self.calib_roi_h = ttk.Entry(control)
        self._grid_entry(control, 2, 0, "ROI x", self.calib_roi_x)
        self._grid_entry(control, 2, 2, "ROI y", self.calib_roi_y)
        self._grid_entry(control, 3, 0, "ROI w", self.calib_roi_w)
        self._grid_entry(control, 3, 2, "ROI h", self.calib_roi_h)
        button_bar = ttk.Frame(control)
        button_bar.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(button_bar, text="Compute", command=self.compute_profile).pack(side="left")
        ttk.Button(button_bar, text="Save Profile", command=self.save_profile).pack(side="left", padx=6)

        preview_frame = ttk.LabelFrame(self.calibration_left_panel, text="ROI Preview", padding=8)
        preview_frame.pack(fill="both", expand=True, pady=(10, 0))
        preview_frame.columnconfigure(0, weight=3)
        preview_frame.columnconfigure(1, weight=2)
        preview_frame.rowconfigure(0, weight=1)
        self.calibration_source_preview = LiveView(preview_frame, "Source Image", size=(520, 260))
        self.calibration_source_preview.grid(row=0, column=0, sticky="nw", padx=(0, 8))
        self.calibration_crop_preview = LiveView(preview_frame, "ROI Crop", size=(300, 260))
        self.calibration_crop_preview.grid(row=0, column=1, sticky="nw")
        self.calibration_preview_info = tk.StringVar(value="Pilih image untuk melihat preview ROI.")
        ttk.Label(
            self.calibration_left_panel,
            textvariable=self.calibration_preview_info,
            foreground="#475569",
            wraplength=640,
            justify="left",
        ).pack(fill="x", pady=(6, 0))

        self.calibration_editor = JsonEditor(self.calibration_left_panel, "Computed Profile", {})
        self.calibration_editor.pack(fill="both", expand=True, pady=(10, 0))

        ttk.Label(self.calibration_right_panel, text="Saved Profiles", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.profiles_list = tk.Listbox(self.calibration_right_panel, height=6)
        self.profiles_list.pack(fill="x", pady=(8, 0))
        self.profiles_list.bind("<<ListboxSelect>>", lambda _event: self.on_profile_selected())
        action_bar = ttk.Frame(self.calibration_right_panel)
        action_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(action_bar, text="Refresh", command=self.refresh_profiles).pack(side="left")
        ttk.Button(action_bar, text="Delete Selected", command=self.delete_selected_profile).pack(side="left", padx=6)
        self.profile_detail = JsonEditor(self.calibration_right_panel, "Profile Detail", {})
        self.profile_detail.pack(fill="x", pady=(10, 0))

        # ── Sticker ROI Visual Setup ─────────────────────────────────
        ttk.Separator(self.calibration_right_panel, orient="horizontal").pack(fill="x", pady=(12, 8))
        ttk.Label(self.calibration_right_panel, text="Sticker ROI Visual Setup", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            self.calibration_right_panel,
            text="Muat gambar referensi → atur ROI sticker → klik pada area kuning untuk set expected center.",
            foreground="#475569",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        sticker_setup_ctrl = ttk.LabelFrame(self.calibration_right_panel, text="Sticker ROI (rasio 0-1)", padding=8)
        sticker_setup_ctrl.pack(fill="x")
        sticker_setup_ctrl.columnconfigure(1, weight=1)
        sticker_setup_ctrl.columnconfigure(3, weight=1)

        self.sticker_setup_roi_x = ttk.Entry(sticker_setup_ctrl)
        self.sticker_setup_roi_y = ttk.Entry(sticker_setup_ctrl)
        self.sticker_setup_roi_w = ttk.Entry(sticker_setup_ctrl)
        self.sticker_setup_roi_h = ttk.Entry(sticker_setup_ctrl)
        for entry, default in zip(
            (self.sticker_setup_roi_x, self.sticker_setup_roi_y, self.sticker_setup_roi_w, self.sticker_setup_roi_h),
            ("0.14", "0.25", "0.73", "0.37"),
        ):
            entry.insert(0, default)
        self._grid_entry(sticker_setup_ctrl, 0, 0, "x", self.sticker_setup_roi_x)
        self._grid_entry(sticker_setup_ctrl, 0, 2, "y", self.sticker_setup_roi_y)
        self._grid_entry(sticker_setup_ctrl, 1, 0, "w", self.sticker_setup_roi_w)
        self._grid_entry(sticker_setup_ctrl, 1, 2, "h", self.sticker_setup_roi_h)

        self.sticker_setup_picker = RoiPickerCanvas(self.calibration_right_panel, "Visual Picker — klik area kuning", size=(340, 200))
        self.sticker_setup_picker.pack(fill="x", pady=(8, 0))
        self.sticker_setup_picker.on_center_changed = self._on_sticker_setup_center_changed

        center_bar = ttk.Frame(self.calibration_right_panel)
        center_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(center_bar, text="Expected Center X:").pack(side="left")
        self.sticker_setup_cx_var = tk.StringVar(value="0.5")
        self.sticker_setup_cy_var = tk.StringVar(value="0.5")
        ttk.Entry(center_bar, textvariable=self.sticker_setup_cx_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(center_bar, text="Y:").pack(side="left")
        ttk.Entry(center_bar, textvariable=self.sticker_setup_cy_var, width=7).pack(side="left", padx=(4, 0))

        sticker_btn_bar = ttk.Frame(self.calibration_right_panel)
        sticker_btn_bar.pack(fill="x", pady=(6, 0))
        ttk.Button(sticker_btn_bar, text="Load Image", command=self._sticker_setup_load_image).pack(side="left", padx=(0, 6))
        ttk.Button(sticker_btn_bar, text="Clear", command=self.sticker_setup_picker.clear).pack(side="left", padx=(0, 6))
        ttk.Button(sticker_btn_bar, text="Copy Values", command=self._sticker_setup_copy).pack(side="left")

        for entry in (self.sticker_setup_roi_x, self.sticker_setup_roi_y, self.sticker_setup_roi_w, self.sticker_setup_roi_h):
            entry.bind("<KeyRelease>", lambda _: self._sticker_setup_sync())
            entry.bind("<FocusOut>", lambda _: self._sticker_setup_sync())

        for entry in (self.calib_roi_x, self.calib_roi_y, self.calib_roi_w, self.calib_roi_h):
            entry.bind("<KeyRelease>", self._on_calibration_roi_changed)
            entry.bind("<FocusOut>", self._on_calibration_roi_changed)
        self._layout_split_shell(self.calibration_container, self.calibration_left_panel, self.calibration_right_outer, compact=False, left_weight=2, right_weight=2)

    def _grid_entry(self, master, row: int, column: int, label: str, widget) -> None:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=row, column=column + 1, sticky="ew", pady=4)

    def _selected_listbox_index(self, listbox: tk.Listbox) -> int | None:
        if not listbox.curselection():
            return None
        return int(listbox.curselection()[0])

    def _selected_dataset_id(self) -> str | None:
        index = self._selected_listbox_index(self.dataset_list)
        if index is None or index >= len(self._dataset_cache):
            return None
        return str(self._dataset_cache[index]["id"])

    def _selected_dataset_record(self) -> dict | None:
        index = self._selected_listbox_index(self.dataset_list)
        if index is None or index >= len(self._dataset_cache):
            return None
        return self._dataset_cache[index]

    def _dataset_record_for_id(self, dataset_id: str | None) -> dict | None:
        target_id = str(dataset_id or "").strip()
        if not target_id:
            return None
        for item in self._dataset_cache:
            if str(item.get("id") or "").strip() == target_id:
                return item
        return None

    def _select_dataset_in_listbox(self, dataset_id: str | None) -> None:
        target_id = str(dataset_id or "").strip()
        if not target_id or not hasattr(self, "dataset_list"):
            return
        for index, item in enumerate(self._dataset_cache):
            item_id = str(item.get("id") or "").strip()
            if item_id != target_id:
                continue
            self.dataset_list.selection_clear(0, "end")
            self.dataset_list.selection_set(index)
            self.dataset_list.see(index)
            return

    def _dataset_id_options(self) -> list[str]:
        options: list[str] = []
        for item in self._dataset_cache:
            dataset_id = str(item.get("id") or "").strip()
            if dataset_id and dataset_id not in options:
                options.append(dataset_id)
        return options

    def _sync_annotation_dataset_selector(self, *, preferred_dataset_id: str | None = None) -> None:
        if not hasattr(self, "annot_dataset"):
            return
        values = self._dataset_id_options()
        self.annot_dataset.configure(values=values)
        preferred = str(preferred_dataset_id or "").strip()
        current = self.annot_dataset_var.get().strip()
        if preferred and preferred in values:
            self.annot_dataset_var.set(preferred)
            return
        if current and current in values:
            return
        if not values:
            self.annot_dataset_var.set("")

    def _current_dataset_id(self) -> str | None:
        for candidate in (
            self._selected_dataset_id(),
            self.upload_dataset_id.get().strip(),
            self._annotation_dataset_id,
            self.augment_dataset.get().strip(),
            self.train_dataset.get().strip(),
        ):
            if candidate:
                return candidate
        return None

    def _selected_dataset_version_id(self) -> str | None:
        index = self._selected_listbox_index(self.dataset_versions)
        if index is None or index >= len(self._dataset_version_cache):
            return None
        return str(self._dataset_version_cache[index].get("id") or "").strip() or None

    def _selected_dataset_version_record(self) -> dict | None:
        index = self._selected_listbox_index(self.dataset_versions)
        if index is None or index >= len(self._dataset_version_cache):
            return None
        return self._dataset_version_cache[index]

    def _active_dataset_version_record(self) -> dict | None:
        active_id = str(self._active_dataset_version_id or "").strip()
        if not active_id:
            return None
        record = self._dataset_version_lookup.get(active_id)
        if record is not None:
            return record
        for item in self._dataset_version_cache:
            if str(item.get("id") or "").strip() == active_id:
                return item
        return None

    def _version_lookup_key(self, version: dict) -> str:
        return str(version.get("display_label") or version.get("id") or "").strip()

    def _select_training_job_in_listbox(self, job_id: str | None) -> bool:
        target_id = str(job_id or "").strip()
        if not target_id or not hasattr(self, "train_jobs"):
            return False
        for index, item in enumerate(self._training_jobs):
            if str(item.get("id") or "").strip() != target_id:
                continue
            self.train_jobs.selection_clear(0, "end")
            self.train_jobs.selection_set(index)
            self.train_jobs.see(index)
            return True
        return False

    def _update_dataset_version_summary(self, version: dict | None) -> None:
        if not version:
            self.dataset_version_summary.reset()
            self.dataset_version_detail.set_payload({})
            self.train_dataset_version.set("")
            self.train_dataset_version_info.set(
                "Pilih dataset version agar training berjalan di atas snapshot yang bisa direproduksi."
            )
            return
        self.dataset_version_summary.set_values(
            {
                "version_number": version.get("version_number"),
                "status": version.get("status"),
                "export_format": version.get("export_format"),
                "image_count": version.get("image_count", 0),
                "annotated_image_count": version.get("annotated_image_count", 0),
                "coverage_percent": f"{float(version.get('coverage_percent') or 0):.1f}%",
            }
        )
        self.dataset_version_detail.set_payload(version)
        self.train_dataset_version.set(self._version_lookup_key(version))
        self._refresh_train_dataset_version_info()

    def _refresh_train_dataset_version_info(self) -> None:
        spec = self._selected_dataset_version_spec()
        if spec is None:
            self.train_dataset_version_info.set(
                "Pilih dataset version agar training berjalan di atas snapshot yang bisa direproduksi."
            )
            return
        split_ratios = spec.get("split_ratios") or {}
        split_text = ", ".join(f"{key}={value}" for key, value in split_ratios.items()) if isinstance(split_ratios, dict) else ""
        self.train_dataset_version_info.set(
            f"{spec.get('display_label')} | export {spec.get('export_format')} | {spec.get('export_root')}"
            f"{f' | {split_text}' if split_text else ''}"
        )

    def _update_dataset_summary(self, dataset: dict | None) -> None:
        if not dataset:
            self.dataset_summary.reset()
            return
        coverage = dataset.get("annotation_coverage")
        coverage_text = f"{float(coverage) * 100:.0f}%" if coverage is not None else "-"
        self.dataset_summary.set_values(
            {
                "image_count": dataset.get("image_count", 0),
                "label_count": dataset.get("label_count", 0),
                "annotated_image_count": dataset.get("annotated_image_count", 0),
                "augmented_count": dataset.get("augmented_count", 0),
                "annotation_coverage": coverage_text,
            }
        )

    def on_dataset_selected(self) -> None:
        if self._ignore_next_dataset_list_selection_event:
            self._ignore_next_dataset_list_selection_event = False
            return
        dataset = self._selected_dataset_record()
        dataset_id = str(dataset.get("id")) if dataset else None
        if not dataset_id:
            fallback_dataset_id = self._resolve_annotation_dataset_id() or self._current_dataset_id()
            dataset = self._dataset_record_for_id(fallback_dataset_id)
            if fallback_dataset_id and dataset is None:
                self._reset_dataset_context()
                return
            if not fallback_dataset_id:
                self._reset_annotation_state()
                return
            dataset_id = fallback_dataset_id
            if dataset is None:
                dataset = {"id": dataset_id, "name": self._annotation_dataset_name or dataset_id}
            if self._selected_dataset_id() != dataset_id:
                self._ignore_next_dataset_list_selection_event = True
                self.after_idle(lambda dataset_id=dataset_id: self._select_dataset_in_listbox(dataset_id))
        previous_dataset_id = self._annotation_dataset_id
        if previous_dataset_id and previous_dataset_id != dataset_id:
            self._annotation_manual_classes = []
            self._active_dataset_version_id = None
        self._annotation_dataset_id = dataset_id
        self._annotation_dataset_name = str(dataset.get("name") or dataset_id).strip()
        self.annot_dataset_var.set(dataset_id)
        self._sync_annotation_dataset_selector(preferred_dataset_id=dataset_id)
        self._select_dataset_in_listbox(dataset_id)
        for widget in (self.upload_dataset_id, self.augment_dataset, self.train_dataset):
            widget.delete(0, "end")
            widget.insert(0, dataset_id)
        self._update_dataset_summary(dataset)
        self.refresh_dataset_files()
        self.refresh_annotation_images()
        self.refresh_dataset_versions()
        self._sync_annotation_class_name()

    def refresh_datasets(self):
        try:
            items = self.api.list_datasets()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset", str(exc))
            return
        previous_selected_id = self._selected_dataset_id()
        self._dataset_cache = items
        self._sync_annotation_dataset_selector(preferred_dataset_id=previous_selected_id)
        self.dataset_list.delete(0, "end")
        for item in items:
            summary = f"{item.get('image_count', 0)} imgs / {item.get('annotated_image_count', 0)} ann / {item.get('augmented_count', 0)} aug"
            self.dataset_list.insert("end", f"{item['id']} | {item['name']} | {summary}")
        selected_dataset = None
        if previous_selected_id:
            for index, item in enumerate(items):
                if str(item.get("id") or "") == previous_selected_id:
                    self.dataset_list.selection_clear(0, "end")
                    self.dataset_list.selection_set(index)
                    self.dataset_list.see(index)
                    selected_dataset = item
                    break
        if selected_dataset is None and items:
            selected_dataset = items[0]
            self.dataset_list.selection_clear(0, "end")
            self.dataset_list.selection_set(0)
            self.dataset_list.see(0)
        self.dataset_summary.reset()
        self._update_dataset_summary(selected_dataset)
        if selected_dataset is not None:
            self.on_dataset_selected()
        else:
            self._reset_dataset_context()

    def _on_annotation_dataset_selected(self, _event=None) -> None:
        dataset_id = self.annot_dataset_var.get().strip()
        if not dataset_id:
            return
        for index, item in enumerate(self._dataset_cache):
            item_id = str(item.get("id") or "").strip()
            if item_id != dataset_id:
                continue
            self.dataset_list.selection_clear(0, "end")
            self.dataset_list.selection_set(index)
            self.dataset_list.see(index)
            self.on_dataset_selected()
            return

    def _restore_annotation_context_after_toolbar_change(self) -> None:
        dataset_id = self._resolve_annotation_dataset_id()
        if dataset_id:
            self._sync_annotation_dataset_selector(preferred_dataset_id=dataset_id)
        if not hasattr(self, "annotation_canvas"):
            return
        if self._annotation_files and self.annotation_canvas._source_frame is None:
            self._reload_active_annotation_image()
            return
        if self.annotation_canvas._source_frame is not None:
            if self.annotation_canvas._photo is None:
                self.annotation_canvas.redraw()
            else:
                self.annotation_canvas.request_redraw()

    def create_dataset(self):
        try:
            self.api.create_dataset({"name": self.dataset_name.get().strip(), "description": self.dataset_desc.get().strip()})
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset", str(exc))
            return
        self.dataset_name.delete(0, "end")
        self.dataset_desc.delete(0, "end")
        self.refresh_datasets()

    def delete_dataset(self):
        dataset = self._selected_dataset_record()
        dataset_id = self._selected_dataset_id() or self._resolve_annotation_dataset_id() or self._current_dataset_id()
        if not dataset_id and len(self._dataset_cache) == 1:
            dataset_id = str(self._dataset_cache[0].get("id") or "").strip() or None
        if not dataset_id:
            messagebox.showwarning("Dataset", "Pilih dataset dulu atau pastikan context annotation aktif.", parent=self.winfo_toplevel())
            return
        dataset_name = ""
        if dataset and str(dataset.get("id") or "").strip() == dataset_id:
            dataset_name = str(dataset.get("name") or "").strip()
        if not dataset_name:
            dataset_name = self._annotation_dataset_name or dataset_id
        confirm_message = f"Delete dataset '{dataset_name}' ({dataset_id})?"
        if not messagebox.askyesno("Dataset", confirm_message, parent=self.winfo_toplevel()):
            return
        try:
            self.api.delete_dataset(dataset_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset", str(exc), parent=self.winfo_toplevel())
            return
        self.refresh_datasets()

    def _choose_upload_file(self):
        target = self.upload_target.get().strip() or "images"
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")] if target == "images" else [("All files", "*.*")]
        paths = filedialog.askopenfilenames(title="Choose files", filetypes=filetypes)
        if paths:
            self.upload_paths = list(paths)
            self.upload_path = self.upload_paths[0] if len(self.upload_paths) == 1 else None
            self.upload_file_label.configure(text=self._upload_selection_text())

    def _upload_selection_text(self) -> str:
        if self.upload_paths:
            names = [Path(path).name for path in self.upload_paths]
        elif self.upload_path:
            names = [Path(self.upload_path).name]
        else:
            return "No file selected"
        if len(names) == 1:
            return names[0]
        preview = ", ".join(names[:3])
        if len(names) > 3:
            preview = f"{preview}, +{len(names) - 3} more"
        return f"{len(names)} files selected: {preview}"

    def _upload_file(self):
        selected_paths = list(self.upload_paths)
        if not selected_paths and self.upload_path:
            selected_paths = [self.upload_path]
        if not selected_paths:
            messagebox.showwarning("Upload", "Choose file dulu.")
            return
        dataset_id = self.upload_dataset_id.get().strip()
        if not dataset_id:
            messagebox.showwarning("Upload", "Dataset ID wajib diisi.")
            return
        try:
            self.api.upload_dataset_files(
                dataset_id,
                selected_paths,
                target=self.upload_target.get().strip() or "images",
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Upload", str(exc))
            return
        self.upload_paths = []
        self.upload_path = None
        self.upload_file_label.configure(text="No file selected")
        self.refresh_dataset_files()
        messagebox.showinfo("Upload", "Upload selesai.")

    def refresh_dataset_files(self):
        dataset_id = self.upload_dataset_id.get().strip() or self._selected_dataset_id()
        self.dataset_files.delete(0, "end")
        if not dataset_id:
            return
        try:
            items = self.api.list_dataset_files(dataset_id, self.browser_target.get().strip() or "images")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset Files", str(exc))
            return
        for item in items:
            marker = "✓" if item.get("annotation_exists") else "•"
            self.dataset_files.insert("end", f"{marker} {item['name']} | {item.get('size', 0)} bytes")

    def refresh_dataset_versions(self, *, preferred_version_id: str | None = None, preserve_selection: bool = True):
        previous_selected = self._selected_dataset_version_record()
        previous_selected_id = None
        if preserve_selection:
            previous_selected_id = self._active_dataset_version_id
            if previous_selected_id is None and previous_selected is not None:
                previous_selected_id = str(previous_selected.get("id") or "").strip() or None
        preferred_id = str(preferred_version_id or "").strip() or None

        dataset_id = self._current_dataset_id()
        self.dataset_versions.delete(0, "end")
        self._dataset_version_cache = []
        self._dataset_version_lookup = {}
        if not dataset_id:
            self.train_dataset_version["values"] = []
            self._active_dataset_version_id = None
            self._update_dataset_version_summary(None)
            self._sync_annotation_class_name()
            return

        try:
            items = self.api.list_dataset_versions(dataset_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset Versions", str(exc))
            return

        visible_versions: list[dict] = []
        values: list[str] = []
        for item in items:
            display_label = str(item.get("display_label") or item.get("id") or "").strip()
            if not display_label:
                continue
            visible_versions.append(item)
            values.append(display_label)
            self._dataset_version_lookup[display_label] = item
            self._dataset_version_lookup[str(item.get("id") or "")] = item
            self.dataset_versions.insert(
                "end",
                f"{display_label} | {item.get('export_format') or 'yolo'} | {item.get('export_root') or '-'}",
            )
        self._dataset_version_cache = visible_versions

        self.train_dataset_version["values"] = values
        if values:
            selected_index = 0
            target_id = preferred_id or previous_selected_id
            if target_id:
                for index, item in enumerate(visible_versions):
                    if str(item.get("id") or "").strip() == target_id:
                        selected_index = index
                        break
            self.dataset_versions.selection_clear(0, "end")
            self.dataset_versions.selection_set(selected_index)
            self.dataset_versions.see(selected_index)
            selected_version = visible_versions[selected_index]
            self.train_dataset_version.current(selected_index)
            self._active_dataset_version_id = str(selected_version.get("id") or "").strip() or None
            self._update_dataset_version_summary(selected_version)
        else:
            self.train_dataset_version.set("")
            self._active_dataset_version_id = None
            self._update_dataset_version_summary(None)
        self._sync_annotation_class_name()
        self._refresh_train_dataset_version_info()

    def create_dataset_version(self):
        dataset_id = self._current_dataset_id()
        if not dataset_id:
            messagebox.showwarning("Dataset Versions", "Dataset ID wajib diisi atau dataset harus dipilih dulu.")
            return

        def _ratio(entry: ttk.Entry, default: float) -> float:
            raw = entry.get().strip()
            if not raw:
                return default
            return float(raw)

        try:
            payload = {
                "name": self.version_name.get().strip(),
                "description": self.version_description.get().strip(),
                "export_format": "yolo",
                "split_ratios": {
                    "train": _ratio(self.version_train_ratio, 0.7),
                    "valid": _ratio(self.version_valid_ratio, 0.2),
                    "test": _ratio(self.version_test_ratio, 0.1),
                },
            }
        except ValueError:
            messagebox.showerror("Dataset Versions", "Split ratios harus numerik.")
            return

        try:
            created = self.api.create_dataset_version(dataset_id, payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset Versions", str(exc))
            return

        self.dataset_version_detail.set_payload(created)
        self.refresh_dataset_versions(preferred_version_id=str(created.get("id") or ""), preserve_selection=False)
        self._reload_active_annotation_image()
        messagebox.showinfo("Dataset Versions", f"Version '{created.get('display_label')}' berhasil dibuat.")

    def rebuild_dataset_version_export(self):
        dataset_id = self._current_dataset_id()
        version = self._selected_dataset_version_spec() or self._selected_dataset_version_record()
        if not dataset_id or not version:
            messagebox.showwarning("Dataset Versions", "Pilih dataset dan version dulu.")
            return
        try:
            refreshed = self.api.export_dataset_version(dataset_id, str(version.get("id") or ""))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset Versions", str(exc))
            return
        self.dataset_version_detail.set_payload(refreshed)
        self.refresh_dataset_versions(preferred_version_id=str(refreshed.get("id") or ""), preserve_selection=False)
        self._reload_active_annotation_image()
        messagebox.showinfo("Dataset Versions", f"Export version '{refreshed.get('display_label')}' selesai diperbarui.")

    def _reload_active_annotation_image(self) -> None:
        dataset_id = self._resolve_annotation_dataset_id()
        if not dataset_id:
            return
        if not self._annotation_files:
            self.refresh_annotation_images()
            return
        index = self._annotation_image_index()
        if index is None:
            self.refresh_annotation_images()
            return
        self._load_annotation_for_index(index, save_current=False)

    def _reset_annotation_state(self) -> None:
        self._annotation_files = []
        self._annotation_index = None
        self._annotation_dataset_id = None
        self._annotation_dataset_name = ""
        self._active_dataset_version_id = None
        self.annot_dataset_var.set("")
        self.annot_image_var.set("")
        self.annotation_status.set("Select a dataset to start annotating.")
        if hasattr(self, "annotation_canvas"):
            self.annotation_canvas.clear()
        if hasattr(self, "annot_shape"):
            self.annot_shape.set("bbox")
        self._reset_annotation_class_state()
        self._update_annotation_nav_state()

    def _reset_dataset_context(self) -> None:
        for widget in (self.upload_dataset_id, self.augment_dataset, self.train_dataset):
            widget.delete(0, "end")
        self.dataset_files.delete(0, "end")
        self.dataset_versions.delete(0, "end")
        self._dataset_version_cache = []
        self._dataset_version_lookup = {}
        self.dataset_summary.reset()
        self._reset_annotation_state()
        self._sync_annotation_dataset_selector()
        self._update_dataset_version_summary(None)

    def _resolve_annotation_dataset_id(self) -> str | None:
        for candidate in (
            self.annot_dataset_var.get().strip(),
            self._annotation_dataset_id,
            self._selected_dataset_id(),
        ):
            if candidate:
                return candidate
        return None

    def _sync_annotation_mode(self) -> None:
        if not hasattr(self, "annotation_canvas"):
            return
        mode = (self.annot_shape.get().strip() or "bbox").lower()
        if mode not in {"bbox", "polygon"}:
            mode = "bbox"
        self.annotation_canvas.set_mode(mode)

    def _on_data_tab_scrolled(self, _event=None) -> None:
        if hasattr(self, "annotation_canvas"):
            self.annotation_canvas.request_redraw()

    def _reset_annotation_class_state(self) -> None:
        self._annotation_class_name = "object"
        self._annotation_class_options = ["object"]
        self._annotation_manual_classes = []
        self._annotation_selected_label_index = None
        if hasattr(self, "annot_class"):
            self.annot_class.configure(values=self._annotation_class_options)
        if hasattr(self, "annot_class_var"):
            self.annot_class_var.set("object")
        if hasattr(self, "annotation_canvas"):
            self.annotation_canvas.set_class_name("object")
            self.annotation_canvas.set_selected_label_index(None)
        if hasattr(self, "annot_apply_class_button"):
            self.annot_apply_class_button.configure(state="disabled")
        if hasattr(self, "annot_delete_label_button"):
            self.annot_delete_label_button.configure(state="disabled")

    def _normalize_annotation_class_name(self, value: str | None) -> str:
        return str(value or "").strip() or "object"

    def _annotation_label_class_name(self, label: dict | None) -> str | None:
        if not isinstance(label, dict):
            return None
        class_name = str(label.get("class_name") or label.get("class") or label.get("label") or "").strip()
        return class_name or None

    def _ensure_manual_annotation_class(self, class_name: str) -> None:
        normalized = self._normalize_annotation_class_name(class_name)
        if normalized == "object":
            return
        if normalized not in self._annotation_manual_classes:
            self._annotation_manual_classes.append(normalized)

    def _label_class_names(self, labels: list[dict] | None = None) -> list[str]:
        source = labels if labels is not None else (self.annotation_canvas.get_labels() if hasattr(self, "annotation_canvas") else [])
        names: list[str] = []
        for label in source:
            class_name = self._annotation_label_class_name(label)
            if class_name and class_name not in names:
                names.append(class_name)
        return names

    def _version_class_names(self) -> list[str]:
        names: list[str] = []
        for version in self._dataset_version_cache:
            class_names = version.get("class_names")
            if not isinstance(class_names, list):
                continue
            for item in class_names:
                class_name = str(item).strip()
                if class_name and class_name not in names:
                    names.append(class_name)
        return names

    def _sync_annotation_class_name(self, *, preferred_class: str | None = None, labels: list[dict] | None = None) -> None:
        if preferred_class:
            self._ensure_manual_annotation_class(preferred_class)

        class_options: list[str] = []

        def _push(name: str) -> None:
            normalized = self._normalize_annotation_class_name(name)
            if normalized not in class_options:
                class_options.append(normalized)

        _push("object")
        for name in self._version_class_names():
            _push(name)
        for name in self._annotation_manual_classes:
            _push(name)
        for name in self._label_class_names(labels):
            _push(name)

        selected_label = None
        if self._annotation_selected_label_index is not None and hasattr(self, "annotation_canvas"):
            current_labels = self.annotation_canvas.get_labels()
            if 0 <= self._annotation_selected_label_index < len(current_labels):
                selected_label = current_labels[self._annotation_selected_label_index]
        selected_class = self._annotation_label_class_name(selected_label)

        active_class = self._normalize_annotation_class_name(
            preferred_class
            or selected_class
            or self.annot_class_var.get().strip()
            or self._annotation_class_name
        )
        _push(active_class)

        self._annotation_class_options = class_options
        self._annotation_class_name = active_class
        if hasattr(self, "annot_class"):
            self.annot_class.configure(values=self._annotation_class_options)
        if hasattr(self, "annot_class_var"):
            self.annot_class_var.set(active_class)
        if hasattr(self, "annotation_canvas"):
            self.annotation_canvas.set_class_name(active_class)

    def _apply_class_to_active_annotation(self, class_name: str) -> bool:
        canvas_selected = self.annotation_canvas.get_selected_label_index() if hasattr(self, "annotation_canvas") else None
        if canvas_selected is None:
            return False
        if not self.annotation_canvas.set_selected_label_class_name(class_name):
            return False
        self.annotation_status.set(f"Class '{class_name}' applied to selected annotation.")
        return True

    def _on_annotation_class_input(self, _event=None) -> None:
        class_name = self._normalize_annotation_class_name(self.annot_class_var.get())
        self._ensure_manual_annotation_class(class_name)
        if self._apply_class_to_active_annotation(class_name):
            return
        self._sync_annotation_class_name(preferred_class=class_name)

    def _apply_class_to_selected_annotation(self) -> None:
        class_name = self._normalize_annotation_class_name(self.annot_class_var.get())
        self._ensure_manual_annotation_class(class_name)
        if not self._apply_class_to_active_annotation(class_name):
            messagebox.showinfo("Annotate", "Pilih anotasi dulu dengan klik kanan pada object di canvas.")
            self._sync_annotation_class_name(preferred_class=class_name)
            return
        self._sync_annotation_class_name(preferred_class=class_name)

    def _delete_selected_annotation(self) -> None:
        if self._annotation_selected_label_index is None:
            messagebox.showinfo("Annotate", "Pilih anotasi dulu dengan klik kanan pada object di canvas.")
            return
        if not self.annotation_canvas.delete_selected_label():
            messagebox.showwarning("Annotate", "Anotasi terpilih tidak ditemukan.")
            return
        self.annotation_status.set("Selected annotation deleted.")

    def _on_annotation_selection_changed(self, label: dict | None, index: int | None) -> None:
        self._annotation_selected_label_index = index
        has_selection = index is not None
        if hasattr(self, "annot_apply_class_button"):
            self.annot_apply_class_button.configure(state="normal" if has_selection else "disabled")
        if hasattr(self, "annot_delete_label_button"):
            self.annot_delete_label_button.configure(state="normal" if has_selection else "disabled")
        class_name = self._annotation_label_class_name(label)
        if class_name:
            self._sync_annotation_class_name(preferred_class=class_name)

    def _update_annotation_nav_state(self) -> None:
        if not hasattr(self, "annot_prev_button") or not hasattr(self, "annot_next_button"):
            return
        has_items = bool(self._annotation_files)
        index = self._annotation_index if self._annotation_index is not None else -1
        self.annot_prev_button.configure(state="normal" if has_items and index > 0 else "disabled")
        self.annot_next_button.configure(state="normal" if has_items and 0 <= index < len(self._annotation_files) - 1 else "disabled")

    def _annotation_image_index(self, image_name: str | None = None) -> int | None:
        if not self._annotation_files:
            return None
        target = (image_name or self.annot_image_var.get()).strip()
        if target:
            for index, item in enumerate(self._annotation_files):
                if str(item.get("name") or "").strip() == target:
                    return index
        if self._annotation_index is not None and 0 <= self._annotation_index < len(self._annotation_files):
            return self._annotation_index
        return 0

    def _load_annotation_for_index(self, index: int, *, save_current: bool = True) -> None:
        if index < 0 or index >= len(self._annotation_files):
            return
        if save_current and not self._save_current_annotation(silent=True):
            self.annotation_status.set("Autosave failed. Stay on the current image and try again.")
            return
        item = self._annotation_files[index]
        dataset_id = self._resolve_annotation_dataset_id()
        image_name = str(item.get("name") or "").strip()
        image_path = str(item.get("path") or "").strip()
        self._annotation_index = index
        self.annot_image_var.set(image_name or "-")
        base_status = f"{index + 1} / {len(self._annotation_files)} images"
        self.annotation_status.set(base_status)
        loaded = False
        loaded_source = ""
        if dataset_id and image_name:
            try:
                image_bytes = self.api.download_dataset_image(dataset_id, image_name)
            except Exception:
                image_bytes = b""
            if image_bytes:
                loaded = self.annotation_canvas.load_image_bytes(image_bytes, image_name=image_name)
                if loaded:
                    loaded_source = "backend"
        if not loaded and image_path:
            loaded = self.annotation_canvas.load_image_path(image_path)
            if loaded:
                loaded_source = "local"
        if not loaded:
            self.annotation_canvas.clear()
            self.annotation_status.set(f"{image_name or 'image'} could not be loaded")
        elif loaded_source:
            self.annotation_status.set(f"{base_status} | loaded via {loaded_source}")
        self.annotation_canvas.set_image_name(image_name)
        self.annotation_canvas.set_class_name(self._annotation_class_name)
        labels_payload: list[dict] = []
        if dataset_id and image_name:
            try:
                payload = self.api.get_annotation(dataset_id, image_name)
            except Exception:
                payload = {"labels": []}
            labels = payload.get("labels") if isinstance(payload, dict) else []
            if isinstance(labels, list):
                labels_payload = labels
        self.annotation_canvas.set_labels(labels_payload)
        self._sync_annotation_class_name(labels=labels_payload)
        self._sync_annotation_mode()
        self.annotation_canvas.request_redraw()
        self._update_annotation_nav_state()

    def refresh_annotation_images(self) -> None:
        dataset_id = self._resolve_annotation_dataset_id()
        self._annotation_files = []
        self._annotation_index = None
        if not dataset_id:
            self._reset_annotation_state()
            return
        self._annotation_dataset_id = dataset_id
        self.annot_dataset_var.set(dataset_id)
        try:
            items = self.api.list_dataset_files(dataset_id, "images")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Annotate", str(exc))
            self._reset_annotation_state()
            return
        self._annotation_files = items
        if not items:
            self.annot_image_var.set("-")
            self.annotation_canvas.clear()
            self.annotation_status.set("Dataset has no images to annotate.")
            self._update_annotation_nav_state()
            return
        preferred = self._annotation_image_index()
        if preferred is None:
            preferred = 0
        self._load_annotation_for_index(preferred, save_current=False)

    def on_annotation_image_selected(self):
        index = self._annotation_image_index()
        if index is None:
            return
        self._load_annotation_for_index(index)

    def previous_annotation_image(self) -> None:
        if self._annotation_index is None:
            return
        self._load_annotation_for_index(max(0, self._annotation_index - 1))

    def next_annotation_image(self) -> None:
        if self._annotation_index is None:
            return
        self._load_annotation_for_index(min(len(self._annotation_files) - 1, self._annotation_index + 1))

    def _save_current_annotation(self, *, silent: bool = False) -> bool:
        dataset_id = self._resolve_annotation_dataset_id()
        image_name = self.annot_image_var.get().strip()
        if not dataset_id or not image_name or image_name == "-":
            return False
        try:
            payload = self.annotation_canvas.get_labels()
            self.api.save_annotation(dataset_id, image_name, payload)
        except Exception as exc:  # noqa: BLE001
            self.annotation_status.set(f"Autosave failed for {image_name}")
            if not silent:
                messagebox.showerror("Annotate", str(exc))
            return False
        self.annotation_status.set(f"Saved {image_name}")
        return True

    def save_current_annotation(self) -> None:
        dataset_id = self._resolve_annotation_dataset_id()
        image_name = self.annot_image_var.get().strip()
        if not dataset_id or not image_name or image_name == "-":
            messagebox.showwarning("Annotate", "Pilih dataset dan image dulu.")
            return
        self._save_current_annotation(silent=False)

    def _on_annotation_labels_changed(self, _labels: list[dict]) -> None:
        self._sync_annotation_class_name(labels=_labels)
        self._save_current_annotation(silent=True)

    def on_dataset_version_selected(self, event=None) -> None:
        source_widget = getattr(event, "widget", None)
        if source_widget is self.train_dataset_version:
            version = self._selected_dataset_version_spec() or self._selected_dataset_version_record()
        elif source_widget is self.dataset_versions:
            version = self._selected_dataset_version_record() or self._selected_dataset_version_spec()
        else:
            version = self._selected_dataset_version_spec() or self._selected_dataset_version_record()
        if not version:
            self._active_dataset_version_id = None
            self._update_dataset_version_summary(None)
            self._sync_annotation_class_name()
            return
        selected_id = str(version.get("id") or "").strip() or None
        self._active_dataset_version_id = selected_id
        if selected_id:
            for index, item in enumerate(self._dataset_version_cache):
                if str(item.get("id") or "").strip() != selected_id:
                    continue
                self.dataset_versions.selection_clear(0, "end")
                self.dataset_versions.selection_set(index)
                self.dataset_versions.see(index)
                break
        self._update_dataset_version_summary(version)
        self._sync_annotation_class_name()
        if self.annotation_canvas._source_frame is None:
            self._reload_active_annotation_image()
        else:
            self.annotation_canvas.request_redraw()

    def refresh_augment_jobs(self):
        try:
            items = self.api.list_augment_jobs()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Augment", str(exc))
            return
        self.augment_jobs.delete(0, "end")
        for item in items:
            transforms = ", ".join(item.get("transforms") or [])
            multiplier = item.get("multiplier") or 1
            self.augment_jobs.insert("end", f"{item.get('id')} | {item.get('dataset_id')} | {item.get('status')} | x{multiplier} | {transforms}")

    def create_augment_job(self):
        raw_transforms = self.augment_transforms.get().strip()
        transforms = [item.strip() for item in raw_transforms.split(",") if item.strip()] if raw_transforms else []
        if not transforms:
            transforms = ["flip_h", "brightness", "blur"]
        try:
            multiplier = max(1, min(10, int(float(self.augment_multiplier.get() or 2))))
        except ValueError:
            multiplier = 2
        try:
            self.api.create_augment_job(
                {
                    "dataset_id": self.augment_dataset.get().strip(),
                    "transforms": transforms,
                    "multiplier": multiplier,
                }
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Augment", str(exc))
            return
        self.refresh_augment_jobs()

    def _training_metric_sources(self, item: dict) -> list[dict]:
        sources: list[dict] = []
        for key in ("metrics", "evaluation", "results"):
            value = item.get(key)
            if isinstance(value, dict):
                sources.append(value)
        params = item.get("params")
        if isinstance(params, dict):
            for key in ("metrics", "evaluation", "results"):
                value = params.get(key)
                if isinstance(value, dict):
                    sources.append(value)
        return sources

    def _training_metric_value(self, item: dict, *keys: str) -> object:
        for key in keys:
            value = item.get(key)
            if value is not None and value != "":
                return value
        for source in self._training_metric_sources(item):
            for key in keys:
                value = source.get(key)
                if value is not None and value != "":
                    return value
        return None

    @staticmethod
    def _format_training_percent(value: object) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "-"
        if raw.endswith("%"):
            return raw
        try:
            numeric = Decimal(raw)
        except InvalidOperation:
            return raw
        if numeric <= 1:
            numeric *= Decimal("100")
        return f"{numeric.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):f}%"

    @staticmethod
    def _format_training_decimal(value: object, *, digits: int = 3) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "-"
        if raw.endswith("%"):
            return raw
        try:
            numeric = Decimal(raw)
        except InvalidOperation:
            return raw
        quantize_pattern = Decimal("1").scaleb(-digits)
        return f"{numeric.quantize(quantize_pattern, rounding=ROUND_HALF_UP):f}"

    def _training_error_summary(self, item: dict) -> str:
        for label, keys in (
            ("RMSE", ("rmse", "validation_rmse", "val_rmse")),
            ("MAE", ("mae", "validation_mae", "val_mae")),
            ("MSE", ("mse", "validation_mse", "val_mse")),
            ("Loss", ("loss", "val_loss", "validation_loss")),
            ("Error", ("error",)),
        ):
            value = self._training_metric_value(item, *keys)
            if value is None:
                continue
            formatted = self._format_training_decimal(value, digits=4)
            if formatted != "-":
                return f"{label} {formatted}"
        return "-"

    def refresh_training_jobs(self):
        try:
            items = self.api.list_training_jobs()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training", str(exc))
            return
        self._training_jobs = items
        self.train_jobs.delete(0, "end")
        for item in items:
            device_mode = item.get("requested_device_mode") or item.get("device_mode") or item.get("params", {}).get("device_mode") or "auto"
            effective_device = item.get("effective_device") or "pending"
            base_model = item.get("base_model_display_name") or item.get("base_model") or "-"
            version_label = item.get("dataset_version_display_label") or item.get("dataset_version_name") or item.get("dataset_version_id") or "-"
            self.train_jobs.insert(
                "end",
                f"{item['id']} | {item['dataset_id']} | {version_label} | {item['status']} | {base_model} | {device_mode} -> {effective_device}",
            )

        self.refresh_base_models()
        self._refresh_train_dataset_version_info()
        if items:
            if not self._select_training_job_in_listbox(self._active_training_job_id):
                self._select_training_job_in_listbox(items[0].get("id"))
            self.on_training_selected()
        else:
            self._active_training_job_id = None
            self.training_summary.reset()
            self.training_detail.set_payload({})

    def create_training_job(self):
        spec = self._selected_base_model_spec()
        if spec is None:
            messagebox.showwarning("Training", "Pilih base model dulu.")
            return
        dataset_id = self.train_dataset.get().strip()
        if not dataset_id:
            messagebox.showwarning("Training", "Dataset ID wajib diisi.")
            return
        version = self._selected_dataset_version_spec()
        payload = {
            "dataset_id": dataset_id,
            "base_model": spec.get("id"),
            "base_model_family": spec.get("family"),
            "base_model_variant": spec.get("variant"),
            "base_model_display_name": spec.get("display_name"),
            "base_model_weights_name": spec.get("weights_name"),
            "device_mode": self.train_device.get().strip() or "auto",
        }
        version = version or self._selected_dataset_version_record() or self._active_dataset_version_record()
        if version is not None:
            payload["dataset_version_id"] = version.get("id")
        try:
            created = self.api.create_training_job(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training", str(exc))
            return
        if isinstance(created, dict):
            self._active_training_job_id = str(created.get("id") or "").strip() or self._active_training_job_id
        self.refresh_training_jobs()

    def cancel_training_job(self):
        index = self._selected_listbox_index(self.train_jobs)
        if index is None or index >= len(self._training_jobs):
            return
        try:
            self.api.cancel_training_job(self._training_jobs[index]["id"])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training", str(exc))
            return
        self.refresh_training_jobs()

    def on_training_selected(self):
        index = self._selected_listbox_index(self.train_jobs)
        if index is None or index >= len(self._training_jobs):
            self.training_summary.reset()
            self.training_detail.set_payload({})
            return
        item = self._training_jobs[index]
        self._active_training_job_id = str(item.get("id") or "").strip() or None
        self.training_summary.set_values(
            {
                "base_model": item.get("base_model_display_name") or item.get("base_model") or "-",
                "status": item.get("status") or "-",
                "accuracy": self._format_training_percent(
                    self._training_metric_value(item, "accuracy", "acc", "val_accuracy", "val_acc")
                ),
                "map_score": self._format_training_percent(
                    self._training_metric_value(item, "mAP", "map", "map50", "mAP50", "map_50", "mAP_50", "map_50_95", "mAP_50_95")
                ),
                "r2_score": self._format_training_decimal(
                    self._training_metric_value(item, "r2", "r2_score", "r_squared"),
                    digits=3,
                ),
                "error": self._training_error_summary(item),
            }
        )
        self.training_detail.set_payload(item)

    def _selected_base_model_spec(self) -> dict | None:
        selected = self.train_base_model.get().strip()
        if not selected:
            return None
        if selected in self._base_model_lookup:
            return self._base_model_lookup[selected]
        normalized = selected.lower().strip()
        return next((item for item in self._base_model_cache if str(item.get("id") or "").lower() == normalized), None)

    def _selected_dataset_version_spec(self) -> dict | None:
        selected = self.train_dataset_version.get().strip()
        if not selected:
            return None
        if selected in self._dataset_version_lookup:
            return self._dataset_version_lookup[selected]
        normalized = selected.lower().strip()
        return next(
            (
                item
                for item in self._dataset_version_cache
                if str(item.get("id") or "").lower() == normalized
                or str(item.get("display_label") or "").lower() == normalized
            ),
            None,
        )

    def _refresh_base_model_info(self) -> None:
        spec = self._selected_base_model_spec()
        if spec is None:
            self.train_base_model_info.set("Pilih base model dari katalog YOLOv5 / YOLOv11.")
            return
        self.train_base_model_info.set(
            f"{spec.get('display_name')} | {spec.get('family_label')} {spec.get('variant_label')} | runtime {spec.get('runtime')} | weights {spec.get('weights_name')}"
        )

    def refresh_base_models(self) -> None:
        try:
            items = self.api.list_base_models()
        except Exception:
            try:
                items = catalog_list_base_models()
            except Exception:
                items = []
        self._base_model_cache = list(items)
        self._base_model_lookup = {}
        values: list[str] = []
        preferred_value: str | None = None
        for item in items:
            label = str(item.get("display_label") or item.get("display_name") or item.get("id") or "").strip()
            if not label:
                continue
            values.append(label)
            self._base_model_lookup[label] = item
            if str(item.get("id") or "").lower() == "yolov5s":
                preferred_value = label
        self.train_base_model["values"] = values
        if values:
            current = self.train_base_model.get().strip()
            if current not in values:
                self.train_base_model.set(preferred_value or values[0])
        else:
            self.train_base_model.set("")
        self._refresh_base_model_info()

    def refresh_models(self):
        try:
            items = self.api.list_models()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Models", str(exc))
            return
        self._model_cache = items
        self.models_list.delete(0, "end")
        for item in items:
            self.models_list.insert("end", f"{item['id']} | {item['name']} | {item['path']}")

    def create_model(self):
        payload = {
            "name": self.model_name.get().strip(),
            "path": self.model_path.get().strip(),
            "meta_path": self.model_meta_path.get().strip(),
            "source": "manual",
            "runtime": self.model_runtime.get().strip(),
            "task": self.model_task.get().strip(),
            "class_names": [item.strip() for item in self.model_class_names.get().split(",") if item.strip()],
            "architecture_family": self.model_arch_family.get().strip(),
            "architecture_variant": self.model_arch_variant.get().strip(),
        }
        try:
            created = self.api.create_model(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Models", str(exc))
            return
        self.model_detail.set_payload(created)
        self.refresh_models()

    def _upload_model_file(self):
        pt_path = filedialog.askopenfilename(
            title="Pilih Model File (.pt)",
            filetypes=[("PyTorch Model", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not pt_path:
            return
        name = self.model_name.get().strip()
        if not name:
            messagebox.showwarning("Upload Model", "Isi field 'Name' terlebih dahulu.")
            return
        class_names = [item.strip() for item in self.model_class_names.get().split(",") if item.strip()]
        try:
            with open(pt_path, "rb") as f:
                content_b64 = base64.b64encode(f.read()).decode("ascii")
        except OSError as exc:
            messagebox.showerror("Upload Model", f"Gagal membaca file:\n{exc}")
            return
        file_name = Path(pt_path).name
        payload = {
            "name": name,
            "file_name": file_name,
            "content_b64": content_b64,
            "runtime": self.model_runtime.get().strip() or "ultralytics",
            "task": self.model_task.get().strip() or "detection",
            "class_names": class_names,
            "architecture_family": self.model_arch_family.get().strip() or None,
            "architecture_variant": self.model_arch_variant.get().strip() or None,
        }
        try:
            result = self.api.upload_model_file(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Upload Model", str(exc))
            return
        messagebox.showinfo("Upload Model", f"Model '{name}' berhasil diupload.\nDisimpan di: {result.get('saved_to')}")
        self.model_detail.set_payload(result)
        self.refresh_models()

    def on_model_selected(self):
        index = self._selected_listbox_index(self.models_list)
        if index is None or index >= len(self._model_cache):
            self.model_detail.set_payload({})
            return
        self.model_detail.set_payload(self._model_cache[index])

    # ── Sticker ROI Setup helpers ────────────────────────────────────

    def _sticker_setup_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Pilih Gambar Referensi Sticker",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Load Image", f"Gagal membaca gambar: {path}")
            return
        self.sticker_setup_picker.load_image(frame)
        self._sticker_setup_sync()

    def _sticker_setup_sync(self) -> None:
        try:
            sr = {
                "x": float(self.sticker_setup_roi_x.get() or 0),
                "y": float(self.sticker_setup_roi_y.get() or 0),
                "w": float(self.sticker_setup_roi_w.get() or 1),
                "h": float(self.sticker_setup_roi_h.get() or 1),
            }
        except ValueError:
            return
        try:
            cx = float(self.sticker_setup_cx_var.get() or 0.5)
            cy = float(self.sticker_setup_cy_var.get() or 0.5)
        except ValueError:
            cx, cy = 0.5, 0.5
        self.sticker_setup_picker.set_rois(sticker_roi=sr)
        self.sticker_setup_picker.set_expected_center(cx, cy)

    def _on_sticker_setup_center_changed(self, cx: float, cy: float) -> None:
        self.sticker_setup_cx_var.set(str(round(cx, 4)))
        self.sticker_setup_cy_var.set(str(round(cy, 4)))

    def _sticker_setup_copy(self) -> None:
        cx = self.sticker_setup_cx_var.get()
        cy = self.sticker_setup_cy_var.get()
        text = f"expected_center_x={cx}, expected_center_y={cy}"
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copy Values", f"Tersalin ke clipboard:\n{text}\n\nPaste ke Admin → Template Editor → Sticker tab.")

    # ── Part Ready Calibration ───────────────────────────────────────

    def choose_calibration_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.selected_calibration_path = path
            self.calibration_path_label.configure(text=Path(path).name)
            raw = Path(path).read_bytes()
            arr = np.frombuffer(raw, np.uint8)
            self.calibration_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if self.calibration_image is None:
                self.calibration_preview_info.set("Gagal membaca image calibration.")
                self.calibration_source_preview.reset()
                self.calibration_crop_preview.reset()
                return
            self._refresh_calibration_preview()

    def _calibration_roi(self) -> dict | None:
        values = {
            "x": self.calib_roi_x.get().strip(),
            "y": self.calib_roi_y.get().strip(),
            "w": self.calib_roi_w.get().strip(),
            "h": self.calib_roi_h.get().strip(),
        }
        if not any(values.values()):
            return None
        try:
            return {key: float(value) for key, value in values.items() if value}
        except ValueError as exc:
            raise ValueError("Calibration ROI harus numerik.") from exc

    def _calibration_roi_preview_payload(self) -> dict | None:
        values = {
            "x": self.calib_roi_x.get().strip(),
            "y": self.calib_roi_y.get().strip(),
            "w": self.calib_roi_w.get().strip(),
            "h": self.calib_roi_h.get().strip(),
        }
        if not any(values.values()):
            return None
        parsed: dict[str, float] = {}
        for key, raw in values.items():
            if not raw:
                return None
            try:
                value = float(raw)
            except ValueError:
                return None
            if key in {"x", "y"} and not 0.0 <= value <= 1.0:
                return None
            if key in {"w", "h"} and not 0.0 < value <= 1.0:
                return None
            parsed[key] = value
        return parsed

    def _on_calibration_roi_changed(self, _event=None) -> None:
        self.after_idle(self._refresh_calibration_preview)

    def _refresh_calibration_preview(self) -> None:
        if self.calibration_image is None:
            self.calibration_preview_info.set("Pilih image untuk melihat preview ROI.")
            self.calibration_source_preview.reset()
            self.calibration_crop_preview.reset()
            return

        overlay = self.calibration_image.copy()
        height, width = overlay.shape[:2]
        roi = self._calibration_roi_preview_payload()
        if not roi:
            self.calibration_source_preview.update_bgr(overlay)
            self.calibration_crop_preview.reset()
            self.calibration_preview_info.set(
                f"Image loaded: {width}x{height}. Isi ROI x/y/w/h lengkap untuk melihat crop preview."
            )
            return

        x = max(0, min(width - 1, int(float(roi["x"]) * width)))
        y = max(0, min(height - 1, int(float(roi["y"]) * height)))
        roi_w = max(1, int(float(roi["w"]) * width))
        roi_h = max(1, int(float(roi["h"]) * height))
        x2 = min(width, x + roi_w)
        y2 = min(height, y + roi_h)
        cv2.rectangle(overlay, (x, y), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            overlay,
            f"ROI {x},{y} {x2 - x}x{y2 - y}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        self.calibration_source_preview.update_bgr(overlay)

        try:
            cropped = CalibrationService.apply_roi(self.calibration_image, roi)
        except ValueError:
            self.calibration_crop_preview.reset()
            self.calibration_preview_info.set("ROI invalid atau menghasilkan crop kosong.")
            return

        self.calibration_crop_preview.update_bgr(cropped)
        self.calibration_preview_info.set(
            f"Preview memakai ROI yang sama dengan request backend. Source {width}x{height} -> crop {cropped.shape[1]}x{cropped.shape[0]}."
        )

    def compute_profile(self):
        if not self.selected_calibration_path:
            messagebox.showwarning("Calibrate", "Pilih image dulu.")
            return
        raw = Path(self.selected_calibration_path).read_bytes()
        try:
            payload = {
                "image_b64": base64.b64encode(raw).decode("ascii"),
                "colorspace": "LAB",
            }
            roi = self._calibration_roi()
            if roi:
                payload["roi"] = roi
            self.computed_profile = self.api.compute_color_profile(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Calibrate", str(exc))
            return
        self.calibration_editor.set_payload(self.computed_profile)
        self._refresh_calibration_preview()

    def save_profile(self):
        if not self.computed_profile:
            messagebox.showwarning("Calibrate", "Compute profile dulu.")
            return
        try:
            created = self.api.save_profile({"name": self.profile_name.get().strip() or "New Profile", "profile": self.computed_profile})
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Calibrate", str(exc))
            return
        self.profile_detail.set_payload(created)
        self.refresh_profiles()
        messagebox.showinfo("Calibrate", "Profile saved.")

    def refresh_profiles(self):
        try:
            items = self.api.list_profiles()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Profiles", str(exc))
            return
        self._profile_cache = items
        self.profiles_list.delete(0, "end")
        for item in items:
            profile = item.get("profile") or {}
            self.profiles_list.insert("end", f"{item['id']} | {item['name']} | {profile.get('colorspace') or 'LAB'}")

    def on_profile_selected(self):
        index = self._selected_listbox_index(self.profiles_list)
        if index is None or index >= len(self._profile_cache):
            self.profile_detail.set_payload({})
            return
        self.profile_detail.set_payload(self._profile_cache[index])

    def delete_selected_profile(self):
        index = self._selected_listbox_index(self.profiles_list)
        if index is None or index >= len(self._profile_cache):
            return
        try:
            self.api.delete_profile(self._profile_cache[index]["id"])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Profiles", str(exc))
            return
        self.refresh_profiles()

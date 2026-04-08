from __future__ import annotations

import base64
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from backend.app.services.calibration import CalibrationService
from backend.app.core.model_catalog import list_base_models as catalog_list_base_models
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
        self._model_cache: list[dict] = []
        self._profile_cache: list[dict] = []

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

        self._build_data_tab()
        self._build_training_tab()
        self._build_models_tab()
        self._build_calibration_tab()

        self.refresh_datasets()
        self.refresh_base_models()
        self.refresh_augment_jobs()
        self.refresh_training_jobs()
        self.refresh_models()
        self.refresh_profiles()

    def _build_data_tab(self) -> None:
        container = ttk.Panedwindow(self.data_tab, orient="vertical")
        container.pack(fill="both", expand=True, padx=6, pady=6)

        top = ttk.Panedwindow(container, orient="horizontal")
        bottom = ttk.Frame(container, padding=8)
        container.add(top, weight=3)
        container.add(bottom, weight=2)

        dataset_panel = ttk.Frame(top, padding=8)
        upload_panel = ttk.Frame(top, padding=8)
        top.add(dataset_panel, weight=2)
        top.add(upload_panel, weight=2)

        ttk.Label(dataset_panel, text="Datasets", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            dataset_panel,
            text="Pilih dataset untuk sinkron ke upload, annotation, augment, dan training.",
            foreground="#475569",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))
        self.dataset_list = tk.Listbox(dataset_panel, height=14)
        self.dataset_list.pack(fill="both", expand=True)
        self.dataset_list.bind("<<ListboxSelect>>", lambda _event: self.on_dataset_selected())

        dataset_form = ttk.Frame(dataset_panel)
        dataset_form.pack(fill="x", pady=(8, 0))
        self.dataset_name = ttk.Entry(dataset_form)
        self.dataset_desc = ttk.Entry(dataset_form)
        self._grid_entry(dataset_form, 0, 0, "Name", self.dataset_name)
        self._grid_entry(dataset_form, 1, 0, "Description", self.dataset_desc)
        action_bar = ttk.Frame(dataset_panel)
        action_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(action_bar, text="Create", command=self.create_dataset).pack(side="left")
        ttk.Button(action_bar, text="Delete Selected", command=self.delete_dataset).pack(side="left", padx=6)
        ttk.Button(action_bar, text="Refresh", command=self.refresh_datasets).pack(side="left")

        self.dataset_summary = LabeledValuePanel(
            dataset_panel,
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

        ttk.Label(upload_panel, text="Upload and Browse Files", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            upload_panel,
            text="Upload satu atau banyak file ke folder `images`, `labels`, atau `exports`, lalu lihat isi dataset aktif di browser file.",
            foreground="#475569",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))
        upload_form = ttk.Frame(upload_panel)
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

        browser_frame = ttk.LabelFrame(upload_panel, text="Dataset Files", padding=8)
        browser_frame.pack(fill="both", expand=True, pady=(10, 0))
        toolbar = ttk.Frame(browser_frame)
        toolbar.pack(fill="x")
        self.browser_target = ttk.Combobox(toolbar, values=["images", "labels", "exports"], state="readonly")
        self.browser_target.set("images")
        self.browser_target.pack(side="left")
        ttk.Button(toolbar, text="Refresh Files", command=self.refresh_dataset_files).pack(side="left", padx=6)
        self.dataset_files = tk.Listbox(browser_frame)
        self.dataset_files.pack(fill="both", expand=True, pady=(8, 0))

        version_frame = ttk.LabelFrame(upload_panel, text="Dataset Versions", padding=8)
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
        self.dataset_versions.bind("<<ListboxSelect>>", lambda _event: self.on_dataset_version_selected())
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
        self.dataset_version_detail = JsonEditor(version_frame, "Version Detail", {})
        self.dataset_version_detail.pack(fill="both", expand=True, pady=(8, 0))

        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=2)
        bottom.rowconfigure(2, weight=1)
        ttk.Label(bottom, text="Annotation Workflow", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            bottom,
            text="Dataset terpilih akan otomatis mengisi Dataset ID annotation. Pilih image lalu load/save labels JSON.",
            foreground="#475569",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))

        annot_left = ttk.Frame(bottom)
        annot_left.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        annot_right = ttk.Frame(bottom)
        annot_right.grid(row=2, column=1, sticky="nsew")

        annot_form = ttk.Frame(annot_left)
        annot_form.pack(fill="x")
        self.annot_dataset = ttk.Entry(annot_form)
        self.annot_image = ttk.Entry(annot_form)
        self._grid_entry(annot_form, 0, 0, "Dataset ID", self.annot_dataset)
        self._grid_entry(annot_form, 1, 0, "Image Name", self.annot_image)
        ttk.Button(annot_form, text="Load Annotation", command=self.load_annotation).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(annot_form, text="Save Annotation", command=self.save_annotation).grid(row=2, column=1, sticky="e", pady=(6, 0))
        annot_form.columnconfigure(1, weight=1)

        quick_label = ttk.LabelFrame(annot_left, text="Quick Label Builder", padding=8)
        quick_label.pack(fill="x", pady=(10, 0))
        quick_label.columnconfigure(1, weight=1)
        quick_label.columnconfigure(3, weight=1)
        self.annot_shape = ttk.Combobox(quick_label, values=["bbox", "polygon"], state="readonly")
        self.annot_shape.set("bbox")
        self.annot_label_class = ttk.Entry(quick_label)
        self.annot_bbox_x = ttk.Entry(quick_label)
        self.annot_bbox_y = ttk.Entry(quick_label)
        self.annot_bbox_w = ttk.Entry(quick_label)
        self.annot_bbox_h = ttk.Entry(quick_label)
        self.annot_polygon_points = ttk.Entry(quick_label)
        for entry, default in (
            (self.annot_bbox_x, "0.1"),
            (self.annot_bbox_y, "0.1"),
            (self.annot_bbox_w, "0.2"),
            (self.annot_bbox_h, "0.2"),
        ):
            entry.insert(0, default)
        self._grid_entry(quick_label, 0, 0, "Shape", self.annot_shape)
        self._grid_entry(quick_label, 0, 2, "Class", self.annot_label_class)
        self._grid_entry(quick_label, 1, 0, "BBox x", self.annot_bbox_x)
        self._grid_entry(quick_label, 1, 2, "BBox y", self.annot_bbox_y)
        self._grid_entry(quick_label, 2, 0, "BBox w", self.annot_bbox_w)
        self._grid_entry(quick_label, 2, 2, "BBox h", self.annot_bbox_h)
        ttk.Label(quick_label, text="Polygon points (x,y; x,y; ...)", foreground="#475569").grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.annot_polygon_points.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        quick_action_bar = ttk.Frame(quick_label)
        quick_action_bar.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Button(quick_action_bar, text="Add Label", command=self.append_annotation_label).pack(side="left")
        ttk.Label(quick_action_bar, text="BBox uses normalized coordinates 0..1.", foreground="#64748b").pack(side="left", padx=8)

        image_frame = ttk.LabelFrame(annot_left, text="Images", padding=8)
        image_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.annot_images = tk.Listbox(image_frame)
        self.annot_images.pack(fill="both", expand=True)
        self.annot_images.bind("<<ListboxSelect>>", lambda _event: self.on_annotation_image_selected())

        self.annot_editor = JsonEditor(annot_right, "Labels JSON", {"schema_version": 1, "image_name": "", "labels": []})
        self.annot_editor.pack(fill="both", expand=True)

    def _build_training_tab(self) -> None:
        container = ttk.Panedwindow(self.training_tab, orient="horizontal")
        container.pack(fill="both", expand=True, padx=6, pady=6)

        augment_panel = ttk.Frame(container, padding=8)
        train_panel = ttk.Frame(container, padding=8)
        container.add(augment_panel, weight=1)
        container.add(train_panel, weight=2)

        ttk.Label(augment_panel, text="Augment Jobs", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            augment_panel,
            text="MVP ini masih metadata-only, tapi panel ini tetap memisahkan pekerjaan augment dari training.",
            foreground="#475569",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))
        augment_form = ttk.LabelFrame(augment_panel, text="Recipe", padding=8)
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

        self.augment_jobs = tk.Listbox(augment_panel, height=12)
        self.augment_jobs.pack(fill="both", expand=True, pady=(8, 0))

        ttk.Label(train_panel, text="Training Jobs", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            train_panel,
            text="Training job untuk sticker detector. Pilih dataset, base model, dan device; backend akan memilih GPU dulu lalu fallback ke CPU bila GPU tidak tersedia.",
            foreground="#475569",
            wraplength=540,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        top = ttk.Frame(train_panel)
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
        ttk.Label(train_panel, textvariable=self.train_base_model_info, foreground="#475569", wraplength=540, justify="left").pack(anchor="w", pady=(4, 0))
        self.train_base_model.bind("<<ComboboxSelected>>", lambda _event: self._refresh_base_model_info())

        version_panel = ttk.LabelFrame(train_panel, text="Dataset Version", padding=8)
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
        self.train_dataset_version.bind("<<ComboboxSelected>>", lambda _event: self._refresh_train_dataset_version_info())

        button_bar = ttk.Frame(train_panel)
        button_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(button_bar, text="Start Training", command=self.create_training_job).pack(side="left")
        ttk.Button(button_bar, text="Cancel Selected", command=self.cancel_training_job).pack(side="left", padx=6)
        ttk.Button(button_bar, text="Refresh", command=self.refresh_training_jobs).pack(side="left")

        lower = ttk.Panedwindow(train_panel, orient="horizontal")
        lower.pack(fill="both", expand=True, pady=(10, 0))
        jobs_panel = ttk.Frame(lower)
        detail_panel = ttk.Frame(lower)
        lower.add(jobs_panel, weight=2)
        lower.add(detail_panel, weight=3)

        self.train_jobs = tk.Listbox(jobs_panel)
        self.train_jobs.pack(fill="both", expand=True)
        self.train_jobs.bind("<<ListboxSelect>>", lambda _event: self.on_training_selected())

        self.training_summary = LabeledValuePanel(
            detail_panel,
            "Training Summary",
            [
                ("id", "Job ID"),
                ("dataset_id", "Dataset"),
                ("status", "Status"),
                ("base_model", "Base Model"),
                ("base_model_display_name", "Model Name"),
                ("base_model_family", "Family"),
                ("base_model_variant", "Variant"),
                ("requested_device_mode", "Requested Device"),
                ("effective_device", "Effective Device"),
                ("device_backend", "Device Backend"),
                ("device_fallback_reason", "Fallback Reason"),
                ("trained_model_path", "Output Model"),
                ("created_at", "Created"),
                ("finished_at", "Finished"),
            ],
        )
        self.training_summary.pack(fill="x")
        self.training_detail = JsonEditor(detail_panel, "Training Job Detail", {})
        self.training_detail.pack(fill="both", expand=True, pady=(10, 0))

    def _build_models_tab(self) -> None:
        container = ttk.Panedwindow(self.models_tab, orient="horizontal")
        container.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=2)
        container.add(right, weight=3)

        ttk.Label(left, text="Model Registry", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.models_list = tk.Listbox(left)
        self.models_list.pack(fill="both", expand=True, pady=(8, 0))
        self.models_list.bind("<<ListboxSelect>>", lambda _event: self.on_model_selected())
        ttk.Button(left, text="Refresh", command=self.refresh_models).pack(anchor="e", pady=(8, 0))

        ttk.Label(right, text="Register Sticker Model", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            right,
            text="Registry ini menjadi sumber resmi model sticker yang akan dipakai template. Simpan `path`, `meta_path`, runtime, task, dan daftar class.",
            foreground="#475569",
            wraplength=620,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 8))

        form = ttk.LabelFrame(right, text="Model Form", padding=10)
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

        self.model_detail = JsonEditor(right, "Model Detail", {})
        self.model_detail.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

    def _build_calibration_tab(self) -> None:
        self.calibration_scroller = ScrollableFrame(self.calibration_tab)
        self.calibration_scroller.pack(fill="both", expand=True, padx=6, pady=6)

        container = ttk.Panedwindow(self.calibration_scroller.body, orient="horizontal")
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container, padding=8)
        right_outer = ttk.Frame(container)
        right_scroller = ScrollableFrame(right_outer)
        right_scroller.pack(fill="both", expand=True)
        right = ttk.Frame(right_scroller.body, padding=8)
        right.pack(fill="both", expand=True)
        container.add(left, weight=2)
        container.add(right_outer, weight=2)

        ttk.Label(left, text="Part Ready Color Calibration", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            left,
            text="Flow: pilih image -> optional ROI -> compute profile -> save profile -> pakai `profile_id` itu di template part-ready.",
            foreground="#475569",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        control = ttk.LabelFrame(left, text="Compute Profile", padding=10)
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

        preview_frame = ttk.LabelFrame(left, text="ROI Preview", padding=8)
        preview_frame.pack(fill="both", expand=True, pady=(10, 0))
        preview_frame.columnconfigure(0, weight=3)
        preview_frame.columnconfigure(1, weight=2)
        preview_frame.rowconfigure(0, weight=1)
        self.calibration_source_preview = LiveView(preview_frame, "Source Image", size=(520, 260))
        self.calibration_source_preview.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.calibration_crop_preview = LiveView(preview_frame, "ROI Crop", size=(300, 260))
        self.calibration_crop_preview.grid(row=0, column=1, sticky="nsew")
        self.calibration_preview_info = tk.StringVar(value="Pilih image untuk melihat preview ROI.")
        ttk.Label(
            left,
            textvariable=self.calibration_preview_info,
            foreground="#475569",
            wraplength=640,
            justify="left",
        ).pack(fill="x", pady=(6, 0))

        self.calibration_editor = JsonEditor(left, "Computed Profile", {})
        self.calibration_editor.pack(fill="both", expand=True, pady=(10, 0))

        ttk.Label(right, text="Saved Profiles", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.profiles_list = tk.Listbox(right, height=6)
        self.profiles_list.pack(fill="x", pady=(8, 0))
        self.profiles_list.bind("<<ListboxSelect>>", lambda _event: self.on_profile_selected())
        action_bar = ttk.Frame(right)
        action_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(action_bar, text="Refresh", command=self.refresh_profiles).pack(side="left")
        ttk.Button(action_bar, text="Delete Selected", command=self.delete_selected_profile).pack(side="left", padx=6)
        self.profile_detail = JsonEditor(right, "Profile Detail", {})
        self.profile_detail.pack(fill="x", pady=(10, 0))

        # ── Sticker ROI Visual Setup ─────────────────────────────────
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=(12, 8))
        ttk.Label(right, text="Sticker ROI Visual Setup", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            right,
            text="Muat gambar referensi → atur ROI sticker → klik pada area kuning untuk set expected center.",
            foreground="#475569",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        sticker_setup_ctrl = ttk.LabelFrame(right, text="Sticker ROI (rasio 0-1)", padding=8)
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

        self.sticker_setup_picker = RoiPickerCanvas(right, "Visual Picker — klik area kuning", size=(340, 200))
        self.sticker_setup_picker.pack(fill="x", pady=(8, 0))
        self.sticker_setup_picker.on_center_changed = self._on_sticker_setup_center_changed

        center_bar = ttk.Frame(right)
        center_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(center_bar, text="Expected Center X:").pack(side="left")
        self.sticker_setup_cx_var = tk.StringVar(value="0.5")
        self.sticker_setup_cy_var = tk.StringVar(value="0.5")
        ttk.Entry(center_bar, textvariable=self.sticker_setup_cx_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(center_bar, text="Y:").pack(side="left")
        ttk.Entry(center_bar, textvariable=self.sticker_setup_cy_var, width=7).pack(side="left", padx=(4, 0))

        sticker_btn_bar = ttk.Frame(right)
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

    def _current_dataset_id(self) -> str | None:
        for candidate in (
            self._selected_dataset_id(),
            self.upload_dataset_id.get().strip(),
            self.annot_dataset.get().strip(),
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

    def _version_lookup_key(self, version: dict) -> str:
        return str(version.get("display_label") or version.get("id") or "").strip()

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
        dataset = self._selected_dataset_record()
        dataset_id = str(dataset.get("id")) if dataset else None
        if not dataset_id:
            return
        for widget in (self.upload_dataset_id, self.annot_dataset, self.augment_dataset, self.train_dataset):
            widget.delete(0, "end")
            widget.insert(0, dataset_id)
        self._update_dataset_summary(dataset)
        self.refresh_dataset_files()
        self.refresh_annotation_images()
        self.refresh_dataset_versions()

    def refresh_datasets(self):
        try:
            items = self.api.list_datasets()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset", str(exc))
            return
        self._dataset_cache = items
        self.dataset_list.delete(0, "end")
        for item in items:
            summary = f"{item.get('image_count', 0)} imgs / {item.get('annotated_image_count', 0)} ann / {item.get('augmented_count', 0)} aug"
            self.dataset_list.insert("end", f"{item['id']} | {item['name']} | {summary}")
        selected_dataset = self._selected_dataset_record()
        self.dataset_summary.reset()
        self._update_dataset_summary(selected_dataset)
        if selected_dataset:
            self.refresh_dataset_versions()
        else:
            self._update_dataset_version_summary(None)

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
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        if not messagebox.askyesno("Dataset", "Delete selected dataset?"):
            return
        try:
            self.api.delete_dataset(dataset_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset", str(exc))
            return
        self.dataset_list.selection_clear(0, "end")
        for widget in (self.upload_dataset_id, self.annot_dataset, self.augment_dataset, self.train_dataset):
            widget.delete(0, "end")
        self.train_dataset_version.set("")
        self.refresh_datasets()
        self.dataset_files.delete(0, "end")
        self.annot_images.delete(0, "end")
        self.dataset_summary.reset()
        self._dataset_version_cache = []
        self._dataset_version_lookup = {}
        self.dataset_versions.delete(0, "end")
        self._update_dataset_version_summary(None)

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

    def refresh_dataset_versions(self):
        dataset_id = self._current_dataset_id()
        self.dataset_versions.delete(0, "end")
        self._dataset_version_cache = []
        self._dataset_version_lookup = {}
        if not dataset_id:
            self.train_dataset_version["values"] = []
            self._update_dataset_version_summary(None)
            return

        try:
            items = self.api.list_dataset_versions(dataset_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset Versions", str(exc))
            return

        self._dataset_version_cache = items
        values: list[str] = []
        for item in items:
            display_label = str(item.get("display_label") or item.get("id") or "").strip()
            if not display_label:
                continue
            values.append(display_label)
            self._dataset_version_lookup[display_label] = item
            self._dataset_version_lookup[str(item.get("id") or "")] = item
            self.dataset_versions.insert(
                "end",
                f"{display_label} | {item.get('export_format') or 'yolo'} | {item.get('export_root') or '-'}",
            )

        self.train_dataset_version["values"] = values
        if values:
            current = self.train_dataset_version.get().strip()
            if current not in values:
                self.train_dataset_version.set(values[0])
            self.dataset_versions.selection_clear(0, "end")
            self.dataset_versions.selection_set(0)
            self.dataset_versions.see(0)
            self.on_dataset_version_selected()
        else:
            self.train_dataset_version.set("")
            self._update_dataset_version_summary(None)
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
        self.refresh_dataset_versions()
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
        self.refresh_dataset_versions()
        messagebox.showinfo("Dataset Versions", f"Export version '{refreshed.get('display_label')}' selesai diperbarui.")

    def refresh_annotation_images(self):
        dataset_id = self.annot_dataset.get().strip() or self._selected_dataset_id()
        self.annot_images.delete(0, "end")
        if not dataset_id:
            return
        try:
            items = self.api.list_dataset_files(dataset_id, "images")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Annotate", str(exc))
            return
        for item in items:
            marker = "✓" if item.get("annotation_exists") else "•"
            self.annot_images.insert("end", f"{marker} {item['name']}")

    def on_annotation_image_selected(self):
        index = self._selected_listbox_index(self.annot_images)
        if index is None:
            return
        image_name = self.annot_images.get(index).split(" ", 1)[-1]
        self.annot_image.delete(0, "end")
        self.annot_image.insert(0, image_name)

    def on_dataset_version_selected(self) -> None:
        version = self._selected_dataset_version_record()
        if not version:
            self._update_dataset_version_summary(None)
            return
        self._update_dataset_version_summary(version)

    def _annotation_editor_payload(self) -> dict:
        try:
            payload = self.annot_editor.get_payload()
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            record = dict(payload)
        elif isinstance(payload, list):
            record = {"schema_version": 1, "image_name": self.annot_image.get().strip(), "labels": payload}
        else:
            record = {"schema_version": 1, "image_name": self.annot_image.get().strip(), "labels": []}
        record.setdefault("schema_version", 1)
        record.setdefault("image_name", self.annot_image.get().strip())
        labels = record.get("labels")
        if not isinstance(labels, list):
            labels = []
        record["labels"] = labels
        return record

    def _parse_polygon_points(self, raw_points: str) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        for chunk in (piece.strip() for piece in raw_points.split(";")):
            if not chunk:
                continue
            if "," not in chunk:
                raise ValueError("Polygon points harus berformat x,y; x,y; ...")
            x_raw, y_raw = (piece.strip() for piece in chunk.split(",", 1))
            points.append({"x": float(x_raw), "y": float(y_raw)})
        if len(points) < 3:
            raise ValueError("Polygon membutuhkan minimal 3 titik.")
        return points

    def append_annotation_label(self):
        class_name = self.annot_label_class.get().strip()
        if not class_name:
            messagebox.showwarning("Annotate", "Class Name wajib diisi.")
            return
        shape = (self.annot_shape.get().strip() or "bbox").lower()
        payload = self._annotation_editor_payload()
        labels = list(payload.get("labels") or [])
        try:
            if shape == "polygon":
                label = {
                    "type": "polygon",
                    "shape_type": "polygon",
                    "class": class_name,
                    "class_name": class_name,
                    "points": self._parse_polygon_points(self.annot_polygon_points.get().strip()),
                    "normalized": True,
                    "source": "manual",
                }
            else:
                bbox = {
                    "x": float(self.annot_bbox_x.get().strip()),
                    "y": float(self.annot_bbox_y.get().strip()),
                    "w": float(self.annot_bbox_w.get().strip()),
                    "h": float(self.annot_bbox_h.get().strip()),
                }
                label = {
                    "type": "bbox",
                    "shape_type": "bbox",
                    "class": class_name,
                    "class_name": class_name,
                    "bbox": bbox,
                    "normalized": True,
                    "source": "manual",
                }
        except ValueError as exc:
            messagebox.showerror("Annotate", str(exc))
            return
        labels.append(label)
        payload["labels"] = labels
        payload["label_count"] = len(labels)
        self.annot_editor.set_payload(payload)

    def _annotation_labels(self) -> list[dict]:
        payload = self._annotation_editor_payload()
        labels = payload.get("labels")
        return list(labels) if isinstance(labels, list) else []

    def load_annotation(self):
        dataset_id = self.annot_dataset.get().strip()
        image_name = self.annot_image.get().strip()
        if not dataset_id or not image_name:
            messagebox.showwarning("Annotate", "Dataset ID dan Image Name wajib diisi.")
            return
        try:
            payload = self.api.get_annotation(dataset_id, image_name)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Annotate", str(exc))
            return
        self.annot_editor.set_payload(payload)

    def save_annotation(self):
        dataset_id = self.annot_dataset.get().strip()
        image_name = self.annot_image.get().strip()
        if not dataset_id or not image_name:
            messagebox.showwarning("Annotate", "Dataset ID dan Image Name wajib diisi.")
            return
        try:
            payload = self._annotation_editor_payload()
            self.api.save_annotation(dataset_id, image_name, payload.get("labels") or [])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Annotate", str(exc))
            return
        messagebox.showinfo("Annotate", "Annotation saved.")

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
        if version is not None:
            payload["dataset_version_id"] = version.get("id")
        try:
            self.api.create_training_job(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training", str(exc))
            return
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
        self.training_summary.set_values(
            {
                "id": item.get("id"),
                "dataset_id": item.get("dataset_id"),
                "dataset_version_id": item.get("dataset_version_id"),
                "dataset_version_number": item.get("dataset_version_number"),
                "dataset_version_display_label": item.get("dataset_version_display_label") or item.get("dataset_version_name") or item.get("dataset_version_id"),
                "status": item.get("status"),
                "base_model": item.get("base_model"),
                "base_model_display_name": item.get("base_model_display_name"),
                "base_model_family": item.get("base_model_family"),
                "base_model_variant": item.get("base_model_variant"),
                "requested_device_mode": item.get("requested_device_mode") or item.get("device_mode") or item.get("params", {}).get("device_mode"),
                "effective_device": item.get("effective_device"),
                "device_backend": item.get("device_backend"),
                "device_fallback_reason": item.get("device_fallback_reason"),
                "trained_model_path": item.get("trained_model_path"),
                "created_at": item.get("created_at"),
                "finished_at": item.get("finished_at"),
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

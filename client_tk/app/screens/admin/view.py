from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from client_tk.app.components.template_forms import JsonEditor, LabeledValuePanel, StatCard, TemplateEditorForm


class AdminScreen(ttk.Frame):
    def __init__(self, master, api_client, session_state):
        super().__init__(master, padding=8)
        self.api = api_client
        self.state = session_state
        self.current_template_id: int | None = None
        self._template_summary_lookup: dict[str, dict] = {}

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.templates_tab = ttk.Frame(notebook)
        self.deployments_tab = ttk.Frame(notebook)
        self.users_tab = ttk.Frame(notebook)
        self.results_tab = ttk.Frame(notebook)
        self.dashboard_tab = ttk.Frame(notebook)

        notebook.add(self.templates_tab, text="Templates")
        notebook.add(self.deployments_tab, text="Deployments")
        notebook.add(self.users_tab, text="Users")
        notebook.add(self.results_tab, text="Results")
        notebook.add(self.dashboard_tab, text="Dashboard")

        self._build_templates_tab()
        self._build_deployments_tab()
        self._build_users_tab()
        self._build_results_tab()
        self._build_dashboard_tab()

        self.refresh_template_dependencies()
        self.refresh_templates()
        self.refresh_deployments()
        self.refresh_users()
        self.refresh_results()
        self.refresh_dashboard()

    def _build_templates_tab(self) -> None:
        container = ttk.Panedwindow(self.templates_tab, orient="horizontal")
        container.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=1)
        container.add(right, weight=3)

        ttk.Label(left, text="Template Versions", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(
            left,
            text="Pilih template untuk edit structured form dua ROI. Update akan membuat version baru.",
            wraplength=280,
            justify="left",
            foreground="#475569",
        ).pack(anchor="w", pady=(2, 8))

        self.template_list = tk.Listbox(left, width=42)
        self.template_list.pack(fill="both", expand=True)
        self.template_list.bind("<Double-Button-1>", lambda _event: self.load_selected_template())

        action_bar = ttk.Frame(left)
        action_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(action_bar, text="Refresh", command=self.refresh_templates).pack(side="left")
        ttk.Button(action_bar, text="New", command=self.new_template).pack(side="left", padx=6)
        ttk.Button(action_bar, text="Load", command=self.load_selected_template).pack(side="left")
        ttk.Button(action_bar, text="Delete", command=self.delete_selected_template).pack(side="left", padx=6)

        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="Template Editor", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            right,
            text="Structured form memisahkan `part_ready_roi`, `sticker_roi`, model, dan validator. Raw JSON tetap tersedia untuk advanced editing.",
            foreground="#475569",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 8))

        editor_tabs = ttk.Notebook(right)
        editor_tabs.grid(row=2, column=0, sticky="nsew")
        right.rowconfigure(2, weight=1)

        structured_tab = ttk.Frame(editor_tabs, padding=4)
        raw_tab = ttk.Frame(editor_tabs, padding=4)
        editor_tabs.add(structured_tab, text="Structured Form")
        editor_tabs.add(raw_tab, text="Raw JSON")

        self.template_form = TemplateEditorForm(structured_tab)
        self.template_form._api_client_ref = self.api
        self.template_form.pack(fill="both", expand=True)
        self.template_raw_editor = JsonEditor(raw_tab, "Template Raw JSON", {})
        self.template_raw_editor.pack(fill="both", expand=True)

        footer = ttk.Frame(right)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(footer, text="Preview Raw JSON", command=self.preview_template_json).pack(side="left")
        ttk.Button(footer, text="Load Form From Raw JSON", command=self.apply_raw_template).pack(side="left", padx=6)
        ttk.Button(footer, text="Save Template", command=self.save_template).pack(side="right")

    def _build_deployments_tab(self) -> None:
        container = ttk.Panedwindow(self.deployments_tab, orient="horizontal")
        container.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=2)
        container.add(right, weight=2)

        ttk.Label(left, text="Active Deployments", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.deployment_list = tk.Listbox(left, height=18)
        self.deployment_list.pack(fill="both", expand=True, pady=(8, 0))

        action_bar = ttk.Frame(left)
        action_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(action_bar, text="Refresh", command=self.refresh_deployments).pack(side="left")
        ttk.Button(action_bar, text="Deactivate Selected", command=self.deactivate_selected_deployment).pack(side="left", padx=6)

        ttk.Label(right, text="Deploy Template to Line / Station", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            right,
            text="Pilih template aktif dan assign ke pasangan line/station. Operator akan menarik deployment ini saat `Load Deployment`.",
            wraplength=420,
            foreground="#475569",
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))

        form = ttk.LabelFrame(right, text="Deployment Form", padding=10)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        self.dep_template_choice = tk.StringVar()
        self.dep_template_selector = ttk.Combobox(form, textvariable=self.dep_template_choice, state="readonly")
        ttk.Label(form, text="Template").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.dep_template_selector.grid(row=0, column=1, columnspan=3, sticky="ew", pady=4)
        self.dep_template_selector.bind("<<ComboboxSelected>>", self._on_deployment_template_selected)

        self.dep_template_id = ttk.Entry(form)
        self.dep_version_id = ttk.Entry(form)
        self.dep_line = ttk.Entry(form)
        self.dep_station = ttk.Entry(form)
        self._grid_entry(form, 1, 0, "Template ID", self.dep_template_id)
        self._grid_entry(form, 1, 2, "Version ID", self.dep_version_id)
        self._grid_entry(form, 2, 0, "Line", self.dep_line)
        self._grid_entry(form, 2, 2, "Station", self.dep_station)

        footer = ttk.Frame(right)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(footer, text="Deploy", command=self.deploy_template).pack(side="right")

    def _build_users_tab(self) -> None:
        container = ttk.Panedwindow(self.users_tab, orient="horizontal")
        container.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=2)
        container.add(right, weight=1)

        ttk.Label(left, text="Users", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.users_list = tk.Listbox(left, height=18)
        self.users_list.pack(fill="both", expand=True, pady=(8, 0))

        action_bar = ttk.Frame(left)
        action_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(action_bar, text="Refresh", command=self.refresh_users).pack(side="left")
        ttk.Button(action_bar, text="Enable Selected", command=lambda: self.set_selected_user_active(True)).pack(side="left", padx=6)
        ttk.Button(action_bar, text="Disable Selected", command=lambda: self.set_selected_user_active(False)).pack(side="left")

        form = ttk.LabelFrame(right, text="Create User", padding=10)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.user_name = ttk.Entry(form)
        self.user_pass = ttk.Entry(form, show="*")
        self.user_role = ttk.Combobox(form, values=["admin", "operator", "engineer"], state="readonly")
        self.user_role.set("operator")
        self._grid_entry(form, 0, 0, "Username", self.user_name)
        self._grid_entry(form, 1, 0, "Password", self.user_pass)
        ttk.Label(form, text="Role").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.user_role.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(form, text="Create User", command=self.create_user).grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))

    def _build_results_tab(self) -> None:
        self.results_tab.columnconfigure(0, weight=1)
        self.results_tab.rowconfigure(1, weight=1)

        filters = ttk.LabelFrame(self.results_tab, text="Filters", padding=10)
        filters.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        for index in range(10):
            filters.columnconfigure(index, weight=1 if index % 2 else 0)

        self.result_filter_line = ttk.Entry(filters)
        self.result_filter_station = ttk.Entry(filters)
        self.result_filter_part = ttk.Entry(filters)
        self.result_filter_template = ttk.Entry(filters)
        self.result_filter_decision = ttk.Combobox(filters, values=["", "ACCEPT", "REJECT"], state="readonly")
        self.result_filter_decision.set("")
        self._grid_entry(filters, 0, 0, "Line", self.result_filter_line)
        self._grid_entry(filters, 0, 2, "Station", self.result_filter_station)
        self._grid_entry(filters, 0, 4, "Part", self.result_filter_part)
        self._grid_entry(filters, 1, 0, "Template Ver", self.result_filter_template)
        ttk.Label(filters, text="Decision").grid(row=1, column=2, sticky="w", padx=(0, 8), pady=4)
        self.result_filter_decision.grid(row=1, column=3, sticky="ew", pady=4)
        ttk.Button(filters, text="Refresh", command=self.refresh_results).grid(row=1, column=5, sticky="e", pady=4)
        ttk.Button(filters, text="Export CSV", command=self._export_csv).grid(row=1, column=6, sticky="e", padx=(6, 0), pady=4)

        container = ttk.Panedwindow(self.results_tab, orient="horizontal")
        container.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=2)
        container.add(right, weight=3)

        ttk.Label(left, text="Inspection Results", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.results_list = tk.Listbox(left, width=54)
        self.results_list.pack(fill="both", expand=True, pady=(8, 0))
        self.results_list.bind("<Double-Button-1>", lambda _event: self.open_result())
        ttk.Button(left, text="Open Selected", command=self.open_result).pack(anchor="e", pady=(8, 0))

        right.columnconfigure(0, weight=1)
        self.result_summary = LabeledValuePanel(
            right,
            "Result Summary",
            [
                ("decision", "Decision"),
                ("reason", "Reason"),
                ("part_name", "Part"),
                ("line_id", "Line"),
                ("station_id", "Station"),
                ("detected_class", "Detected"),
                ("expected_class", "Expected"),
                ("sticker_backend", "Backend"),
                ("part_ready_status", "Part Ready"),
                ("part_ready_match_ratio", "Match Ratio"),
                ("sticker_confidence", "Sticker Conf"),
                ("push_status", "Push Status"),
            ],
            columns=2,
        )
        self.result_summary.grid(row=0, column=0, sticky="ew")
        self.result_detail = JsonEditor(right, "Raw Result Payload", {})
        self.result_detail.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        right.rowconfigure(1, weight=1)

    def _build_dashboard_tab(self) -> None:
        self.dashboard_tab.columnconfigure(0, weight=1)
        self.dashboard_tab.rowconfigure(2, weight=1)

        filters = ttk.LabelFrame(self.dashboard_tab, text="Dashboard Filters", padding=10)
        filters.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        for index in range(12):
            filters.columnconfigure(index, weight=1 if index % 2 else 0)

        self.dashboard_filter_line = ttk.Entry(filters)
        self.dashboard_filter_station = ttk.Entry(filters)
        self.dashboard_filter_part = ttk.Entry(filters)
        self.dashboard_filter_template = ttk.Entry(filters)
        self.dashboard_granularity = ttk.Combobox(filters, values=["minute", "hour", "day"], state="readonly")
        self.dashboard_granularity.set("hour")
        self._grid_entry(filters, 0, 0, "Line", self.dashboard_filter_line)
        self._grid_entry(filters, 0, 2, "Station", self.dashboard_filter_station)
        self._grid_entry(filters, 0, 4, "Part", self.dashboard_filter_part)
        self._grid_entry(filters, 1, 0, "Template Ver", self.dashboard_filter_template)
        ttk.Label(filters, text="Granularity").grid(row=1, column=2, sticky="w", padx=(0, 8), pady=4)
        self.dashboard_granularity.grid(row=1, column=3, sticky="ew", pady=4)
        ttk.Button(filters, text="Refresh Dashboard", command=self.refresh_dashboard).grid(row=1, column=5, sticky="e", pady=4)

        cards = ttk.Frame(self.dashboard_tab, padding=(6, 0, 6, 6))
        cards.grid(row=1, column=0, sticky="ew")
        for index in range(6):
            cards.columnconfigure(index, weight=1)
        self.dashboard_cards = {
            "total": StatCard(cards, "Total", background="#0f172a", foreground="#f8fafc"),
            "accept": StatCard(cards, "Accept", background="#166534", foreground="#f0fdf4"),
            "reject": StatCard(cards, "Reject", background="#991b1b", foreground="#fef2f2"),
            "part_ready": StatCard(cards, "Part Ready", background="#1d4ed8", foreground="#eff6ff"),
            "avg_conf": StatCard(cards, "Avg Sticker Conf", background="#7c2d12", foreground="#fff7ed"),
            "backend": StatCard(cards, "ML Backend", background="#334155", foreground="#f8fafc"),
        }
        for column, key in enumerate(("total", "accept", "reject", "part_ready", "avg_conf", "backend")):
            self.dashboard_cards[key].grid(row=0, column=column, sticky="ew", padx=4)

        lower = ttk.Panedwindow(self.dashboard_tab, orient="horizontal")
        lower.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 6))

        bucket_panel = ttk.Frame(lower, padding=8)
        raw_panel = ttk.Frame(lower, padding=8)
        lower.add(bucket_panel, weight=3)
        lower.add(raw_panel, weight=2)

        ttk.Label(bucket_panel, text="Bucket Trend", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.dashboard_buckets_list = tk.Listbox(bucket_panel)
        self.dashboard_buckets_list.pack(fill="both", expand=True, pady=(8, 0))

        self.dashboard_raw = JsonEditor(raw_panel, "Dashboard Raw", {})
        self.dashboard_raw.pack(fill="both", expand=True)

    def _grid_entry(self, master, row: int, column: int, label: str, widget) -> None:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=row, column=column + 1, sticky="ew", pady=4)

    def _selected_listbox_id(self, listbox: tk.Listbox) -> int | None:
        if not listbox.curselection():
            return None
        selected = listbox.get(listbox.curselection()[0])
        try:
            return int(str(selected).split("|", 1)[0].strip())
        except ValueError:
            return None

    def refresh_template_dependencies(self) -> None:
        try:
            models = self.api.list_models()
            profiles = self.api.list_profiles()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Admin", f"Failed to load template dependencies: {exc}")
            return
        self.template_form.set_model_options(models)
        self.template_form.set_profile_options(profiles)

    def refresh_templates(self) -> None:
        try:
            items = self.api.list_templates()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", str(exc))
            return
        self.template_list.delete(0, "end")
        self._template_summary_lookup = {}
        dep_values: list[str] = []
        for item in items:
            label = f"{item['id']} | v{item.get('version_number')} | {item['name']}"
            self.template_list.insert("end", label)
            dep_label = f"{item['id']} | {item['name']} | v{item.get('version_number')}"
            self._template_summary_lookup[dep_label] = item
            dep_values.append(dep_label)
        self.dep_template_selector.configure(values=dep_values)
        self.refresh_template_dependencies()

    def new_template(self) -> None:
        self.current_template_id = None
        self.template_form.reset()
        self.preview_template_json()

    def load_selected_template(self) -> None:
        template_id = self._selected_listbox_id(self.template_list)
        if template_id is None:
            return
        try:
            detail = self.api.get_template(template_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", str(exc))
            return
        self.current_template_id = template_id
        self.template_form.set_payload(detail)
        self.template_raw_editor.set_payload(detail)

    def preview_template_json(self) -> None:
        try:
            payload = self.template_form.get_payload()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", f"Template form invalid: {exc}")
            return
        self.template_raw_editor.set_payload(payload)

    def apply_raw_template(self) -> None:
        try:
            payload = self.template_raw_editor.get_payload()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", f"Raw JSON invalid: {exc}")
            return
        self.template_form.set_payload(payload)

    def save_template(self) -> None:
        try:
            payload = self.template_form.get_payload()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", f"Template form invalid: {exc}")
            return
        try:
            if self.current_template_id:
                saved = self.api.update_template(self.current_template_id, payload)
            else:
                saved = self.api.create_template(payload)
                self.current_template_id = int(saved["id"])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", str(exc))
            return
        self.template_form.set_payload(saved)
        self.template_raw_editor.set_payload(saved)
        self.refresh_templates()
        messagebox.showinfo("Templates", "Template saved.")

    def delete_selected_template(self) -> None:
        template_id = self._selected_listbox_id(self.template_list)
        if template_id is None:
            return
        if not messagebox.askyesno("Templates", "Delete selected template?"):
            return
        try:
            self.api.delete_template(template_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", str(exc))
            return
        self.current_template_id = None
        self.template_form.reset()
        self.template_raw_editor.set_payload({})
        self.refresh_templates()

    def _on_deployment_template_selected(self, _event=None) -> None:
        item = self._template_summary_lookup.get(self.dep_template_choice.get().strip())
        if not item:
            return
        self.dep_template_id.delete(0, "end")
        self.dep_template_id.insert(0, str(item.get("id") or ""))
        self.dep_version_id.delete(0, "end")
        self.dep_version_id.insert(0, str(item.get("version_id") or ""))

    def refresh_deployments(self) -> None:
        try:
            items = self.api.list_deployments()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Deployments", str(exc))
            return
        self.deployment_list.delete(0, "end")
        for item in items:
            self.deployment_list.insert(
                "end",
                f"{item['id']} | {item['line_id']}/{item['station_id']} | template={item['template_name']} | version={item['template_version_id']} | active={item['is_active']}",
            )

    def deploy_template(self) -> None:
        payload = {
            "template_id": int(self.dep_template_id.get() or 0),
            "template_version_id": int(self.dep_version_id.get() or 0),
            "line_id": self.dep_line.get().strip(),
            "station_id": self.dep_station.get().strip(),
        }
        try:
            self.api.deploy_template(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Deployments", str(exc))
            return
        self.refresh_deployments()
        messagebox.showinfo("Deployments", "Deployment saved.")

    def deactivate_selected_deployment(self) -> None:
        deployment_id = self._selected_listbox_id(self.deployment_list)
        if deployment_id is None:
            return
        try:
            self.api.deactivate_deployment(deployment_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Deployments", str(exc))
            return
        self.refresh_deployments()

    def refresh_users(self) -> None:
        try:
            items = self.api.list_users()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Users", str(exc))
            return
        self.users_list.delete(0, "end")
        for item in items:
            self.users_list.insert("end", f"{item['id']} | {item['username']} | {item['role']} | active={item['is_active']}")

    def create_user(self) -> None:
        try:
            self.api.create_user(
                {
                    "username": self.user_name.get().strip(),
                    "password": self.user_pass.get().strip(),
                    "role": self.user_role.get().strip(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Users", str(exc))
            return
        self.user_name.delete(0, "end")
        self.user_pass.delete(0, "end")
        self.refresh_users()

    def set_selected_user_active(self, is_active: bool) -> None:
        user_id = self._selected_listbox_id(self.users_list)
        if user_id is None:
            return
        try:
            self.api.set_user_active(user_id, is_active)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Users", str(exc))
            return
        self.refresh_users()

    def _results_filters(self) -> dict[str, object]:
        params: dict[str, object] = {}
        if self.result_filter_line.get().strip():
            params["line_id"] = self.result_filter_line.get().strip()
        if self.result_filter_station.get().strip():
            params["station_id"] = self.result_filter_station.get().strip()
        if self.result_filter_part.get().strip():
            params["part_name"] = self.result_filter_part.get().strip()
        if self.result_filter_template.get().strip():
            params["template_version_id"] = self.result_filter_template.get().strip()
        if self.result_filter_decision.get().strip():
            params["decision_code"] = self.result_filter_decision.get().strip()
        return params

    def refresh_results(self) -> None:
        try:
            items = self.api.list_inspections(self._results_filters())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Results", str(exc))
            return
        self.results_list.delete(0, "end")
        for item in items:
            self.results_list.insert(
                "end",
                f"{item['id']} | {item.get('decision')} | {item.get('part_name')} | {item.get('line_id')}/{item.get('station_id') or '-'} | {item.get('reject_reason_code') or 'OK'}",
            )

    def _export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile="inspections.csv",
            title="Export Inspection Results",
        )
        if not path:
            return
        try:
            csv_text = self.api.export_inspections_csv(self._results_filters())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export CSV", str(exc))
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)
        except OSError as exc:
            messagebox.showerror("Export CSV", f"Could not write file:\n{exc}")
            return
        messagebox.showinfo("Export CSV", f"Saved {len(csv_text.splitlines()) - 1} rows to:\n{path}")

    def open_result(self) -> None:
        result_id = self._selected_listbox_id(self.results_list)
        if result_id is None:
            return
        try:
            detail = self.api.get_inspection(result_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Results", str(exc))
            return
        self.result_summary.set_values(
            {
                "decision": detail.get("decision"),
                "reason": detail.get("reject_reason_code") or "OK",
                "part_name": detail.get("part_name"),
                "line_id": detail.get("line_id"),
                "station_id": detail.get("station_id"),
                "detected_class": detail.get("detected_class"),
                "expected_class": detail.get("expected_class"),
                "sticker_backend": detail.get("sticker_backend"),
                "part_ready_status": detail.get("part_ready_status"),
                "part_ready_match_ratio": detail.get("part_ready_match_ratio"),
                "sticker_confidence": detail.get("sticker_confidence"),
                "push_status": detail.get("push_status"),
            }
        )
        self.result_detail.set_payload(detail)

    def _dashboard_filters(self) -> tuple[dict[str, object], dict[str, object]]:
        base: dict[str, object] = {}
        if self.dashboard_filter_line.get().strip():
            base["line_id"] = self.dashboard_filter_line.get().strip()
        if self.dashboard_filter_station.get().strip():
            base["station_id"] = self.dashboard_filter_station.get().strip()
        if self.dashboard_filter_part.get().strip():
            base["part_name"] = self.dashboard_filter_part.get().strip()
        if self.dashboard_filter_template.get().strip():
            base["template_version_id"] = self.dashboard_filter_template.get().strip()
        buckets = dict(base)
        buckets["granularity"] = self.dashboard_granularity.get().strip() or "hour"
        return base, buckets

    def refresh_dashboard(self) -> None:
        summary_params, bucket_params = self._dashboard_filters()
        try:
            summary = self.api.dashboard_summary(summary_params)
            buckets = self.api.dashboard_buckets(bucket_params)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dashboard", str(exc))
            return

        self.dashboard_cards["total"].set_value(summary.get("total_inspections", 0))
        self.dashboard_cards["accept"].set_value(summary.get("total_accept", 0))
        self.dashboard_cards["reject"].set_value(summary.get("total_reject", 0))
        self.dashboard_cards["part_ready"].set_value(
            summary.get("total_part_ready", 0),
            note=f"not ready: {summary.get('total_part_not_ready', 0)}",
        )
        avg_conf = summary.get("avg_sticker_confidence")
        self.dashboard_cards["avg_conf"].set_value("-" if avg_conf is None else f"{float(avg_conf):.3f}")
        self.dashboard_cards["backend"].set_value(
            summary.get("backend_ultralytics", 0),
            note=f"classic {summary.get('backend_classic', 0)}",
        )

        self.dashboard_buckets_list.delete(0, "end")
        for item in buckets:
            self.dashboard_buckets_list.insert(
                "end",
                f"{item.get('bucket')} | total={item.get('total_inspections')} | accept={item.get('total_accept')} | reject={item.get('total_reject')} | {item.get('line_id') or '-'} / {item.get('station_id') or '-'}",
            )
        self.dashboard_raw.set_payload({"summary": summary, "buckets": buckets})

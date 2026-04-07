from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from client_tk.app.components.async_bridge import run_async
from client_tk.app.components.scrollable_frame import AutoHideScrollbar, ScrollableFrame
from client_tk.app.components.template_forms import JsonEditor, LabeledValuePanel, TemplateEditorForm


MUTED_TEXT = "#475569"


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


class CompactStatCard(ttk.Frame):
    def __init__(self, master, title: str, *, background: str, foreground: str):
        super().__init__(master)
        shell = tk.Frame(self, bg=background, padx=10, pady=6)
        shell.pack(fill="both", expand=True)
        tk.Label(shell, text=title, bg=background, fg=foreground, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.value_label = tk.Label(shell, text="0", bg=background, fg=foreground, font=("Segoe UI", 13, "bold"))
        self.value_label.pack(anchor="w", pady=(2, 0))
        self.note_label = tk.Label(shell, text="", bg=background, fg=foreground, font=("Segoe UI", 7))
        self.note_label.pack(anchor="w", pady=(1, 0))

    def set_value(self, value: object, note: str = "") -> None:
        self.value_label.configure(text=str(value))
        self.note_label.configure(text=note)


class AdminScreen(ttk.Frame):
    def __init__(self, master, api_client, session_state):
        super().__init__(master, padding=0)
        self.api = api_client
        self.state = session_state
        self.current_template_id: int | None = None

        self._template_summary_lookup: dict[str, dict] = {}
        self._templates_cache: list[dict] = []
        self._deployments_cache: list[dict] = []
        self._users_cache: list[dict] = []
        self._results_cache: list[dict] = []
        self._overview_cards_visible = True

        self.status_var = tk.StringVar(value="Admin workspace ready.")
        self.template_search_var = tk.StringVar()
        self.user_role_filter_var = tk.StringVar(value="all")
        self.template_count_var = tk.StringVar(value="0 templates")
        self.deployment_count_var = tk.StringVar(value="0 deployments")
        self.user_count_var = tk.StringVar(value="0 users")
        self.results_count_var = tk.StringVar(value="0 results")
        self.dashboard_count_var = tk.StringVar(value="0 buckets")
        self.result_filter_push_status_var = tk.StringVar(value="")
        self.template_context_var = tk.StringVar(value="Belum ada template yang dimuat.")
        self.deployment_context_var = tk.StringVar(value="Pilih deployment untuk melihat konteks cepat.")
        self.user_context_var = tk.StringVar(value="Pilih user untuk melihat status akun.")
        self.results_context_var = tk.StringVar(value="Gunakan filter untuk memuat hasil inspeksi yang relevan.")
        self.dashboard_context_var = tk.StringVar(value="Dashboard belum direfresh.")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabs()
        self._build_status_bar()

        self.template_search_var.trace_add("write", lambda *_args: self._render_templates())
        self.user_role_filter_var.trace_add("write", lambda *_args: self._render_users())
        self.bind("<Configure>", self._on_resize)
        self.after_idle(self._apply_responsive_layout)

        self.refresh_all()

    def _build_header(self) -> None:
        header = ttk.Frame(self, padding=(12, 8, 12, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        top = ttk.Frame(header)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        user = self.state.user or {}
        identity = f"{_safe_text(user.get('username'))} ({_safe_text(user.get('role'))})"
        ttk.Label(top, text="Admin Workspace", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            top,
            text=f"{identity}  |  {_safe_text(self.state.base_url)}  |  Kelola template, deployment, user, hasil inspeksi, dan dashboard.",
            foreground=MUTED_TEXT,
            wraplength=860,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        actions = ttk.Frame(top)
        actions.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Button(actions, text="Refresh All", command=self.refresh_all).pack(side="left")

        self.overview_cards_frame = ttk.Frame(header, padding=(0, 8, 0, 0))
        self.overview_cards_frame.grid(row=1, column=0, sticky="ew")
        for index in range(4):
            self.overview_cards_frame.columnconfigure(index, weight=1)
        self.admin_cards = {
            "templates": CompactStatCard(self.overview_cards_frame, "Templates", background="#0f172a", foreground="#f8fafc"),
            "deployments": CompactStatCard(self.overview_cards_frame, "Deployments", background="#134e4a", foreground="#ecfdf5"),
            "users": CompactStatCard(self.overview_cards_frame, "Users", background="#7c2d12", foreground="#fff7ed"),
            "results": CompactStatCard(self.overview_cards_frame, "Visible Results", background="#1d4ed8", foreground="#eff6ff"),
        }
        for column, key in enumerate(("templates", "deployments", "users", "results")):
            self.admin_cards[key].grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 4 if column < 3 else 0))

    def _build_tabs(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))

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

    def _build_status_bar(self) -> None:
        status_bar = ttk.Frame(self, padding=(12, 0, 12, 12))
        status_bar.grid(row=2, column=0, sticky="ew")
        ttk.Label(status_bar, textvariable=self.status_var, foreground=MUTED_TEXT).pack(anchor="w")

    def _on_resize(self, _event=None) -> None:
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        height = self.winfo_height()
        should_show_cards = height <= 1 or height >= 760
        if should_show_cards == self._overview_cards_visible:
            return
        self._overview_cards_visible = should_show_cards
        if should_show_cards:
            self.overview_cards_frame.grid()
        else:
            self.overview_cards_frame.grid_remove()

    def _build_templates_tab(self) -> None:
        container = ttk.Panedwindow(self.templates_tab, orient="horizontal")
        container.pack(fill="both", expand=True, padx=2, pady=2)

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=2)
        container.add(right, weight=5)

        library = ttk.LabelFrame(left, text="Template Library", padding=12)
        library.pack(fill="both", expand=True)
        library.columnconfigure(0, weight=1)
        library.rowconfigure(3, weight=1)

        ttk.Label(library, text="Template aktif dan histori versi", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            library,
            text="Cari template berdasarkan ID, nama, atau deskripsi. Double-click untuk memuat ke editor.",
            foreground=MUTED_TEXT,
            wraplength=320,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 8))

        search_row = ttk.Frame(library)
        search_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        search_row.columnconfigure(2, weight=1)
        ttk.Label(search_row, textvariable=self.template_count_var, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(search_row, text="Search").grid(row=0, column=1, sticky="w", padx=(12, 8))
        ttk.Entry(search_row, textvariable=self.template_search_var).grid(row=0, column=2, sticky="ew")
        ttk.Button(search_row, text="Clear", command=lambda: self.template_search_var.set("")).grid(row=0, column=3, padx=(8, 0))

        self.template_table = self._build_table(
            library,
            [
                ("id", "ID", 60, "center"),
                ("version", "Version", 80, "center"),
                ("name", "Name", 160, "w"),
                ("status", "Status", 90, "center"),
                ("updated", "Updated", 150, "w"),
            ],
            row=3,
            height=16,
        )
        self.template_table.bind("<<TreeviewSelect>>", self._on_template_selected)
        self.template_table.bind("<Double-1>", lambda _event: self.load_selected_template())

        actions = ttk.Frame(library)
        actions.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="Refresh", command=self.refresh_templates).pack(side="left")
        ttk.Button(actions, text="New Draft", command=self.new_template).pack(side="left", padx=6)
        ttk.Button(actions, text="Load Selected", command=self.load_selected_template).pack(side="left")
        ttk.Button(actions, text="Delete Selected", command=self.delete_selected_template).pack(side="right")

        editor = ttk.LabelFrame(right, text="Template Editor", padding=12)
        editor.pack(fill="both", expand=True)
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(2, weight=1)

        ttk.Label(
            editor,
            text="Edit template dengan form terstruktur. Raw JSON tetap tersedia untuk advanced editing dan debugging kontrak.",
            foreground=MUTED_TEXT,
            wraplength=900,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(editor, textvariable=self.template_context_var, font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", pady=(8, 8))

        editor_tabs = ttk.Notebook(editor)
        editor_tabs.grid(row=2, column=0, sticky="nsew")

        structured_tab = ttk.Frame(editor_tabs, padding=4)
        raw_tab = ttk.Frame(editor_tabs, padding=4)
        editor_tabs.add(structured_tab, text="Structured Form")
        editor_tabs.add(raw_tab, text="Raw JSON")

        structured_tab.columnconfigure(0, weight=1)
        structured_tab.rowconfigure(0, weight=1)
        structured_scroller = ScrollableFrame(structured_tab)
        structured_scroller.grid(row=0, column=0, sticky="nsew")
        structured_scroller.body.columnconfigure(0, weight=1)

        self.template_form = TemplateEditorForm(structured_scroller.body)
        self.template_form._api_client_ref = self.api
        self.template_form.pack(fill="both", expand=True)

        self.template_raw_editor = JsonEditor(raw_tab, "Template Raw JSON", {})
        self.template_raw_editor.pack(fill="both", expand=True)

        footer = ttk.Frame(editor)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(footer, text="Preview Raw JSON", command=self.preview_template_json).pack(side="left")
        ttk.Button(footer, text="Apply Raw to Form", command=self.apply_raw_template).pack(side="left", padx=6)
        ttk.Button(footer, text="Reset Draft", command=self.new_template).pack(side="left")
        ttk.Button(footer, text="Save Template", command=self.save_template).pack(side="right")

    def _build_deployments_tab(self) -> None:
        container = ttk.Panedwindow(self.deployments_tab, orient="horizontal")
        container.pack(fill="both", expand=True, padx=2, pady=2)

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=3)
        container.add(right, weight=2)

        listing = ttk.LabelFrame(left, text="Active Deployments", padding=12)
        listing.pack(fill="both", expand=True)
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(2, weight=1)

        ttk.Label(listing, text="Deployment line/station aktif", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(listing, textvariable=self.deployment_context_var, foreground=MUTED_TEXT, wraplength=560, justify="left").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(4, 8),
        )

        self.deployment_table = self._build_table(
            listing,
            [
                ("id", "ID", 60, "center"),
                ("line", "Line", 120, "w"),
                ("station", "Station", 120, "w"),
                ("template", "Template", 190, "w"),
                ("version", "Version", 90, "center"),
                ("status", "Status", 90, "center"),
            ],
            row=2,
            height=16,
        )
        self.deployment_table.bind("<<TreeviewSelect>>", self._on_deployment_selected)

        footer = ttk.Frame(listing)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(footer, textvariable=self.deployment_count_var, font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Button(footer, text="Refresh", command=self.refresh_deployments).pack(side="right")
        ttk.Button(footer, text="Deactivate Selected", command=self.deactivate_selected_deployment).pack(side="right", padx=(0, 6))

        form_card = ttk.LabelFrame(right, text="Deploy Template to Line / Station", padding=12)
        form_card.pack(fill="both", expand=True)
        form_card.columnconfigure(1, weight=1)
        form_card.columnconfigure(3, weight=1)

        ttk.Label(
            form_card,
            text=(
                "Pilih template versi aktif lalu assign ke pasangan line/station. Operator akan menarik deployment "
                "ini saat melakukan load deployment."
            ),
            foreground=MUTED_TEXT,
            wraplength=420,
            justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        self.dep_template_choice = tk.StringVar()
        self.dep_template_selector = ttk.Combobox(form_card, textvariable=self.dep_template_choice, state="readonly")
        ttk.Label(form_card, text="Template").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.dep_template_selector.grid(row=1, column=1, columnspan=3, sticky="ew", pady=4)
        self.dep_template_selector.bind("<<ComboboxSelected>>", self._on_deployment_template_selected)

        self.dep_template_id = ttk.Entry(form_card)
        self.dep_version_id = ttk.Entry(form_card)
        self.dep_line = ttk.Entry(form_card)
        self.dep_station = ttk.Entry(form_card)
        self._grid_entry(form_card, 2, 0, "Template ID", self.dep_template_id)
        self._grid_entry(form_card, 2, 2, "Version ID", self.dep_version_id)
        self._grid_entry(form_card, 3, 0, "Line", self.dep_line)
        self._grid_entry(form_card, 3, 2, "Station", self.dep_station)

        ttk.Label(
            form_card,
            text="Tip: pilih template dari dropdown agar Template ID dan Version ID terisi otomatis.",
            foreground=MUTED_TEXT,
            wraplength=420,
            justify="left",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

        action_bar = ttk.Frame(form_card)
        action_bar.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        ttk.Button(action_bar, text="Deploy", command=self.deploy_template).pack(side="right")

    def _build_users_tab(self) -> None:
        container = ttk.Panedwindow(self.users_tab, orient="horizontal")
        container.pack(fill="both", expand=True, padx=2, pady=2)

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=3)
        container.add(right, weight=2)

        listing = ttk.LabelFrame(left, text="User Access", padding=12)
        listing.pack(fill="both", expand=True)
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(3, weight=1)

        ttk.Label(listing, text="Akun pengguna", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(listing, textvariable=self.user_context_var, foreground=MUTED_TEXT, wraplength=560, justify="left").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(4, 8),
        )

        filters = ttk.Frame(listing)
        filters.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        filters.columnconfigure(2, weight=1)
        ttk.Label(filters, textvariable=self.user_count_var, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(filters, text="Role Filter").grid(row=0, column=1, sticky="w", padx=(12, 8))
        ttk.Combobox(
            filters,
            textvariable=self.user_role_filter_var,
            values=["all", "admin", "operator", "engineer", "inactive"],
            state="readonly",
        ).grid(row=0, column=2, sticky="w")

        self.users_table = self._build_table(
            listing,
            [
                ("id", "ID", 60, "center"),
                ("username", "Username", 170, "w"),
                ("role", "Role", 110, "center"),
                ("status", "Status", 100, "center"),
                ("created", "Created", 150, "w"),
            ],
            row=3,
            height=16,
        )
        self.users_table.bind("<<TreeviewSelect>>", self._on_user_selected)

        action_bar = ttk.Frame(listing)
        action_bar.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(action_bar, text="Refresh", command=self.refresh_users).pack(side="left")
        ttk.Button(action_bar, text="Enable Selected", command=lambda: self.set_selected_user_active(True)).pack(side="right")
        ttk.Button(action_bar, text="Disable Selected", command=lambda: self.set_selected_user_active(False)).pack(side="right", padx=(0, 6))

        form = ttk.LabelFrame(right, text="Create User", padding=12)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        ttk.Label(
            form,
            text="Tambahkan akun baru untuk admin, operator, atau engineer. Password wajib diisi saat create.",
            foreground=MUTED_TEXT,
            wraplength=380,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.user_name = ttk.Entry(form)
        self.user_pass = ttk.Entry(form, show="*")
        self.user_role = ttk.Combobox(form, values=["admin", "operator", "engineer"], state="readonly")
        self.user_role.set("operator")
        self._grid_entry(form, 1, 0, "Username", self.user_name)
        self._grid_entry(form, 2, 0, "Password", self.user_pass)
        ttk.Label(form, text="Role").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        self.user_role.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Button(form, text="Create User", command=self.create_user).grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))

    def _build_results_tab(self) -> None:
        self.results_tab.columnconfigure(0, weight=1)
        self.results_tab.rowconfigure(1, weight=1)

        filters = ttk.LabelFrame(self.results_tab, text="Inspection Filters", padding=10)
        filters.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        for index in range(10):
            filters.columnconfigure(index, weight=1 if index % 2 else 0)

        ttk.Label(
            filters,
            text="Filter hasil inspeksi lalu double-click row untuk membuka detail lengkap.",
            foreground=MUTED_TEXT,
        ).grid(row=0, column=0, columnspan=10, sticky="w", pady=(0, 8))

        self.result_filter_line = ttk.Entry(filters)
        self.result_filter_station = ttk.Entry(filters)
        self.result_filter_part = ttk.Entry(filters)
        self.result_filter_template = ttk.Entry(filters)
        self.result_filter_decision = ttk.Combobox(filters, values=["", "ACCEPT", "REJECT"], state="readonly")
        self.result_filter_push_status = ttk.Combobox(
            filters,
            textvariable=self.result_filter_push_status_var,
            values=["", "sent", "failed", "pending", "local_only"],
            state="readonly",
        )
        self.result_filter_decision.set("")
        self.result_filter_push_status_var.set("")
        self._grid_entry(filters, 1, 0, "Line", self.result_filter_line)
        self._grid_entry(filters, 1, 2, "Station", self.result_filter_station)
        self._grid_entry(filters, 1, 4, "Part", self.result_filter_part)
        self._grid_entry(filters, 2, 0, "Template Ver", self.result_filter_template)
        ttk.Label(filters, text="Decision").grid(row=2, column=2, sticky="w", padx=(0, 8), pady=4)
        self.result_filter_decision.grid(row=2, column=3, sticky="ew", pady=4)
        ttk.Label(filters, text="Push Status").grid(row=2, column=4, sticky="w", padx=(0, 8), pady=4)
        self.result_filter_push_status.grid(row=2, column=5, sticky="ew", pady=4)
        ttk.Button(filters, text="Reset", command=self._reset_results_filters).grid(row=2, column=7, sticky="e", pady=4)
        ttk.Button(filters, text="Refresh", command=self.refresh_results).grid(row=2, column=8, sticky="e", pady=4, padx=(6, 0))
        ttk.Button(filters, text="Export CSV", command=self._export_csv).grid(row=2, column=9, sticky="e", pady=4, padx=(6, 0))

        container = ttk.Panedwindow(self.results_tab, orient="horizontal")
        container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = ttk.Frame(container, padding=8)
        right = ttk.Frame(container, padding=8)
        container.add(left, weight=4)
        container.add(right, weight=3)

        listing = ttk.LabelFrame(left, text="Inspection Results", padding=12)
        listing.pack(fill="both", expand=True)
        ttk.Label(listing, textvariable=self.results_context_var, foreground=MUTED_TEXT, wraplength=560, justify="left").pack(
            anchor="w"
        )
        ttk.Label(listing, textvariable=self.results_count_var, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 8))

        self.results_table = self._build_table(
            listing,
            [
                ("id", "ID", 60, "center"),
                ("time", "Inspected", 150, "w"),
                ("decision", "Decision", 95, "center"),
                ("part", "Part", 150, "w"),
                ("line", "Line", 90, "w"),
                ("station", "Station", 90, "w"),
                ("push", "Push", 90, "center"),
                ("retry", "Retries", 70, "center"),
                ("reason", "Reason", 140, "w"),
            ],
            row=None,
            height=16,
        )
        self.results_table.bind("<<TreeviewSelect>>", lambda _event: self.open_result())
        self.results_table.bind("<Double-1>", lambda _event: self.open_result())

        result_actions = ttk.Frame(listing)
        result_actions.pack(fill="x", pady=(10, 0))
        ttk.Button(result_actions, text="Retry Failed Visible", command=self.retry_visible_failed_pushes).pack(side="left")
        ttk.Button(result_actions, text="Open Selected", command=self.open_result).pack(side="right")
        ttk.Button(result_actions, text="Retry Selected Push", command=self.retry_selected_push).pack(side="right", padx=(0, 6))

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        detail_tabs = ttk.Notebook(right)
        detail_tabs.grid(row=0, column=0, sticky="nsew")

        summary_tab = ttk.Frame(detail_tabs, padding=6)
        raw_tab = ttk.Frame(detail_tabs, padding=6)
        detail_tabs.add(summary_tab, text="Summary")
        detail_tabs.add(raw_tab, text="Raw JSON")

        summary_tab.columnconfigure(0, weight=1)
        summary_tab.rowconfigure(0, weight=1)
        summary_scroller = ScrollableFrame(summary_tab)
        summary_scroller.grid(row=0, column=0, sticky="nsew")
        summary_scroller.body.columnconfigure(0, weight=1)

        self.result_summary = LabeledValuePanel(
            summary_scroller.body,
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
                ("retry_count", "Retry Count"),
                ("sql_mirror_id", "SQL Mirror ID"),
                ("last_push_error", "Last Push Error"),
            ],
            columns=2,
        )
        self.result_summary.pack(fill="both", expand=True, padx=2, pady=2)
        self.result_detail = JsonEditor(raw_tab, "Raw Result Payload", {})
        self.result_detail.pack(fill="both", expand=True)

    def _build_dashboard_tab(self) -> None:
        self.dashboard_tab.columnconfigure(0, weight=1)
        self.dashboard_tab.rowconfigure(2, weight=1)

        filters = ttk.LabelFrame(self.dashboard_tab, text="Dashboard Filters", padding=10)
        filters.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        for index in range(12):
            filters.columnconfigure(index, weight=1 if index % 2 else 0)

        ttk.Label(
            filters,
            text="Gunakan filter yang sama dengan hasil inspeksi untuk melihat agregasi dan tren bucket.",
            foreground=MUTED_TEXT,
        ).grid(row=0, column=0, columnspan=12, sticky="w", pady=(0, 8))

        self.dashboard_filter_line = ttk.Entry(filters)
        self.dashboard_filter_station = ttk.Entry(filters)
        self.dashboard_filter_part = ttk.Entry(filters)
        self.dashboard_filter_template = ttk.Entry(filters)
        self.dashboard_granularity = ttk.Combobox(filters, values=["minute", "hour", "day"], state="readonly")
        self.dashboard_granularity.set("hour")
        self._grid_entry(filters, 1, 0, "Line", self.dashboard_filter_line)
        self._grid_entry(filters, 1, 2, "Station", self.dashboard_filter_station)
        self._grid_entry(filters, 1, 4, "Part", self.dashboard_filter_part)
        self._grid_entry(filters, 2, 0, "Template Ver", self.dashboard_filter_template)
        ttk.Label(filters, text="Granularity").grid(row=2, column=2, sticky="w", padx=(0, 8), pady=4)
        self.dashboard_granularity.grid(row=2, column=3, sticky="ew", pady=4)
        ttk.Button(filters, text="Reset", command=self._reset_dashboard_filters).grid(row=2, column=9, sticky="e", pady=4)
        ttk.Button(filters, text="Refresh Dashboard", command=self.refresh_dashboard).grid(row=2, column=10, columnspan=2, sticky="e", pady=4, padx=(6, 0))

        cards = ttk.Frame(self.dashboard_tab, padding=(8, 0, 8, 6))
        cards.grid(row=1, column=0, sticky="ew")
        for index in range(6):
            cards.columnconfigure(index, weight=1)
        # Use CompactStatCard (same as header) to avoid wasting vertical space
        self.dashboard_cards = {
            "total": CompactStatCard(cards, "Total", background="#0f172a", foreground="#f8fafc"),
            "accept": CompactStatCard(cards, "Accept", background="#166534", foreground="#f0fdf4"),
            "reject": CompactStatCard(cards, "Reject", background="#991b1b", foreground="#fef2f2"),
            "part_ready": CompactStatCard(cards, "Part Ready", background="#1d4ed8", foreground="#eff6ff"),
            "avg_conf": CompactStatCard(cards, "Avg Sticker Conf", background="#7c2d12", foreground="#fff7ed"),
            "backend": CompactStatCard(cards, "ML Backend", background="#334155", foreground="#f8fafc"),
        }
        for column, key in enumerate(("total", "accept", "reject", "part_ready", "avg_conf", "backend")):
            self.dashboard_cards[key].grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 4 if column < 5 else 0))

        body_tabs = ttk.Notebook(self.dashboard_tab)
        body_tabs.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

        trend_tab = ttk.Frame(body_tabs, padding=8)
        raw_tab = ttk.Frame(body_tabs, padding=8)
        body_tabs.add(trend_tab, text="Trend")
        body_tabs.add(raw_tab, text="Raw JSON")

        bucket_card = ttk.LabelFrame(trend_tab, text="Bucket Trend", padding=12)
        bucket_card.pack(fill="both", expand=True)
        ttk.Label(bucket_card, textvariable=self.dashboard_context_var, foreground=MUTED_TEXT, wraplength=760, justify="left").pack(
            anchor="w"
        )
        ttk.Label(bucket_card, textvariable=self.dashboard_count_var, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 8))

        self.dashboard_bucket_table = self._build_table(
            bucket_card,
            [
                ("bucket", "Bucket", 165, "w"),
                ("total", "Total", 70, "center"),
                ("accept", "Accept", 70, "center"),
                ("reject", "Reject", 70, "center"),
                ("line", "Line", 90, "w"),
                ("station", "Station", 90, "w"),
            ],
            row=None,
            height=16,
        )

        self.dashboard_raw = JsonEditor(raw_tab, "Dashboard Raw", {})
        self.dashboard_raw.pack(fill="both", expand=True)

    def _build_table(
        self,
        master,
        columns: list[tuple[str, str, int, str]],
        *,
        row: int | None = None,
        height: int = 14,
    ) -> ttk.Treeview:
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
        x_scroll = AutoHideScrollbar(shell, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        return tree

    def _grid_entry(self, master, row: int, column: int, label: str, widget) -> None:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=row, column=column + 1, sticky="ew", pady=4)

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        tree.delete(*tree.get_children())

    def _selected_treeview_id(self, tree: ttk.Treeview) -> int | None:
        selection = tree.selection()
        if not selection:
            return None
        try:
            return int(selection[0])
        except ValueError:
            return None

    def _select_tree_item(self, tree: ttk.Treeview, item_id: int | None) -> None:
        if item_id is None:
            return
        iid = str(item_id)
        if tree.exists(iid):
            tree.selection_set(iid)
            tree.focus(iid)
            tree.see(iid)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _update_overview_cards(self) -> None:
        template_active = sum(1 for item in self._templates_cache if bool(item.get("is_active", True)))
        deployment_active = sum(1 for item in self._deployments_cache if bool(item.get("is_active", True)))
        user_active = sum(1 for item in self._users_cache if bool(item.get("is_active", True)))
        result_accept = sum(1 for item in self._results_cache if str(item.get("decision") or "") == "ACCEPT")
        result_reject = sum(1 for item in self._results_cache if str(item.get("decision") or "") == "REJECT")

        self.admin_cards["templates"].set_value(len(self._templates_cache), note=f"{template_active} active")
        self.admin_cards["deployments"].set_value(len(self._deployments_cache), note=f"{deployment_active} active")
        self.admin_cards["users"].set_value(len(self._users_cache), note=f"{user_active} active")
        self.admin_cards["results"].set_value(len(self._results_cache), note=f"{result_accept} accept / {result_reject} reject")

    def refresh_all(self) -> None:
        self._set_status("Refreshing all data…")
        self.refresh_templates()
        self.refresh_deployments()
        self.refresh_users()
        self.refresh_results()
        self.refresh_dashboard()

    def refresh_template_dependencies(self) -> None:
        def _load():
            return self.api.list_models(), self.api.list_profiles()

        def _done(result, error):
            if error:
                self._set_status(f"Dependency load error: {error}")
                return
            models, profiles = result
            self.template_form.set_model_options(models)
            self.template_form.set_profile_options(profiles)

        run_async(self, _load, callback=_done)

    def _render_templates(self) -> None:
        if not hasattr(self, "template_table"):
            return
        query = self.template_search_var.get().strip().lower()
        items = []
        for item in self._templates_cache:
            haystack = " ".join(
                (
                    str(item.get("id") or ""),
                    str(item.get("name") or ""),
                    str(item.get("description") or ""),
                    str(item.get("version_number") or ""),
                )
            ).lower()
            if query and query not in haystack:
                continue
            items.append(item)

        self._clear_tree(self.template_table)
        for item in items:
            self.template_table.insert(
                "",
                "end",
                iid=str(item["id"]),
                values=(
                    item.get("id"),
                    f"v{item.get('version_number') or '-'}",
                    _safe_text(item.get("name")),
                    _format_status(item.get("is_active", True)),
                    _format_timestamp(item.get("updated_at")),
                ),
            )

        self.template_count_var.set(f"{len(items)} templates shown")
        self._select_tree_item(self.template_table, self.current_template_id)

    def _on_template_selected(self, _event=None) -> None:
        template_id = self._selected_treeview_id(self.template_table)
        if template_id is None:
            self.template_context_var.set("Belum ada template yang dipilih.")
            return
        item = next((entry for entry in self._templates_cache if int(entry["id"]) == template_id), None)
        if item:
            self.template_context_var.set(
                f"Selected template #{item['id']} | v{item.get('version_number') or '-'} | {_safe_text(item.get('name'))}"
            )

    def refresh_templates(self) -> None:
        self._set_status("Loading templates…")

        def _done(items, error):
            if error:
                self._set_status(f"Templates error: {error}")
                return
            self._templates_cache = list(items)
            self._template_summary_lookup = {}
            dep_values: list[str] = []
            for item in self._templates_cache:
                dep_label = f"{item['id']} | {item['name']} | v{item.get('version_number')}"
                self._template_summary_lookup[dep_label] = item
                dep_values.append(dep_label)
            self.dep_template_selector.configure(values=dep_values)
            self.refresh_template_dependencies()
            self._render_templates()
            self._update_overview_cards()
            self._set_status(f"Loaded {len(self._templates_cache)} templates.")

        run_async(self, self.api.list_templates, callback=_done)

    def new_template(self) -> None:
        self.current_template_id = None
        self.template_form.reset()
        self.preview_template_json()
        self.template_context_var.set("Draft template baru. Belum tersimpan.")
        self.template_table.selection_remove(self.template_table.selection())
        self._set_status("Template editor direset ke draft baru.")

    def load_selected_template(self) -> None:
        template_id = self._selected_treeview_id(self.template_table)
        if template_id is None:
            return
        self._set_status(f"Loading template #{template_id}…")

        def _done(detail, error):
            if error:
                self._set_status(f"Template load error: {error}")
                messagebox.showerror("Templates", str(error))
                return
            self.current_template_id = template_id
            self.template_form.set_payload(detail)
            self.template_raw_editor.set_payload(detail)
            self.template_context_var.set(
                f"Editing template #{detail.get('id')} | v{detail.get('version_number')} | {_safe_text(detail.get('name'))}"
            )
            self._set_status(f"Template #{template_id} dimuat ke editor.")

        run_async(self, self.api.get_template, callback=_done, args=(template_id,))
        return  # early return so the old code below is bypassed

    def preview_template_json(self) -> None:
        try:
            payload = self.template_form.get_payload()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", f"Template form invalid: {exc}")
            return
        self.template_raw_editor.set_payload(payload)
        self._set_status("Preview raw JSON diperbarui dari structured form.")

    def apply_raw_template(self) -> None:
        try:
            payload = self.template_raw_editor.get_payload()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", f"Raw JSON invalid: {exc}")
            return
        self.template_form.set_payload(payload)
        self._set_status("Structured form diperbarui dari raw JSON.")

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
        self._select_tree_item(self.template_table, self.current_template_id)
        self.template_context_var.set(
            f"Saved template #{saved.get('id')} | v{saved.get('version_number')} | {_safe_text(saved.get('name'))}"
        )
        self._set_status(f"Template #{saved.get('id')} berhasil disimpan.")
        messagebox.showinfo("Templates", "Template saved.")

    def delete_selected_template(self) -> None:
        template_id = self._selected_treeview_id(self.template_table)
        if template_id is None:
            return
        if not messagebox.askyesno("Templates", f"Delete template #{template_id}?"):
            return
        try:
            self.api.delete_template(template_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Templates", str(exc))
            return
        self.current_template_id = None
        self.template_form.reset()
        self.template_raw_editor.set_payload({})
        self.template_context_var.set("Template dihapus. Editor kembali ke draft kosong.")
        self.refresh_templates()
        self._set_status(f"Template #{template_id} dihapus.")

    def _on_deployment_template_selected(self, _event=None) -> None:
        item = self._template_summary_lookup.get(self.dep_template_choice.get().strip())
        if not item:
            return
        self.dep_template_id.delete(0, "end")
        self.dep_template_id.insert(0, str(item.get("id") or ""))
        self.dep_version_id.delete(0, "end")
        self.dep_version_id.insert(0, str(item.get("version_id") or ""))

    def _render_deployments(self) -> None:
        self._clear_tree(self.deployment_table)
        for item in self._deployments_cache:
            self.deployment_table.insert(
                "",
                "end",
                iid=str(item["id"]),
                values=(
                    item.get("id"),
                    _safe_text(item.get("line_id")),
                    _safe_text(item.get("station_id")),
                    _safe_text(item.get("template_name")),
                    _safe_text(item.get("template_version_id")),
                    _format_status(item.get("is_active", True)),
                ),
            )
        self.deployment_count_var.set(f"{len(self._deployments_cache)} deployments loaded")

    def _on_deployment_selected(self, _event=None) -> None:
        deployment_id = self._selected_treeview_id(self.deployment_table)
        if deployment_id is None:
            self.deployment_context_var.set("Pilih deployment untuk melihat konteks cepat.")
            return
        item = next((entry for entry in self._deployments_cache if int(entry["id"]) == deployment_id), None)
        if item:
            self.deployment_context_var.set(
                f"Selected deployment #{item['id']} | {_safe_text(item.get('line_id'))}/{_safe_text(item.get('station_id'))} "
                f"| template={_safe_text(item.get('template_name'))}"
            )

    def refresh_deployments(self) -> None:
        self._set_status("Loading deployments…")

        def _done(items, error):
            if error:
                self._set_status(f"Deployments error: {error}")
                return
            self._deployments_cache = list(items)
            self._render_deployments()
            self._update_overview_cards()
            self._set_status(f"Loaded {len(self._deployments_cache)} deployments.")

        run_async(self, self.api.list_deployments, callback=_done)

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
        self._set_status(
            f"Deployment dibuat untuk {payload['line_id']}/{payload['station_id']} memakai template version {payload['template_version_id']}."
        )
        messagebox.showinfo("Deployments", "Deployment saved.")

    def deactivate_selected_deployment(self) -> None:
        deployment_id = self._selected_treeview_id(self.deployment_table)
        if deployment_id is None:
            return
        if not messagebox.askyesno("Deployments", f"Deactivate deployment #{deployment_id}?"):
            return
        try:
            self.api.deactivate_deployment(deployment_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Deployments", str(exc))
            return
        self.refresh_deployments()
        self._set_status(f"Deployment #{deployment_id} dinonaktifkan.")

    def _render_users(self) -> None:
        if not hasattr(self, "users_table"):
            return
        selected_filter = self.user_role_filter_var.get().strip().lower()
        items = []
        for item in self._users_cache:
            role = str(item.get("role") or "").strip().lower()
            active = bool(item.get("is_active", True))
            if selected_filter == "inactive" and active:
                continue
            if selected_filter not in {"", "all", "inactive"} and role != selected_filter:
                continue
            items.append(item)

        self._clear_tree(self.users_table)
        for item in items:
            self.users_table.insert(
                "",
                "end",
                iid=str(item["id"]),
                values=(
                    item.get("id"),
                    _safe_text(item.get("username")),
                    _safe_text(item.get("role")),
                    _format_status(item.get("is_active", True)),
                    _format_timestamp(item.get("created_at")),
                ),
            )
        self.user_count_var.set(f"{len(items)} users shown")

    def _on_user_selected(self, _event=None) -> None:
        user_id = self._selected_treeview_id(self.users_table)
        if user_id is None:
            self.user_context_var.set("Pilih user untuk melihat status akun.")
            return
        item = next((entry for entry in self._users_cache if int(entry["id"]) == user_id), None)
        if item:
            self.user_context_var.set(
                f"Selected user #{item['id']} | {_safe_text(item.get('username'))} | {_safe_text(item.get('role'))} | "
                f"{_format_status(item.get('is_active', True))}"
            )

    def refresh_users(self) -> None:
        self._set_status("Loading users…")

        def _done(items, error):
            if error:
                self._set_status(f"Users error: {error}")
                return
            self._users_cache = list(items)
            self._render_users()
            self._update_overview_cards()
            self._set_status(f"Loaded {len(self._users_cache)} users.")

        run_async(self, self.api.list_users, callback=_done)

    def create_user(self) -> None:
        username = self.user_name.get().strip()
        password = self.user_pass.get().strip()
        role = self.user_role.get().strip()
        if not username or not password:
            messagebox.showerror("Users", "Username dan password wajib diisi.")
            return
        try:
            self.api.create_user({"username": username, "password": password, "role": role})
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Users", str(exc))
            return
        self.user_name.delete(0, "end")
        self.user_pass.delete(0, "end")
        self.refresh_users()
        self._set_status(f"User `{username}` berhasil dibuat.")

    def set_selected_user_active(self, is_active: bool) -> None:
        user_id = self._selected_treeview_id(self.users_table)
        if user_id is None:
            return
        action = "enable" if is_active else "disable"
        if not messagebox.askyesno("Users", f"{action.title()} user #{user_id}?"):
            return
        try:
            self.api.set_user_active(user_id, is_active)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Users", str(exc))
            return
        self.refresh_users()
        self._set_status(f"User #{user_id} diubah menjadi {_format_status(is_active)}.")

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
        if self.result_filter_push_status_var.get().strip():
            params["push_status"] = self.result_filter_push_status_var.get().strip()
        return params

    def _reset_results_filters(self) -> None:
        for widget in (
            self.result_filter_line,
            self.result_filter_station,
            self.result_filter_part,
            self.result_filter_template,
        ):
            widget.delete(0, "end")
        self.result_filter_decision.set("")
        self.result_filter_push_status_var.set("")
        self.refresh_results()

    def _render_results(self) -> None:
        self._clear_tree(self.results_table)
        for item in self._results_cache:
            self.results_table.insert(
                "",
                "end",
                iid=str(item["id"]),
                values=(
                    item.get("id"),
                    _format_timestamp(item.get("inspected_at")),
                    _safe_text(item.get("decision")),
                    _safe_text(item.get("part_name")),
                    _safe_text(item.get("line_id")),
                    _safe_text(item.get("station_id")),
                    _safe_text(item.get("push_status")),
                    _safe_text(item.get("retry_count"), fallback="0"),
                    _safe_text(item.get("reject_reason_code") or "OK"),
                ),
            )
        self.results_count_var.set(f"{len(self._results_cache)} results shown")

    def refresh_results(self) -> None:
        self._set_status("Loading inspection results…")
        params = self._results_filters()

        def _done(items, error):
            if error:
                self._set_status(f"Results error: {error}")
                return
            self._results_cache = list(items)
            self._render_results()
            self._update_overview_cards()
            if not self._results_cache:
                self.result_summary.reset()
                self.result_detail.set_payload({})
                self.results_context_var.set("Tidak ada hasil inspeksi untuk filter saat ini.")
            else:
                self.results_context_var.set("Pilih hasil inspeksi untuk melihat summary dan payload lengkap.")
            self._set_status(f"Loaded {len(self._results_cache)} inspection results.")

        run_async(self, self.api.list_inspections, callback=_done, args=(params,))

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
            with open(path, "w", encoding="utf-8", newline="") as file_handle:
                file_handle.write(csv_text)
        except OSError as exc:
            messagebox.showerror("Export CSV", f"Could not write file:\n{exc}")
            return
        self._set_status(f"CSV inspection export tersimpan ke {path}.")
        messagebox.showinfo("Export CSV", f"Saved {len(csv_text.splitlines()) - 1} rows to:\n{path}")

    def open_result(self) -> None:
        result_id = self._selected_treeview_id(self.results_table)
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
                "retry_count": detail.get("retry_count"),
                "sql_mirror_id": detail.get("sql_mirror_id"),
                "last_push_error": detail.get("last_push_error"),
            }
        )
        self.result_detail.set_payload(detail)
        self.results_context_var.set(
            f"Selected result #{detail.get('id')} | push={_safe_text(detail.get('push_status'))} | "
            f"retry={_safe_text(detail.get('retry_count'), fallback='0')}"
        )
        self._set_status(f"Inspection result #{result_id} dibuka.")

    def _retryable_visible_result_ids(self) -> list[int]:
        retryable_ids: list[int] = []
        for item in self._results_cache:
            push_status = str(item.get("push_status") or "").strip().lower()
            if push_status in {"failed", "pending"}:
                retryable_ids.append(int(item["id"]))
        return retryable_ids

    def retry_selected_push(self) -> None:
        result_id = self._selected_treeview_id(self.results_table)
        if result_id is None:
            return
        try:
            response = self.api.retry_inspection_push(result_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Results", str(exc))
            return
        result = dict(response.get("result") or {})
        self.refresh_results()
        self._select_tree_item(self.results_table, result_id)
        self.open_result()
        push_status = _safe_text(result.get("push_status"))
        if push_status.lower() == "sent":
            messagebox.showinfo("Results", f"Push result #{result_id} berhasil dikirim ulang ke SQL Server.")
        else:
            messagebox.showwarning(
                "Results",
                f"Push result #{result_id} masih gagal.\n\n{_safe_text(result.get('last_push_error'))}",
            )

    def retry_visible_failed_pushes(self) -> None:
        result_ids = self._retryable_visible_result_ids()
        if not result_ids:
            messagebox.showinfo("Results", "Tidak ada result dengan push status failed/pending pada list saat ini.")
            return
        try:
            response = self.api.retry_failed_inspection_pushes(result_ids=result_ids, limit=len(result_ids))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Results", str(exc))
            return
        self.refresh_results()
        attempted = int(response.get("attempted") or 0)
        succeeded = int(response.get("succeeded") or 0)
        failed = int(response.get("failed") or 0)
        self._set_status(f"Retry push selesai: attempted={attempted}, succeeded={succeeded}, failed={failed}.")
        if failed:
            messagebox.showwarning(
                "Results",
                f"Retry push selesai.\n\nAttempted: {attempted}\nSucceeded: {succeeded}\nFailed: {failed}",
            )
        else:
            messagebox.showinfo("Results", f"Semua {succeeded} push berhasil dikirim ulang ke SQL Server.")

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

    def _reset_dashboard_filters(self) -> None:
        for widget in (
            self.dashboard_filter_line,
            self.dashboard_filter_station,
            self.dashboard_filter_part,
            self.dashboard_filter_template,
        ):
            widget.delete(0, "end")
        self.dashboard_granularity.set("hour")
        self.refresh_dashboard()

    def _render_dashboard_buckets(self, buckets: list[dict]) -> None:
        self._clear_tree(self.dashboard_bucket_table)
        for index, item in enumerate(buckets):
            self.dashboard_bucket_table.insert(
                "",
                "end",
                iid=f"bucket-{index}",
                values=(
                    _safe_text(item.get("bucket") or item.get("bucket_time")),
                    item.get("total_inspections", 0),
                    item.get("total_accept", 0),
                    item.get("total_reject", 0),
                    _safe_text(item.get("line_id")),
                    _safe_text(item.get("station_id")),
                ),
            )
        self.dashboard_count_var.set(f"{len(buckets)} buckets shown")

    def refresh_dashboard(self) -> None:
        self._set_status("Loading dashboard…")
        summary_params, bucket_params = self._dashboard_filters()

        def _load():
            return self.api.dashboard_summary(summary_params), self.api.dashboard_buckets(bucket_params)

        def _done(result, error):
            if error:
                self._set_status(f"Dashboard error: {error}")
                return
            summary, buckets = result
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
            self._render_dashboard_buckets(buckets)
            self.dashboard_raw.set_payload({"summary": summary, "buckets": buckets})
            self.dashboard_context_var.set(
                f"Summary loaded | total={summary.get('total_inspections', 0)} | granularity={bucket_params.get('granularity')}"
            )
            self._set_status("Dashboard berhasil direfresh.")

        run_async(self, _load, callback=_done)

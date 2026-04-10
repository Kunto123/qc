from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from client_tk.app.api_client import ApiClient
from client_tk.app.config import DEFAULT_SERVER_URL
from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    APP_BG,
    PANEL_BG,
    SHELL_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_ON_ACCENT,
    configure_customtkinter,
    configure_ttk_navy_theme,
)
from client_tk.app.screens.admin.view import AdminScreen
from client_tk.app.screens.engineer.view import EngineerScreen
from client_tk.app.screens.operator.view import OperatorScreen
from client_tk.app.services.session_state import SessionState
from shared.contracts.enums import UserRole


ROLE_SCREEN_MAP = {
    UserRole.ADMIN.value: AdminScreen,
    UserRole.OPERATOR.value: OperatorScreen,
    UserRole.ENGINEER.value: EngineerScreen,
}


class LoginFrame(ctk.CTkFrame):
    def __init__(self, master, on_login) -> None:
        super().__init__(master, fg_color=APP_BG, corner_radius=0)
        self._on_login = on_login

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        card = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=20, border_width=1, border_color="#26445f")
        card.grid(row=0, column=0, sticky="n", padx=32, pady=32)
        card.grid_columnconfigure(0, weight=0)
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card,
            text="QC Suite Python",
            font=("Segoe UI", 22, "bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=24, pady=(24, 4))
        ctk.CTkLabel(
            card,
            text="Client desktop interface with a dark navy CustomTkinter shell.",
            font=("Segoe UI", 11),
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=24, pady=(0, 20))

        ctk.CTkLabel(card, text="Server URL", text_color=TEXT_PRIMARY).grid(row=2, column=0, sticky="w", padx=24, pady=8)
        ctk.CTkLabel(card, text="Username", text_color=TEXT_PRIMARY).grid(row=3, column=0, sticky="w", padx=24, pady=8)
        ctk.CTkLabel(card, text="Password", text_color=TEXT_PRIMARY).grid(row=4, column=0, sticky="w", padx=24, pady=8)

        self.base_url_var = tk.StringVar(value=DEFAULT_SERVER_URL)
        self.username_var = tk.StringVar(value="operator")
        self.password_var = tk.StringVar(value="operator123")

        self.base_url_entry = ctk.CTkEntry(card, textvariable=self.base_url_var, width=420, fg_color="#0f1c2b", border_color="#26445f", text_color=TEXT_PRIMARY)
        self.username_entry = ctk.CTkEntry(card, textvariable=self.username_var, width=420, fg_color="#0f1c2b", border_color="#26445f", text_color=TEXT_PRIMARY)
        self.password_entry = ctk.CTkEntry(card, textvariable=self.password_var, show="*", width=420, fg_color="#0f1c2b", border_color="#26445f", text_color=TEXT_PRIMARY)
        self.base_url_entry.grid(row=2, column=1, sticky="ew", padx=(0, 24), pady=8)
        self.username_entry.grid(row=3, column=1, sticky="ew", padx=(0, 24), pady=8)
        self.password_entry.grid(row=4, column=1, sticky="ew", padx=(0, 24), pady=8)

        ctk.CTkButton(
            card,
            text="Login",
            command=self._submit,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            width=140,
        ).grid(row=5, column=1, sticky="e", padx=(0, 24), pady=(18, 24))

        self.username_entry.bind("<Return>", lambda _event: self._submit())
        self.password_entry.bind("<Return>", lambda _event: self._submit())

    def _submit(self) -> None:
        self._on_login(
            self.base_url_var.get().strip(),
            self.username_var.get().strip(),
            self.password_var.get().strip(),
        )

    def focus_credentials(self) -> None:
        self.username_entry.focus_set()

    def set_base_url(self, value: str) -> None:
        self.base_url_var.set(value)


class QcSuiteDesktopApp(ctk.CTk):
    def __init__(self) -> None:
        configure_customtkinter()
        super().__init__()
        self.title("QC Suite Python")
        self.geometry("1440x900")
        self.minsize(1160, 720)
        self.configure(fg_color=APP_BG)

        style = ttk.Style(self)
        configure_ttk_navy_theme(style)

        self.api = ApiClient(DEFAULT_SERVER_URL)
        self.session_state = SessionState(base_url=DEFAULT_SERVER_URL)
        self.active_screen: ttk.Frame | None = None

        self.login_frame = LoginFrame(self, self._handle_login)
        self.shell = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        self.shell.pack_forget()

        header = ctk.CTkFrame(self.shell, fg_color=SHELL_BG, corner_radius=0, border_width=0)
        header.pack(fill="x")
        ctk.CTkLabel(header, text="QC Suite Python", font=("Segoe UI", 16, "bold"), text_color=TEXT_PRIMARY).pack(side="left", padx=(16, 12), pady=14)
        self.user_label = ctk.CTkLabel(header, text="Not authenticated", text_color=TEXT_SECONDARY)
        self.user_label.pack(side="left", padx=16)
        self.endpoint_label = ctk.CTkLabel(header, text=DEFAULT_SERVER_URL, text_color=TEXT_SECONDARY)
        self.endpoint_label.pack(side="left", padx=16)
        ctk.CTkButton(
            header,
            text="Logout",
            command=self._logout,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            width=120,
        ).pack(side="right", padx=16, pady=12)

        self.screen_host = ctk.CTkFrame(self.shell, fg_color=APP_BG, corner_radius=0)
        self.screen_host.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_login()

    def _show_login(self) -> None:
        self._teardown_screen()
        self.shell.pack_forget()
        self.login_frame.set_base_url(self.session_state.base_url)
        self.login_frame.pack(fill="both", expand=True)
        self.login_frame.focus_credentials()

    def _show_shell(self) -> None:
        self.login_frame.pack_forget()
        self.shell.pack(fill="both", expand=True)

    def _handle_login(self, base_url: str, username: str, password: str) -> None:
        if not base_url or not username or not password:
            messagebox.showerror("Login", "Server URL, username, dan password wajib diisi.")
            return
        try:
            self.api = ApiClient(base_url)
            auth_payload = self.api.login(username, password)
            token = auth_payload["token"]
            self.api.set_token(token)
            user = self.api.me()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Login failed", str(exc))
            return

        self.session_state = SessionState(base_url=base_url, token=token, user=user)
        self._mount_screen(user.get("role"))

    def _mount_screen(self, role: str | None) -> None:
        screen_class = ROLE_SCREEN_MAP.get(str(role or "").strip())
        if screen_class is None:
            messagebox.showerror("Role", f"Unsupported role: {role}")
            self._show_login()
            return
        self._teardown_screen()
        self.user_label.configure(text=f"{self.session_state.user.get('username')} ({self.session_state.user.get('role')})")
        self.endpoint_label.configure(text=self.session_state.base_url)
        self.active_screen = screen_class(self.screen_host, self.api, self.session_state)
        self.active_screen.pack(fill="both", expand=True)
        self._show_shell()

    def _teardown_screen(self) -> None:
        if self.active_screen is None:
            return
        if hasattr(self.active_screen, "shutdown"):
            try:
                self.active_screen.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.active_screen.destroy()
        self.active_screen = None

    def _logout(self) -> None:
        base_url = self.session_state.base_url or DEFAULT_SERVER_URL
        if self.session_state.token:
            try:
                self.api.logout()
            except Exception:  # noqa: BLE001
                pass
        self._teardown_screen()
        self.api.set_token(None)
        self.session_state = SessionState(base_url=base_url)
        self.user_label.configure(text="Not authenticated")
        self.endpoint_label.configure(text=base_url)
        self._show_login()

    def _on_close(self) -> None:
        self._teardown_screen()
        self.destroy()


def launch() -> None:
    app = QcSuiteDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    launch()

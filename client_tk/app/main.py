from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from client_tk.app.api_client import ApiClient
from client_tk.app.config import DEFAULT_SERVER_URL
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


class LoginFrame(ttk.Frame):
    def __init__(self, master, on_login) -> None:
        super().__init__(master, padding=24)
        self._on_login = on_login

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        card = ttk.LabelFrame(self, text="Login")
        card.grid(row=0, column=0, sticky="nsew")
        for index in range(2):
            card.columnconfigure(index, weight=1)

        ttk.Label(card, text="Server URL").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Label(card, text="Username").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Label(card, text="Password").grid(row=2, column=0, sticky="w", padx=8, pady=6)

        self.base_url_var = tk.StringVar(value=DEFAULT_SERVER_URL)
        self.username_var = tk.StringVar(value="operator")
        self.password_var = tk.StringVar(value="operator123")

        self.base_url_entry = ttk.Entry(card, textvariable=self.base_url_var, width=48)
        self.username_entry = ttk.Entry(card, textvariable=self.username_var, width=48)
        self.password_entry = ttk.Entry(card, textvariable=self.password_var, show="*", width=48)
        self.base_url_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.username_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        self.password_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=6)

        ttk.Button(card, text="Login", command=self._submit).grid(row=3, column=1, sticky="e", padx=8, pady=12)

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


class QcSuiteDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QC Suite Python")
        self.geometry("1440x900")
        self.minsize(1160, 720)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.api = ApiClient(DEFAULT_SERVER_URL)
        self.state = SessionState(base_url=DEFAULT_SERVER_URL)
        self.active_screen: ttk.Frame | None = None

        self.login_frame = LoginFrame(self, self._handle_login)
        self.shell = ttk.Frame(self)
        self.shell.pack_forget()

        header = ttk.Frame(self.shell, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="QC Suite Python", font=("Segoe UI", 16, "bold")).pack(side="left")
        self.user_label = ttk.Label(header, text="Not authenticated")
        self.user_label.pack(side="left", padx=16)
        self.endpoint_label = ttk.Label(header, text=DEFAULT_SERVER_URL)
        self.endpoint_label.pack(side="left", padx=16)
        ttk.Button(header, text="Logout", command=self._logout).pack(side="right")

        self.screen_host = ttk.Frame(self.shell, padding=(8, 0, 8, 8))
        self.screen_host.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_login()

    def _show_login(self) -> None:
        self._teardown_screen()
        self.shell.pack_forget()
        self.login_frame.set_base_url(self.state.base_url)
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

        self.state = SessionState(base_url=base_url, token=token, user=user)
        self._mount_screen(user.get("role"))

    def _mount_screen(self, role: str | None) -> None:
        screen_class = ROLE_SCREEN_MAP.get(str(role or "").strip())
        if screen_class is None:
            messagebox.showerror("Role", f"Unsupported role: {role}")
            self._show_login()
            return
        self._teardown_screen()
        self.user_label.configure(text=f"{self.state.user.get('username')} ({self.state.user.get('role')})")
        self.endpoint_label.configure(text=self.state.base_url)
        self.active_screen = screen_class(self.screen_host, self.api, self.state)
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
        base_url = self.state.base_url or DEFAULT_SERVER_URL
        if self.state.token:
            try:
                self.api.logout()
            except Exception:  # noqa: BLE001
                pass
        self._teardown_screen()
        self.api.set_token(None)
        self.state = SessionState(base_url=base_url)
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

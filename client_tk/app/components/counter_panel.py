from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from client_tk.app.theme import BORDER, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY


BREAKDOWN_ORDER = (
    "OUT_OF_ANGLE",
    "WRONG_TYPE",
    "COMMIT_TIMEOUT",
)


class CounterPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER)
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Session Counter", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w",
            padx=12,
            pady=(12, 8),
        )

        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(fill="x")

        self.total_value = self._build_card(cards, "Total", "#0f172a", "#f8fafc")
        self.accept_value = self._build_card(cards, "Accept", "#166534", "#f0fdf4")
        self.reject_value = self._build_card(cards, "Reject", "#991b1b", "#fef2f2")

        self.meta_var = ctk.CTkLabel(self, text="Scope: session", text_color=TEXT_SECONDARY)
        self.meta_var.pack(anchor="w", padx=12, pady=(10, 12))

    def _build_card(self, master, title: str, bg: str, fg: str) -> tk.Label:
        frame = ctk.CTkFrame(master, fg_color=bg, corner_radius=14)
        frame.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkLabel(frame, text=title, text_color=fg, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        value = ctk.CTkLabel(frame, text="0", text_color=fg, font=("Segoe UI", 24, "bold"))
        value.pack(anchor="w", padx=12, pady=(0, 10))
        return value

    def update_payload(self, payload: dict | None) -> None:
        counters = (payload or {}).get("counters") or {}
        self.total_value.configure(text=str(counters.get("session_total", 0)))
        self.accept_value.configure(text=str(counters.get("session_accept", 0)))
        self.reject_value.configure(text=str(counters.get("session_reject", 0)))
        self.meta_var.configure(text=f"Scope: {counters.get('scope') or 'session'}")

    def reset(self) -> None:
        self.total_value.configure(text="0")
        self.accept_value.configure(text="0")
        self.reject_value.configure(text="0")
        self.meta_var.configure(text="Scope: session")

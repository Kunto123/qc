from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from client_tk.app.theme import BORDER, PANEL_ALT_BG, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY


BREAKDOWN_ORDER = (
    "NOT_FOUND",
    "WRONG_TYPE",
    "OUT_OF_POSITION",
    "LOW_CLASS_CONF",
    "LOW_ROI_CONF",
    "PART_NOT_READY",
    "ERROR",
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
        self.meta_var.pack(anchor="w", padx=12, pady=(10, 8))

        ctk.CTkLabel(self, text="Reject Breakdown", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w",
            padx=12,
            pady=(4, 6),
        )
        breakdown_frame = ctk.CTkFrame(self, fg_color=PANEL_ALT_BG, corner_radius=12, border_width=1, border_color=BORDER)
        breakdown_frame.pack(fill="x", padx=12, pady=(0, 12))
        breakdown_frame.grid_columnconfigure(1, weight=1)
        self.breakdown_labels: dict[str, ctk.CTkLabel] = {}
        for row, key in enumerate(BREAKDOWN_ORDER):
            ctk.CTkLabel(breakdown_frame, text=f"{key}:", text_color=TEXT_PRIMARY).grid(row=row, column=0, sticky="w", padx=10, pady=3)
            label = ctk.CTkLabel(breakdown_frame, text="0", text_color=TEXT_SECONDARY)
            label.grid(row=row, column=1, sticky="e", padx=10, pady=3)
            self.breakdown_labels[key] = label

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
        breakdown = counters.get("session_reject_breakdown") or {}
        for key in BREAKDOWN_ORDER:
            self.breakdown_labels[key].configure(text=str(breakdown.get(key, 0)))

    def reset(self) -> None:
        self.total_value.configure(text="0")
        self.accept_value.configure(text="0")
        self.reject_value.configure(text="0")
        self.meta_var.configure(text="Scope: session")
        for label in self.breakdown_labels.values():
            label.configure(text="0")

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


BREAKDOWN_ORDER = (
    "NOT_FOUND",
    "WRONG_TYPE",
    "OUT_OF_POSITION",
    "LOW_CLASS_CONF",
    "LOW_ROI_CONF",
    "PART_NOT_READY",
    "ERROR",
)


class CounterPanel(ttk.LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Session Counter", padding=12)
        cards = ttk.Frame(self)
        cards.pack(fill="x")

        self.total_value = self._build_card(cards, "Total", "#0f172a", "#f8fafc")
        self.accept_value = self._build_card(cards, "Accept", "#166534", "#f0fdf4")
        self.reject_value = self._build_card(cards, "Reject", "#991b1b", "#fef2f2")

        self.meta_var = ttk.Label(self, text="Scope: session")
        self.meta_var.pack(anchor="w", pady=(10, 8))

        breakdown_frame = ttk.LabelFrame(self, text="Reject Breakdown")
        breakdown_frame.pack(fill="x")
        self.breakdown_labels: dict[str, ttk.Label] = {}
        for row, key in enumerate(BREAKDOWN_ORDER):
            ttk.Label(breakdown_frame, text=f"{key}:").grid(row=row, column=0, sticky="w", padx=8, pady=2)
            label = ttk.Label(breakdown_frame, text="0")
            label.grid(row=row, column=1, sticky="e", padx=8, pady=2)
            self.breakdown_labels[key] = label

    def _build_card(self, master, title: str, bg: str, fg: str) -> tk.Label:
        frame = tk.Frame(master, bg=bg, padx=14, pady=10)
        frame.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(frame, text=title, bg=bg, fg=fg, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        value = tk.Label(frame, text="0", bg=bg, fg=fg, font=("Segoe UI", 24, "bold"))
        value.pack(anchor="w")
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

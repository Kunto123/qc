from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, *, padding: int = 0):
        super().__init__(master, padding=padding)
        style = ttk.Style()
        background = style.lookup("TFrame", "background") or "#f3f4f6"

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=background)
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.body = ttk.Frame(self)
        self._window_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.body.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_body_width)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux_up, add="+")
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux_down, add="+")

    def _sync_scroll_region(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_body_width(self, event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if event.delta == 0:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux_up(self, _event) -> None:
        self.canvas.yview_scroll(-1, "units")

    def _on_mousewheel_linux_down(self, _event) -> None:
        self.canvas.yview_scroll(1, "units")

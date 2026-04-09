from __future__ import annotations

import tkinter as tk
import weakref
from tkinter import ttk


_SCROLLABLE_FRAMES: weakref.WeakSet["ScrollableFrame"] = weakref.WeakSet()
_SCROLL_DISPATCH_BOUND = False


def _dispatch_mousewheel(event) -> None:
    widget = getattr(event, "widget", None)
    if widget is None:
        return

    widget_path = str(widget)
    matching_frames = []
    for frame in list(_SCROLLABLE_FRAMES):
        if _frame_is_alive(frame) and frame._contains_widget_path(widget_path):
            matching_frames.append(frame)
    if not matching_frames:
        return

    matching_frames.sort(key=lambda frame: len(str(frame.body)), reverse=True)
    for frame in matching_frames:
        if frame._scroll_from_event(event):
            break


def _frame_is_alive(frame: "ScrollableFrame") -> bool:
    try:
        return bool(frame.winfo_exists())
    except tk.TclError:
        return False


class AutoHideScrollbar(ttk.Scrollbar):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._grid_kwargs: dict | None = None
        self._pack_kwargs: dict | None = None

    def grid(self, **kwargs):
        self._grid_kwargs = dict(kwargs)
        self._pack_kwargs = None
        return super().grid(**kwargs)

    def pack(self, **kwargs):
        self._pack_kwargs = dict(kwargs)
        self._grid_kwargs = None
        return super().pack(**kwargs)

    def set(self, first, last):
        first_f = float(first)
        last_f = float(last)
        fully_visible = first_f <= 0.0 and last_f >= 1.0
        manager = self.winfo_manager()

        if fully_visible:
            if manager == "grid":
                self.grid_remove()
            elif manager == "pack":
                self.pack_forget()
        elif not manager:
            if self._grid_kwargs is not None:
                super().grid(**self._grid_kwargs)
            elif self._pack_kwargs is not None:
                super().pack(**self._pack_kwargs)

        super().set(first, last)


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, *, padding: int = 0):
        super().__init__(master, padding=padding)
        style = ttk.Style()
        background = style.lookup("TFrame", "background") or "#f3f4f6"

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=background)
        self.v_scrollbar = AutoHideScrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.body = ttk.Frame(self)
        self._window_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        _SCROLLABLE_FRAMES.add(self)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.body.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_body_width)
        global _SCROLL_DISPATCH_BOUND
        if not _SCROLL_DISPATCH_BOUND:
            self.winfo_toplevel().bind_all("<MouseWheel>", _dispatch_mousewheel, add="+")
            self.winfo_toplevel().bind_all("<Button-4>", _dispatch_mousewheel, add="+")
            self.winfo_toplevel().bind_all("<Button-5>", _dispatch_mousewheel, add="+")
            _SCROLL_DISPATCH_BOUND = True

    def _sync_scroll_region(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_body_width(self, event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _contains_widget_path(self, widget_path: str) -> bool:
        body_path = str(self.body)
        return widget_path == body_path or widget_path.startswith(f"{body_path}.")

    def _scroll_from_event(self, event) -> bool:
        if not self.v_scrollbar.winfo_manager():
            return False

        direction = 0
        event_num = getattr(event, "num", None)
        if event_num == 4:
            direction = -1
        elif event_num == 5:
            direction = 1
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return False
            direction = -1 if delta > 0 else 1

        first, last = self.canvas.yview()
        if direction < 0 and first <= 0.0:
            return False
        if direction > 0 and last >= 1.0:
            return False

        self.canvas.yview_scroll(direction, "units")
        return True

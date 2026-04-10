from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk

import cv2
import numpy as np
from PIL import Image, ImageTk

from client_tk.app.theme import BORDER, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY


_COLOR_PART_READY = (50, 180, 255)   # BGR oranye
_COLOR_STICKER = (0, 200, 255)       # BGR kuning
_COLOR_CROSSHAIR = (255, 255, 255)   # BGR putih
_ARM = 18
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_roi_box(frame, x: int, y: int, w: int, h: int, color, label: str) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(16, y - 6)),
        _LABEL_FONT,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_crosshair(frame, px: int, py: int, color, label: str = "") -> None:
    cv2.line(frame, (px - _ARM, py), (px + _ARM, py), color, 2, cv2.LINE_AA)
    cv2.line(frame, (px, py - _ARM), (px, py + _ARM), color, 2, cv2.LINE_AA)
    cv2.circle(frame, (px, py), 5, color, 1, cv2.LINE_AA)
    if label:
        cv2.putText(
            frame,
            label,
            (px + _ARM + 4, py + 5),
            _LABEL_FONT,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


class RoiPickerCanvas(ctk.CTkFrame):
    """Interactive canvas for visualising ROI boxes and picking expected center.

    Usage::

        picker = RoiPickerCanvas(parent, title="ROI Picker", size=(640, 360))
        picker.on_center_changed = lambda cx, cy: print(cx, cy)
        picker.load_image(bgr_frame)
        picker.set_rois(part_ready_roi={"x":0.1,"y":0.1,"w":0.1,"h":0.1},
                        sticker_roi={"x":0.14,"y":0.25,"w":0.73,"h":0.37})
        picker.set_expected_center(0.5, 0.5)
    """

    def __init__(self, master, title: str = "ROI Visual Picker", *, size: tuple[int, int] = (640, 360)):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self._display_size = size
        self.configure(width=size[0], height=size[1] + 56)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ctk.CTkLabel(self, text=title, font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(10, 6),
        )
        self._source_frame: np.ndarray | None = None
        self._part_ready_roi: dict = {}
        self._sticker_roi: dict = {}
        self._cx: float = 0.5
        self._cy: float = 0.5
        self.on_center_changed: Callable[[float, float], None] | None = None

        self._canvas = tk.Canvas(
            self,
            width=size[0],
            height=size[1],
            bg="#0f172a",
            cursor="crosshair",
            highlightthickness=0,
        )
        self._canvas.grid(row=1, column=0, sticky="nw", padx=10)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Configure>", lambda _: self.redraw())

        self._hint = ctk.CTkLabel(
            self,
            text="Klik dalam area kuning (Sticker ROI) untuk set expected center.",
            font=("Segoe UI", 9),
            wraplength=size[0] - 12,
            text_color=TEXT_SECONDARY,
        )
        self._hint.grid(row=2, column=0, sticky="w", padx=10, pady=(6, 10))
        self._photo = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, bgr_frame: np.ndarray | None) -> None:
        self._source_frame = bgr_frame.copy() if bgr_frame is not None else None
        self.redraw()

    def set_rois(self, part_ready_roi: dict | None = None, sticker_roi: dict | None = None) -> None:
        if part_ready_roi is not None:
            self._part_ready_roi = dict(part_ready_roi)
        if sticker_roi is not None:
            self._sticker_roi = dict(sticker_roi)
        self.redraw()

    def set_expected_center(self, cx: float | None, cy: float | None) -> None:
        self._cx = float(cx) if cx is not None else 0.5
        self._cy = float(cy) if cy is not None else 0.5
        self.redraw()

    def clear(self) -> None:
        self._source_frame = None
        self._canvas.delete("all")
        self._photo = None
        self._canvas.configure(bg="#0f172a")

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def redraw(self) -> None:
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 8:
            cw = self._display_size[0]
        if ch < 8:
            ch = self._display_size[1]

        if self._source_frame is None:
            self._canvas.delete("all")
            self._canvas.create_text(
                cw // 2,
                ch // 2,
                text="Belum ada gambar.\nKlik 'Load Image' atau 'Load from Session'.",
                fill="#475569",
                font=("Segoe UI", 11),
                justify="center",
            )
            return

        frame = self._render_frame(cw, ch)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _render_frame(self, cw: int, ch: int) -> np.ndarray:
        src_h, src_w = self._source_frame.shape[:2]
        scale = min(cw / max(src_w, 1), ch / max(src_h, 1))
        dw = max(1, int(src_w * scale))
        dh = max(1, int(src_h * scale))
        frame = cv2.resize(self._source_frame, (dw, dh), interpolation=cv2.INTER_AREA)

        # Expand canvas to (cw, ch) with black padding
        canvas_frame = np.zeros((ch, cw, 3), dtype=np.uint8)
        off_x = (cw - dw) // 2
        off_y = (ch - dh) // 2
        canvas_frame[off_y:off_y + dh, off_x:off_x + dw] = frame

        def _roi_px(roi: dict) -> tuple[int, int, int, int]:
            rx = off_x + int(float(roi.get("x", 0.0)) * dw)
            ry = off_y + int(float(roi.get("y", 0.0)) * dh)
            rw = max(1, int(float(roi.get("w", 1.0)) * dw))
            rh = max(1, int(float(roi.get("h", 1.0)) * dh))
            return rx, ry, rw, rh

        if self._part_ready_roi:
            rx, ry, rw, rh = _roi_px(self._part_ready_roi)
            _draw_roi_box(canvas_frame, rx, ry, rw, rh, _COLOR_PART_READY, "Part Ready ROI")

        if self._sticker_roi:
            rx, ry, rw, rh = _roi_px(self._sticker_roi)
            _draw_roi_box(canvas_frame, rx, ry, rw, rh, _COLOR_STICKER, "Sticker ROI")
            exp_px = rx + int(self._cx * rw)
            exp_py = ry + int(self._cy * rh)
            _draw_crosshair(canvas_frame, exp_px, exp_py, _COLOR_CROSSHAIR, "EXP CTR")

        return canvas_frame

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:
        if self._source_frame is None or not self._sticker_roi:
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        src_h, src_w = self._source_frame.shape[:2]
        scale = min(cw / max(src_w, 1), ch / max(src_h, 1))
        dw = max(1, int(src_w * scale))
        dh = max(1, int(src_h * scale))
        off_x = (cw - dw) // 2
        off_y = (ch - dh) // 2

        roi = self._sticker_roi
        rx = off_x + int(float(roi.get("x", 0.0)) * dw)
        ry = off_y + int(float(roi.get("y", 0.0)) * dh)
        rw = max(1, int(float(roi.get("w", 1.0)) * dw))
        rh = max(1, int(float(roi.get("h", 1.0)) * dh))

        if not (rx <= event.x <= rx + rw and ry <= event.y <= ry + rh):
            self._hint.configure(text="Klik dalam area kuning (Sticker ROI) untuk set expected center.")
            return

        cx = max(0.0, min(1.0, (event.x - rx) / rw))
        cy = max(0.0, min(1.0, (event.y - ry) / rh))
        self._cx = round(cx, 4)
        self._cy = round(cy, 4)
        self._hint.configure(text=f"Expected center: x={self._cx:.4f}, y={self._cy:.4f}  (tersimpan ke field di bawah)")
        self.redraw()
        if self.on_center_changed:
            self.on_center_changed(self._cx, self._cy)

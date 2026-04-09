from __future__ import annotations

import base64
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk


class LiveView(ttk.LabelFrame):
    def __init__(self, master, title: str, *, size: tuple[int, int] = (360, 240)):
        super().__init__(master, text=title)
        self._size = size
        self.configure(width=size[0], height=size[1])
        self.pack_propagate(False)
        self.grid_propagate(False)
        self._label = tk.Label(
            self,
            text="No frame",
            bg="#0f172a",
            fg="#cbd5e1",
            anchor="center",
            font=("Segoe UI", 11),
        )
        self._label.pack(fill="both", expand=True, padx=8, pady=8)
        self._photo = None

    def update_bgr(self, frame) -> None:
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        target_width = self._label.winfo_width()
        target_height = self._label.winfo_height()
        if target_width <= 8:
            target_width = self._size[0]
        if target_height <= 8:
            target_height = self._size[1]
        image = ImageOps.contain(image, (target_width, target_height))
        self._photo = ImageTk.PhotoImage(image)
        self._label.configure(image=self._photo, text="")

    def update_b64(self, image_b64: str | None) -> None:
        if not image_b64:
            return
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self.update_bgr(frame)

    def reset(self) -> None:
        self._photo = None
        self._label.configure(image="", text="No frame")

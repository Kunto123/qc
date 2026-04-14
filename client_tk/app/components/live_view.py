from __future__ import annotations

import base64
import tkinter as tk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageOps

from client_tk.app.theme import BORDER, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY


class LiveView(ctk.CTkFrame):
    def __init__(self, master, title: str, *, size: tuple[int, int] = (360, 240)):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER)
        self._size = size
        self.configure(width=size[0], height=size[1])
        self.pack_propagate(False)
        self.grid_propagate(False)
        self._title = ctk.CTkLabel(
            self,
            text=title,
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT_PRIMARY,
        )
        self._title.pack(fill="x", padx=12, pady=(10, 0))
        self._label = ctk.CTkLabel(
            self,
            text="No frame",
            fg_color="#0f172a",
            text_color=TEXT_SECONDARY,
            anchor="center",
            font=("Segoe UI", 11),
        )
        self._label.pack(fill="both", expand=True, padx=10, pady=10)
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
        self._photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self._label.configure(image=self._photo, text="", fg_color=PANEL_BG)

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
        self._label.configure(image=None, text="No frame", fg_color="#0f172a")

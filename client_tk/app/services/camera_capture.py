from __future__ import annotations

import threading
import time

import cv2


class CameraCaptureService:
    def __init__(self) -> None:
        self._capture = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._frame = None
        self._camera_index = 0

    def start(self, camera_index: int) -> None:
        self.stop()
        self._camera_index = int(camera_index)
        self._capture = None
        backend_candidates = []
        if hasattr(cv2, "CAP_DSHOW"):
            backend_candidates.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backend_candidates.append(cv2.CAP_MSMF)
        backend_candidates.append(None)

        for backend in backend_candidates:
            capture = cv2.VideoCapture(self._camera_index) if backend is None else cv2.VideoCapture(self._camera_index, backend)
            if capture.isOpened():
                self._capture = capture
                break
            capture.release()

        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._capture.set(cv2.CAP_PROP_FPS, 30)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # read() on most backends blocks until a frame is available; the sleep
        # only paces the loop when read() returns immediately (e.g. DSHOW non-block).
        while self._running and self._capture is not None:
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

    def get_latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._thread = None
        with self._lock:
            self._frame = None

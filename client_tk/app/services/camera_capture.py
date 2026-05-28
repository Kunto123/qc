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
        self._actual_settings: dict[str, float] = {}

    def start(
        self,
        camera_index: int,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
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
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width is not None and int(width) > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None and int(height) > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps is not None and float(fps) > 0:
            self._capture.set(cv2.CAP_PROP_FPS, float(fps))
        self._actual_settings = {
            "width": float(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0),
            "height": float(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0),
            "fps": float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0),
        }
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera-capture")
        self._thread.start()

    def _loop(self) -> None:
        # read() on most backends blocks until a frame is available; the sleep
        # only paces the loop when read() returns immediately (e.g. DSHOW non-block).
        capture = self._capture
        while self._running and capture is not None:
            ok, frame = capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

    def get_latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def actual_settings(self) -> dict[str, float]:
        return dict(self._actual_settings)

    @property
    def is_running(self) -> bool:
        return bool(self._running and self._capture is not None)

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._thread = None
        self._actual_settings = {}
        with self._lock:
            self._frame = None

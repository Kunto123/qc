from __future__ import annotations

import os
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
        self._reconnecting = False
        self._reconnect_interval_s = 5.0
        self._last_frame_before_disconnect = None
        self._disconnect_notified = False
        self._status_callback = None  # callable(status: str)

    def set_status_callback(self, callback) -> None:
        """Set callback for status updates: 'connected', 'reconnecting', 'error'."""
        self._status_callback = callback

    def _notify_status(self, status: str) -> None:
        if self._status_callback:
            try:
                self._status_callback(status)
            except Exception:
                pass

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
        self._reconnecting = False
        self._disconnect_notified = False
        self._last_frame_before_disconnect = None
        self._open_camera(width, height, fps)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera-capture")
        self._thread.start()
        self._notify_status("connected")

    def _open_camera(self, width=None, height=None, fps=None) -> None:
        """Open camera with backend fallback."""
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
            raise RuntimeError(f"Cannot open camera index {self._camera_index}")
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Configurable exposure: QC_SUITE_CAMERA_AUTO_EXPOSURE=0 → manual mode
        _auto_exp = int(os.getenv("QC_SUITE_CAMERA_AUTO_EXPOSURE", "1"))
        if _auto_exp == 0:
            _exp_val = int(os.getenv("QC_SUITE_CAMERA_EXPOSURE_VALUE", "-6"))
            try:
                if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                    self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual
                if hasattr(cv2, "CAP_PROP_EXPOSURE"):
                    self._capture.set(cv2.CAP_PROP_EXPOSURE, _exp_val)
            except Exception:
                pass
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

    def _loop(self) -> None:
        """Main capture loop with auto-reconnect."""
        while self._running:
            capture = self._capture
            if capture is None or not capture.isOpened():
                # Camera disconnected — try to reconnect
                if not self._reconnecting:
                    self._reconnecting = True
                    self._notify_status("reconnecting")
                    # Save last frame for display during reconnect
                    with self._lock:
                        self._last_frame_before_disconnect = (
                            self._frame.copy() if self._frame is not None else None
                        )
                # Try reconnect every N seconds
                time.sleep(self._reconnect_interval_s)
                if not self._running:
                    break
                try:
                    self._open_camera(
                        width=int(self._actual_settings.get("width") or 0) or None,
                        height=int(self._actual_settings.get("height") or 0) or None,
                        fps=self._actual_settings.get("fps") or None,
                    )
                    if self._capture is not None and self._capture.isOpened():
                        self._reconnecting = False
                        self._last_frame_before_disconnect = None
                        self._disconnect_notified = False
                        self._notify_status("connected")
                        continue
                except Exception:
                    pass
                continue

            ok, frame = capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                # Frame read failed — camera may have disconnected
                if not self._reconnecting:
                    self._reconnecting = True
                    self._notify_status("reconnecting")
                    with self._lock:
                        self._last_frame_before_disconnect = (
                            self._frame.copy() if self._frame is not None else None
                        )
                # Release broken capture and try again next loop
                try:
                    capture.release()
                except Exception:
                    pass
                self._capture = None
                time.sleep(self._reconnect_interval_s)

    def get_latest_frame(self):
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            if self._last_frame_before_disconnect is not None:
                return self._last_frame_before_disconnect.copy()
            return None

    def attempt_reconnect(self) -> bool:
        """Force an immediate reconnect attempt. Returns True if successful."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        self._reconnecting = False
        return False  # Next _loop iteration will try to reconnect

    @property
    def actual_settings(self) -> dict[str, float]:
        return dict(self._actual_settings)

    @property
    def is_running(self) -> bool:
        return bool(self._running and self._capture is not None)

    @property
    def is_active(self) -> bool:
        """Return True while capture service is running (even during reconnect)."""
        return self._running

    @property
    def is_reconnecting(self) -> bool:
        return self._reconnecting

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._thread = None
        self._actual_settings = {}
        self._reconnecting = False
        self._last_frame_before_disconnect = None
        with self._lock:
            self._frame = None

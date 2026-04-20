from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable

import cv2


class FrameUploadService:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_client_timings: dict = {}
        self._timings_lock = threading.Lock()

    @property
    def last_client_timings(self) -> dict:
        with self._timings_lock:
            return dict(self._last_client_timings)

    def start(
        self,
        *,
        interval_ms: int,
        get_frame: Callable[[], object],
        send_frame: Callable[[str], dict],
        on_result: Callable[[dict], None],
        on_error: Callable[[str], None],
        jpeg_quality: int = 75,
        resize_width: int | None = None,
    ) -> None:
        self.stop()
        self._running = True
        target_interval_s = max(0.05, interval_ms / 1000.0)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(40, min(100, jpeg_quality))]

        def _loop():
            while self._running:
                loop_started = time.perf_counter()
                try:
                    t0 = time.perf_counter()
                    frame = get_frame()
                    capture_ms = (time.perf_counter() - t0) * 1000.0

                    if frame is None:
                        elapsed_s = time.perf_counter() - loop_started
                        sleep_s = target_interval_s - elapsed_s
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                        continue

                    resize_ms = 0.0
                    if resize_width is not None and frame.shape[1] > resize_width:
                        scale = resize_width / frame.shape[1]
                        new_h = int(frame.shape[0] * scale)
                        t0 = time.perf_counter()
                        frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_LINEAR)
                        resize_ms = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                    encode_ms = (time.perf_counter() - t0) * 1000.0

                    if not ok:
                        raise RuntimeError("Failed to encode frame.")

                    t0 = time.perf_counter()
                    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                    b64_ms = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    result = send_frame(image_b64)
                    request_ms = (time.perf_counter() - t0) * 1000.0

                    client_timings = {
                        "capture_ms": round(capture_ms, 2),
                        "resize_ms": round(resize_ms, 2),
                        "encode_ms": round(encode_ms, 2),
                        "b64_ms": round(b64_ms, 2),
                        "request_ms": round(request_ms, 2),
                        "frame_width": frame.shape[1],
                        "frame_height": frame.shape[0],
                        "jpeg_quality": jpeg_quality,
                        "payload_bytes": len(encoded),
                    }
                    with self._timings_lock:
                        self._last_client_timings = client_timings

                    if isinstance(result, dict):
                        result.setdefault("client_timings", client_timings)

                    on_result(result)
                except Exception as exc:  # noqa: BLE001
                    on_error(str(exc))

                if not self._running:
                    break

                elapsed_s = time.perf_counter() - loop_started
                sleep_s = target_interval_s - elapsed_s
                if sleep_s > 0:
                    time.sleep(sleep_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.3)
        self._thread = None
        with self._timings_lock:
            self._last_client_timings = {}

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

    def start(
        self,
        *,
        interval_ms: int,
        get_frame: Callable[[], object],
        send_frame: Callable[[str], dict],
        on_result: Callable[[dict], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.stop()
        self._running = True
        target_interval_s = max(0.1, interval_ms / 1000.0)

        def _loop():
            while self._running:
                started_at = time.perf_counter()
                try:
                    frame = get_frame()
                    if frame is None:
                        pass
                    else:
                        ok, encoded = cv2.imencode(".jpg", frame)
                        if not ok:
                            raise RuntimeError("Failed to encode frame.")
                        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                        result = send_frame(image_b64)
                        on_result(result)
                except Exception as exc:  # noqa: BLE001
                    on_error(str(exc))

                if not self._running:
                    break

                elapsed_s = time.perf_counter() - started_at
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


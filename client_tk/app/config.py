from __future__ import annotations

import os


DEFAULT_SERVER_URL = os.getenv("QC_SUITE_SERVER_URL", "http://127.0.0.1:8100")
DEFAULT_UPLOAD_INTERVAL_MS = int(os.getenv("QC_SUITE_UPLOAD_INTERVAL_MS", "500"))
# WebSocket streaming URL.  Empty string disables WebSocket streaming and
# falls back to HTTP frame upload (FrameUploadService).
DEFAULT_STREAM_URL = os.getenv("QC_SUITE_STREAM_URL", "ws://127.0.0.1:8101")


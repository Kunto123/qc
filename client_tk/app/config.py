from __future__ import annotations

import os


DEFAULT_LOCAL_ONLY = os.getenv("QC_SUITE_LOCAL_ONLY", "1").strip() != "0"
DEFAULT_SERVER_URL = os.getenv(
    "QC_SUITE_SERVER_URL",
    "local://embedded" if DEFAULT_LOCAL_ONLY else "http://127.0.0.1:8100",
)
DEFAULT_UPLOAD_INTERVAL_MS = int(os.getenv("QC_SUITE_UPLOAD_INTERVAL_MS", "150"))
# WebSocket streaming is disabled by default in local-only mode.
DEFAULT_STREAM_URL = os.getenv(
    "QC_SUITE_STREAM_URL",
    "" if DEFAULT_LOCAL_ONLY else "ws://127.0.0.1:8101",
)


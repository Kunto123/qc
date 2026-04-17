from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("QC_SUITE_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
JSON_STORE_DIR = DATA_ROOT / "json_store"
DATASETS_DIR = DATA_ROOT / "datasets"
MODELS_DIR = DATA_ROOT / "models"
DEFAULT_STICKER_MODEL_PATH = os.getenv(
    "QC_SUITE_DEFAULT_STICKER_MODEL_PATH",
    r"D:\ProjectMagang\akh.pt",
).strip()
DEFAULT_STICKER_MODEL_META_PATH = os.getenv(
    "QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH",
    r"D:\ProjectMagang\ds-43598c556c__yolov5mu__20260402-085412.meta.json",
).strip()


@dataclass(slots=True)
class AppConfig:
    host: str = os.getenv("QC_SUITE_HOST", "127.0.0.1")
    port: int = int(os.getenv("QC_SUITE_PORT", "8100"))
    debug: bool = os.getenv("QC_SUITE_DEBUG", "0").strip() == "1"
    secret_key: str = os.getenv("QC_SUITE_SECRET_KEY", "qc-suite-dev-secret")
    local_only: bool = os.getenv("QC_SUITE_LOCAL_ONLY", "1").strip() != "0"
    sql_enabled_flag: bool = os.getenv("QC_SUITE_SQL_ENABLED", "0").strip() == "1"
    access_token_ttl_seconds: int = max(60, int(os.getenv("QC_SUITE_ACCESS_TOKEN_TTL_SECONDS", "86400")))
    device_mode: str = os.getenv("QC_SUITE_DEVICE", "auto").strip().lower() or "auto"
    cuda_device_id: int = max(0, int(os.getenv("QC_SUITE_CUDA_DEVICE_ID", "0")))
    sql_server: str = os.getenv("MSSQL_SERVER", "").strip()
    sql_database: str = os.getenv("MSSQL_DATABASE", "").strip()
    sql_username: str = os.getenv("MSSQL_USERNAME", "").strip()
    sql_password: str = os.getenv("MSSQL_PASSWORD", "").strip()
    sql_driver: str = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
    sticker_inference_mode: str = os.getenv("QC_SUITE_STICKER_INFERENCE_MODE", "auto").strip().lower() or "auto"
    default_sticker_model_path: str = DEFAULT_STICKER_MODEL_PATH
    default_sticker_model_meta_path: str = DEFAULT_STICKER_MODEL_META_PATH
    training_engine_mode: str = os.getenv("QC_SUITE_TRAINING_ENGINE_MODE", "real").strip().lower() or "real"
    training_timeout_minutes: int = max(1, int(os.getenv("QC_SUITE_TRAINING_TIMEOUT_MINUTES", "30")))
    training_default_epochs: int = max(1, int(os.getenv("QC_SUITE_TRAINING_DEFAULT_EPOCHS", "1")))
    training_default_imgsz: int = max(64, int(os.getenv("QC_SUITE_TRAINING_DEFAULT_IMGSZ", "320")))
    training_default_batch: int = max(1, int(os.getenv("QC_SUITE_TRAINING_DEFAULT_BATCH", "4")))
    training_default_patience: int = max(1, int(os.getenv("QC_SUITE_TRAINING_DEFAULT_PATIENCE", "5")))
    push_worker_interval_seconds: int = max(5, int(os.getenv("QC_SUITE_PUSH_WORKER_INTERVAL_SECONDS", "30")))
    push_worker_max_retry: int = max(1, int(os.getenv("QC_SUITE_PUSH_WORKER_MAX_RETRY", "5")))
    # Logging toggles — default ON for backward compatibility.
    # Set to "0" to suppress 2xx/3xx access log noise while keeping >= 400 errors visible.
    access_logs_enabled: bool = os.getenv("QC_SUITE_ACCESS_LOGS_ENABLED", "1").strip() != "0"
    # Set to "0" to raise werkzeug request logger to ERROR level (suppresses per-request lines).
    werkzeug_logs_enabled: bool = os.getenv("QC_SUITE_WERKZEUG_REQUEST_LOGS_ENABLED", "1").strip() != "0"
    # GPU policy: when "1" (default), a training job requesting device=gpu will fail immediately
    # if CUDA is unavailable instead of silently falling back to CPU.
    # Set to "0" to restore legacy silent-fallback behavior.
    gpu_fail_fast: bool = os.getenv("QC_SUITE_GPU_FAIL_FAST", "1").strip() != "0"
    # Settle-time debounce system-wide default.
    # Templates that do NOT set part_ready_settle_ms (= None) inherit this value.
    # Templates that explicitly set part_ready_settle_ms (including 0 to bypass) ignore it.
    # Default 0 = no settle (backward compatible).
    part_ready_settle_ms_default: int = max(0, int(os.getenv("QC_SUITE_PART_READY_SETTLE_MS", "0")))
    # Training weights resolution policy.
    # When 1 (default): the worker will attempt to auto-download from Ultralytics Hub if the
    # weights file is not found locally. Suitable for dev environments with internet access.
    # When 0 (offline-strict): the worker fails immediately with an actionable error if the local
    # weights file is missing. Use for air-gapped / production environments.
    training_weights_download_allowed: bool = os.getenv("QC_SUITE_TRAINING_WEIGHTS_DOWNLOAD_ALLOWED", "1").strip() != "0"
    # Geometric augmentation feature flag.
    # When 0 (default): geometric transforms (flip_h, flip_v, rotate) are rejected when
    # including augment jobs in a version snapshot — label coords copied verbatim.
    # When 1: geometric transforms are allowed in version snapshots and the label geometry
    # engine transforms bbox/polygon annotations accordingly.
    geometric_augment_enabled: bool = os.getenv("QC_SUITE_GEOMETRIC_AUGMENT_ENABLED", "0").strip() == "1"
    # WebSocket streaming sidecar.
    # QC_SUITE_STREAM_PORT: port for the persistent-frame streaming server (default 8101).
    # QC_SUITE_STREAM_HOST: bind address for the streaming server; empty string means
    #   inherit QC_SUITE_HOST so the two servers share the same bind address.
    stream_port: int = max(1, int(os.getenv("QC_SUITE_STREAM_PORT", "8101")))
    stream_host: str = os.getenv("QC_SUITE_STREAM_HOST", "").strip()

    @property
    def sql_enabled(self) -> bool:
        if not self.sql_enabled_flag:
            return False
        return bool(
            self.sql_server
            and self.sql_database
            and self.sql_username
            and self.sql_password
        )


def ensure_data_dirs() -> None:
    for path in (DATA_ROOT, JSON_STORE_DIR, DATASETS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)

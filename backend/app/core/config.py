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
    relational_backend: str = os.getenv("QC_SUITE_DATABASE_BACKEND", "").strip().lower()
    sql_enabled_flag: bool = os.getenv("QC_SUITE_SQL_ENABLED", "0").strip() == "1"
    access_token_ttl_seconds: int = max(60, int(os.getenv("QC_SUITE_ACCESS_TOKEN_TTL_SECONDS", "86400")))
    device_mode: str = os.getenv("QC_SUITE_DEVICE", "auto").strip().lower() or "auto"
    cuda_device_id: int = max(0, int(os.getenv("QC_SUITE_CUDA_DEVICE_ID", "0")))
    sql_server: str = os.getenv("MSSQL_SERVER", "").strip()
    sql_database: str = os.getenv("MSSQL_DATABASE", "").strip()
    sql_username: str = os.getenv("MSSQL_USERNAME", "").strip()
    sql_password: str = os.getenv("MSSQL_PASSWORD", "").strip()
    sql_driver: str = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
    postgresql_host: str = os.getenv("POSTGRESQL_HOST", "").strip()
    postgresql_port: int = max(1, int(os.getenv("POSTGRESQL_PORT", "5432")))
    postgresql_database: str = os.getenv("POSTGRESQL_DATABASE", "").strip()
    postgresql_username: str = os.getenv("POSTGRESQL_USERNAME", "").strip()
    postgresql_password: str = os.getenv("POSTGRESQL_PASSWORD", "").strip()
    postgresql_schema: str = os.getenv("POSTGRESQL_SCHEMA", "public").strip() or "public"
    postgresql_sslmode: str = os.getenv("POSTGRESQL_SSLMODE", "prefer").strip().lower() or "prefer"
    sticker_inference_mode: str = os.getenv("QC_SUITE_STICKER_INFERENCE_MODE", "auto").strip().lower() or "auto"
    sticker_ocr_mode: str = os.getenv("QC_SUITE_STICKER_OCR_MODE", "legacy").strip().lower() or "legacy"
    sticker_ocr_required: bool = os.getenv("QC_SUITE_STICKER_OCR_REQUIRED", "0").strip() == "1"
    sticker_ocr_fail_fast: bool = os.getenv("QC_SUITE_STICKER_OCR_FAIL_FAST", "0").strip() == "1"
    default_ocr_engine: str = os.getenv("QC_SUITE_OCR_ENGINE", "disabled").strip().lower() or "disabled"
    default_ocr_min_confidence: float = max(0.0, min(1.0, float(os.getenv("QC_SUITE_OCR_MIN_CONFIDENCE", "0.70"))))
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
    # Consecutive reject threshold — number of consecutive reject decisions required
    # before a reject is actually committed (PLC reject). 0 = immediate (no delay).
    # Set to 2-3 to allow operator to reposition sticker before final reject.
    max_consecutive_rejects: int = max(0, int(os.getenv("QC_SUITE_MAX_CONSECUTIVE_REJECTS", "0")))
    # Part-ready latch release debounce (ms).
    # When raw part_ready drops (e.g. shadow), the latch stays active for this duration
    # so sticker inference is not cancelled by brief noise.
    # 0 = legacy behavior (latch resets immediately when raw drops).
    part_ready_release_ms_default: int = max(0, int(os.getenv("QC_SUITE_PART_READY_RELEASE_MS", "300")))
    # ── Inspection Policy ──
    # Hard reject reasons: only these trigger PLC buzzer/reject.
    # Non-hard reject reasons (NOT_FOUND, WRONG_TYPE, etc.) become pending/adjust
    # without PLC commit. Comma-separated list of RejectReasonCode values.
    inspect_hard_reject_reasons: str = os.getenv(
        "QC_SUITE_INSPECT_HARD_REJECT_REASONS", "OUT_OF_ANGLE"
    ).strip()
    # Commit grace period (ms): minimum time after inference starts before any
    # commit (accept or hard reject) is allowed. This gives operator time to
    # adjust sticker before final decision. Default 1500ms.
    commit_grace_ms: int = max(0, int(os.getenv("QC_SUITE_STICKER_COMMIT_GRACE_MS", "1500")))
    # Stability thresholds for commit:
    # accept: minimal consecutive stable frames before commit (default 2).
    accept_stable_frames: int = max(1, int(os.getenv("QC_SUITE_ACCEPT_STABLE_FRAMES", "2")))
    # accept: minimal stable elapsed ms before commit (default 200).
    accept_stable_ms: int = max(0, int(os.getenv("QC_SUITE_ACCEPT_STABLE_MS", "200")))
    # hard_reject: minimal consecutive stable frames before commit (default 3).
    hard_reject_stable_frames: int = max(1, int(os.getenv("QC_SUITE_HARD_REJECT_STABLE_FRAMES", "3")))
    # hard_reject: minimal stable elapsed ms before commit (default 500).
    hard_reject_stable_ms: int = max(0, int(os.getenv("QC_SUITE_HARD_REJECT_STABLE_MS", "500")))
    # Operator phase pacing. These are non-blocking gates; preview stays live.
    # STICKER_INSTALL delays sticker inference after clamp is ready, giving the
    # operator time to attach the sticker.
    # NEXT_PART keeps the system from immediately re-clamping after release.
    phase_sticker_install_delay_ms: int = max(0, int(os.getenv("QC_SUITE_PHASE_STICKER_INSTALL_DELAY_MS", "0")))
    phase_next_part_delay_ms: int = max(0, int(os.getenv("QC_SUITE_PHASE_NEXT_PART_DELAY_MS", "2000")))
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
    # Inference interval (ms): minimum time between YOLO inference runs.
    # 200 = max ~5 fps inference. 0 = every frame (unlimited).
    inference_interval_ms: int = max(0, int(os.getenv("QC_SUITE_INFERENCE_INTERVAL_MS", "0")))

    stream_port: int = max(1, int(os.getenv("QC_SUITE_STREAM_PORT", "8101")))
    stream_host: str = os.getenv("QC_SUITE_STREAM_HOST", "").strip()
    # PLC / Remote-IO clamp control.
    # QC_SUITE_PLC_ENABLED=1  — activate the PLC worker (default off).
    # QC_SUITE_PLC_DRY_RUN=1  — log commands only, no real socket (default on so
    #   accidentally enabling PLC without hardware never opens a TCP connection).
    # QC_SUITE_PLC_TRANSPORT=tcp|rtu — choose Modbus TCP gateway vs serial RTU relay.
    # QC_SUITE_PLC_HOST / QC_SUITE_PLC_PORT — Modbus TCP endpoint or gateway address.
    # QC_SUITE_PLC_SERIAL_PORT / BAUDRATE / PARITY / BYTESIZE / STOPBITS — RTU serial line.
    # QC_SUITE_PLC_TIMEOUT_MS — socket connect/send timeout.
    # QC_SUITE_PLC_CLAMP_HOLD_MS — ms to keep clamp engaged before auto-release.
    plc_enabled: bool = os.getenv("QC_SUITE_PLC_ENABLED", "0").strip() == "1"
    plc_dry_run: bool = os.getenv("QC_SUITE_PLC_DRY_RUN", "1").strip() != "0"
    plc_transport: str = os.getenv("QC_SUITE_PLC_TRANSPORT", "tcp").strip().lower() or "tcp"
    plc_host: str = os.getenv("QC_SUITE_PLC_HOST", "127.0.0.1").strip()
    plc_port: int = max(1, int(os.getenv("QC_SUITE_PLC_PORT", "5020")))
    plc_serial_port: str = os.getenv("QC_SUITE_PLC_SERIAL_PORT", "").strip()
    plc_serial_baudrate: int = int(os.getenv("QC_SUITE_PLC_SERIAL_BAUDRATE", "9600").strip() or "9600")
    plc_serial_parity: str = os.getenv("QC_SUITE_PLC_SERIAL_PARITY", "N").strip().upper() or "N"
    plc_serial_bytesize: int = int(os.getenv("QC_SUITE_PLC_SERIAL_BYTESIZE", "8").strip() or "8")
    plc_serial_stopbits: int = int(os.getenv("QC_SUITE_PLC_SERIAL_STOPBITS", "1").strip() or "1")
    plc_timeout_ms: int = max(100, int(os.getenv("QC_SUITE_PLC_TIMEOUT_MS", "1000")))
    plc_clamp_hold_ms: int = max(0, int(os.getenv("QC_SUITE_PLC_CLAMP_HOLD_MS", "2000")))
    # Modbus TCP mapping for clamp control.
    # Defaults are placeholders until the remote I/O datasheet is finalized.
    plc_modbus_unit_id: int = max(0, int(os.getenv("QC_SUITE_PLC_MODBUS_UNIT_ID", "1")))
    plc_modbus_command_mode: str = os.getenv("QC_SUITE_PLC_MODBUS_COMMAND_MODE", "coil").strip().lower() or "coil"
    plc_modbus_hold_address: int = max(0, int(os.getenv("QC_SUITE_PLC_MODBUS_HOLD_ADDRESS", "0")))
    plc_modbus_release_address: int = max(0, int(os.getenv("QC_SUITE_PLC_MODBUS_RELEASE_ADDRESS", "0")))
    plc_modbus_hold_value: int = max(0, int(os.getenv("QC_SUITE_PLC_MODBUS_HOLD_VALUE", "1")))
    plc_modbus_release_value: int = max(0, int(os.getenv("QC_SUITE_PLC_MODBUS_RELEASE_VALUE", "0")))
    plc_modbus_zero_based_addressing: bool = os.getenv("QC_SUITE_PLC_MODBUS_ZERO_BASED_ADDRESSING", "1").strip() != "0"
    plc_modbus_readback_enabled: bool = os.getenv("QC_SUITE_PLC_MODBUS_READBACK_ENABLED", "0").strip() == "1"
    plc_modbus_readback_mode: str = os.getenv("QC_SUITE_PLC_MODBUS_READBACK_MODE", "discrete_input").strip().lower() or "discrete_input"
    plc_modbus_readback_address: int = max(0, int(os.getenv("QC_SUITE_PLC_MODBUS_READBACK_ADDRESS", "0")))
    plc_modbus_readback_expected_hold_value: int = max(
        0,
        int(os.getenv("QC_SUITE_PLC_MODBUS_READBACK_EXPECTED_HOLD_VALUE", "1")),
    )
    plc_modbus_readback_expected_release_value: int = max(
        0,
        int(os.getenv("QC_SUITE_PLC_MODBUS_READBACK_EXPECTED_RELEASE_VALUE", "0")),
    )

    # ── 4-Relay + 2-Input Modbus mapping ──
    # Coil addresses (FC05/FC15) — 4 output relays
    # CH1=Enji Buzzer, CH2=OK Light+Buzzer, CH3=Clamp, CH4=Spare
    plc_relay_clamp_address: int = max(0, int(os.getenv("QC_SUITE_PLC_RELAY_CLAMP_ADDRESS", "3")))
    plc_relay_ok_light_buzzer_address: int = max(0, int(os.getenv("QC_SUITE_PLC_RELAY_OK_LIGHT_BUZZER_ADDRESS", "2")))
    plc_relay_enji_buzzer_address: int = max(0, int(os.getenv("QC_SUITE_PLC_RELAY_ENJI_BUZZER_ADDRESS", "1")))
    plc_relay_spare_address: int = max(0, int(os.getenv("QC_SUITE_PLC_RELAY_SPARE_ADDRESS", "0")))
    # Input addresses (FC02) — release, template cycle, optional clamp feedback
    plc_input_release_address: int = max(0, int(os.getenv("QC_SUITE_PLC_INPUT_RELEASE_ADDRESS", "0")))
    plc_input_template_address: int = max(0, int(os.getenv("QC_SUITE_PLC_INPUT_TEMPLATE_ADDRESS", "1")))
    plc_clamp_feedback_enabled: bool = os.getenv("QC_SUITE_PLC_CLAMP_FEEDBACK_ENABLED", "0").strip() == "1"
    plc_input_clamp_engaged_address: int = max(0, int(os.getenv("QC_SUITE_PLC_INPUT_CLAMP_ENGAGED_ADDRESS", "2")))
    plc_clamp_feedback_timeout_ms: int = max(0, int(os.getenv("QC_SUITE_PLC_CLAMP_FEEDBACK_TIMEOUT_MS", "1500")))
    plc_clamp_feedback_fallback_delay_ms: int = max(0, int(os.getenv("QC_SUITE_PLC_CLAMP_FEEDBACK_FALLBACK_DELAY_MS", "300")))
    # Accept pulse duration (ms)
    plc_accept_pulse_ms: int = max(100, int(os.getenv("QC_SUITE_PLC_ACCEPT_PULSE_MS", "1000")))
    # Camera default rotation (degrees) applied to all templates unless overridden
    camera_default_rotation_degrees: float = float(os.getenv("QC_SUITE_CAMERA_DEFAULT_ROTATION_DEGREES", "0"))
    # PLC guard: minimum interval between clamp ON after any release/accept/manual (ms)
    # Prevents rapid clamp cycling. Default 3000ms.
    plc_min_reclamp_interval_ms: int = max(0, int(os.getenv("QC_SUITE_PLC_MIN_RECLAMP_INTERVAL_MS", "3000")))
    # PLC guard: debounce for input release (ms)
    # Input release must be stable for this long before triggering all_off
    plc_release_input_debounce_ms: int = max(0, int(os.getenv("QC_SUITE_PLC_RELEASE_INPUT_DEBOUNCE_MS", "500")))
    # Session idle timeout (seconds): auto-end session after no frames received
    # for this duration. 0 = disabled (no auto-end). Default 300s (5 minutes).
    session_idle_timeout_s: int = max(0, int(os.getenv("QC_SUITE_SESSION_IDLE_TIMEOUT_S", "300")))

    def _has_sqlserver_credentials(self) -> bool:
        return bool(
            self.sql_server
            and self.sql_database
            and self.sql_username
            and self.sql_password
        )

    def _has_postgresql_credentials(self) -> bool:
        return bool(
            self.postgresql_host
            and self.postgresql_database
            and self.postgresql_username
            and self.postgresql_password
        )

    @property
    def database_backend(self) -> str:
        backend = self.relational_backend
        if backend in {"local", "sqlserver", "postgresql"}:
            if backend == "sqlserver":
                return backend if self._has_sqlserver_credentials() else "local"
            if backend == "postgresql":
                return backend if self._has_postgresql_credentials() else "local"
            return backend

        if self.sql_enabled_flag and self._has_sqlserver_credentials():
            return "sqlserver"
        if self._has_postgresql_credentials():
            return "postgresql"
        return "local"

    @property
    def sql_enabled(self) -> bool:
        return self.database_backend in {"sqlserver", "postgresql"}

    @property
    def postgresql_enabled(self) -> bool:
        return self.database_backend == "postgresql"


def ensure_data_dirs() -> None:
    for path in (DATA_ROOT, JSON_STORE_DIR, DATASETS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)

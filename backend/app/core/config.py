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
    access_token_ttl_seconds: int = max(60, int(os.getenv("QC_SUITE_ACCESS_TOKEN_TTL_SECONDS", "86400")))
    sql_server: str = os.getenv("MSSQL_SERVER", "").strip()
    sql_database: str = os.getenv("MSSQL_DATABASE", "").strip()
    sql_username: str = os.getenv("MSSQL_USERNAME", "").strip()
    sql_password: str = os.getenv("MSSQL_PASSWORD", "").strip()
    sql_driver: str = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
    sticker_inference_mode: str = os.getenv("QC_SUITE_STICKER_INFERENCE_MODE", "auto").strip().lower() or "auto"
    default_sticker_model_path: str = DEFAULT_STICKER_MODEL_PATH
    default_sticker_model_meta_path: str = DEFAULT_STICKER_MODEL_META_PATH

    @property
    def sql_enabled(self) -> bool:
        return bool(
            self.sql_server
            and self.sql_database
            and self.sql_username
            and self.sql_password
        )


def ensure_data_dirs() -> None:
    for path in (DATA_ROOT, JSON_STORE_DIR, DATASETS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)

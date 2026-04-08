#!/usr/bin/env python3
"""Bootstrap script — idempotent environment and data-directory setup.

Run this once before first launch, or after a fresh deploy, to ensure:
- All required data directories exist
- Default .env file is created if missing
- SQL Server schema is up-to-date (if MSSQL_* env vars are set)

Usage:
    py -3.11 scripts/bootstrap_env.py
    py -3.11 scripts/bootstrap_env.py --check   # dry-run: only verify, no changes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_TEMPLATE = """\
QC_SUITE_HOST=0.0.0.0
QC_SUITE_PORT=8100
QC_SUITE_DEBUG=0
QC_SUITE_SECRET_KEY=CHANGE_ME_IN_PRODUCTION
QC_SUITE_ACCESS_TOKEN_TTL_SECONDS=86400
QC_SUITE_DATA_ROOT=./data

QC_SUITE_DEVICE=auto
QC_SUITE_CUDA_DEVICE_ID=0

QC_SUITE_STICKER_INFERENCE_MODE=auto
QC_SUITE_DEFAULT_STICKER_MODEL_PATH=
QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH=

# SQL Server — leave blank to run in local/offline mode
MSSQL_SERVER=
MSSQL_DATABASE=
MSSQL_USERNAME=
MSSQL_PASSWORD=
MSSQL_DRIVER=ODBC Driver 18 for SQL Server

# Push worker tuning
QC_SUITE_PUSH_WORKER_INTERVAL_SECONDS=30
QC_SUITE_PUSH_WORKER_MAX_RETRY=5
"""


def _load_project_env(env_path: Path) -> None:
    if env_path.exists():
        load_dotenv(env_path, override=False)


def main(check: bool = False) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    # ── .env file ─────────────────────────────────────────────────────
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        if check:
            warnings.append(".env file missing — run bootstrap without --check to create it")
        else:
            env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
            print(f"[bootstrap] created {env_path}")
    else:
        print(f"[bootstrap] .env OK ({env_path})")

    # ── Data directories ───────────────────────────────────────────────
    _load_project_env(env_path)
    data_root_str = os.getenv("QC_SUITE_DATA_ROOT", str(PROJECT_ROOT / "data"))
    data_root = Path(data_root_str).resolve()
    required_dirs = [
        data_root,
        data_root / "json_store",
        data_root / "datasets",
        data_root / "models",
        data_root / "backups",
    ]
    for d in required_dirs:
        if not d.exists():
            if check:
                warnings.append(f"Directory missing: {d}")
            else:
                d.mkdir(parents=True, exist_ok=True)
                print(f"[bootstrap] created {d}")
        else:
            print(f"[bootstrap] dir OK: {d}")

    # ── Secret key check ───────────────────────────────────────────────
    secret = os.getenv("QC_SUITE_SECRET_KEY", "")
    if not secret or secret in {"qc-suite-dev-secret", "CHANGE_ME_IN_PRODUCTION", "test"}:
        warnings.append("QC_SUITE_SECRET_KEY is set to an insecure default — change before production")

    # ── SQL Server schema migration ────────────────────────────────────
    sql_vars = ["MSSQL_SERVER", "MSSQL_DATABASE", "MSSQL_USERNAME", "MSSQL_PASSWORD"]
    if all(os.getenv(v) for v in sql_vars):
        print("[bootstrap] SQL Server env detected — ensuring schema …")
        if not check:
            try:
                from backend.app.core.config import AppConfig
                from backend.app.repositories.sqlserver.auth_audit_repository import SqlServerAuthAuditRepository
                from backend.app.repositories.sqlserver.inspection_mirror_repository import SqlServerInspectionMirrorRepository
                from backend.app.repositories.sqlserver.session_store import SqlServerTokenStore
                from backend.app.repositories.sqlserver.users_repository import SqlServerUsersRepository
                cfg = AppConfig()
                SqlServerUsersRepository(cfg)
                print("[bootstrap]   dbo.qc_user_accounts ✓")
                SqlServerTokenStore(cfg)
                print("[bootstrap]   dbo.qc_user_sessions ✓")
                SqlServerInspectionMirrorRepository(cfg)
                print("[bootstrap]   dbo.qc_inspection_push ✓")
                SqlServerAuthAuditRepository(cfg)
                print("[bootstrap]   dbo.qc_auth_audit ✓")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"SQL Server schema migration failed: {exc}")
    else:
        print("[bootstrap] SQL Server env not set — running in local-only mode")

    # ── Report ─────────────────────────────────────────────────────────
    for w in warnings:
        print(f"[bootstrap] WARNING: {w}")
    for e in errors:
        print(f"[bootstrap] ERROR:   {e}", file=sys.stderr)

    if errors:
        print("[bootstrap] FAILED — fix errors above before starting the server")
        return 1
    print("[bootstrap] done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap qc-suite-python environment")
    parser.add_argument("--check", action="store_true", help="Dry-run: verify only, no changes")
    args = parser.parse_args()
    sys.exit(main(check=args.check))

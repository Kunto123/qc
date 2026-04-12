from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig
from backend.app.core.security import hash_password, verify_password
from shared.contracts.auth import UserInfo
from shared.contracts.enums import UserRole


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _seed_users() -> list[dict[str, Any]]:
    now = _utcnow_iso()
    return [
        {
            "id": 1,
            "username": "admin",
            "password_hash": hash_password("admin123"),
            "role": UserRole.ADMIN.value,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        },
        {
            "id": 2,
            "username": "operator",
            "password_hash": hash_password("operator123"),
            "role": UserRole.OPERATOR.value,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        },
        {
            "id": 3,
            "username": "engineer",
            "password_hash": hash_password("engineer123"),
            # Transitional account: keep legacy username but migrate role into supported 2-role policy.
            "role": UserRole.ADMIN.value,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        },
    ]


class SqlServerUsersRepository:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ensure_schema()
        self._seed_defaults_if_empty()
        self.migrate_legacy_roles()

    def migrate_legacy_roles(self) -> int:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE dbo.qc_user_accounts
                SET role = ?, updated_at = ?
                WHERE LOWER(role) = ?
                """,
                UserRole.ADMIN.value,
                now,
                UserRole.ENGINEER.value,
            )
            updated = int(cursor.rowcount or 0)
            conn.commit()
        return max(0, updated)

    def _connect(self):
        import pyodbc

        connection_string = (
            f"DRIVER={{{self._config.sql_driver}}};"
            f"SERVER={self._config.sql_server};"
            f"DATABASE={self._config.sql_database};"
            f"UID={self._config.sql_username};"
            f"PWD={self._config.sql_password};"
            "TrustServerCertificate=yes;"
        )
        return pyodbc.connect(connection_string, timeout=5)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                IF OBJECT_ID('dbo.qc_user_accounts', 'U') IS NULL
                BEGIN
                    CREATE TABLE dbo.qc_user_accounts (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        username NVARCHAR(100) NOT NULL,
                        password_hash NVARCHAR(255) NOT NULL,
                        role NVARCHAR(32) NOT NULL,
                        is_active BIT NOT NULL CONSTRAINT DF_qc_user_accounts_is_active DEFAULT 1,
                        created_at DATETIMEOFFSET NOT NULL CONSTRAINT DF_qc_user_accounts_created_at DEFAULT SYSUTCDATETIME(),
                        updated_at DATETIMEOFFSET NOT NULL CONSTRAINT DF_qc_user_accounts_updated_at DEFAULT SYSUTCDATETIME(),
                        last_login_at DATETIMEOFFSET NULL
                    )
                END
                """
            )
            cursor.execute(
                """
                IF NOT EXISTS (
                    SELECT 1
                    FROM sys.indexes
                    WHERE name = 'UX_qc_user_accounts_username'
                      AND object_id = OBJECT_ID('dbo.qc_user_accounts')
                )
                BEGIN
                    CREATE UNIQUE INDEX UX_qc_user_accounts_username
                    ON dbo.qc_user_accounts (username)
                END
                """
            )
            for column_name, sql_type in (
                ("updated_at", "DATETIMEOFFSET NOT NULL CONSTRAINT DF_qc_user_accounts_updated_at DEFAULT SYSUTCDATETIME()"),
                ("last_login_at", "DATETIMEOFFSET NULL"),
            ):
                cursor.execute(
                    f"""
                    IF COL_LENGTH('dbo.qc_user_accounts', '{column_name}') IS NULL
                    BEGIN
                        ALTER TABLE dbo.qc_user_accounts ADD {column_name} {sql_type}
                    END
                    """
                )
            conn.commit()

    def _seed_defaults_if_empty(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(1) FROM dbo.qc_user_accounts")
            count = int(cursor.fetchone()[0] or 0)
            if count > 0:
                return
            for record in _seed_users():
                cursor.execute(
                    """
                    INSERT INTO dbo.qc_user_accounts (
                        username,
                        password_hash,
                        role,
                        is_active,
                        created_at,
                        updated_at,
                        last_login_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    record["username"],
                    record["password_hash"],
                    record["role"],
                    1 if record["is_active"] else 0,
                    record["created_at"],
                    record["updated_at"],
                    record["last_login_at"],
                )
            conn.commit()

    def _public_record(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        if not record:
            return None
        return {
            "id": int(record["id"]),
            "username": str(record["username"]),
            "role": str(record["role"]),
            "is_active": bool(record.get("is_active", True)),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "last_login_at": record.get("last_login_at"),
        }

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    role,
                    is_active,
                    CONVERT(NVARCHAR(33), created_at, 127) AS created_at,
                    CONVERT(NVARCHAR(33), updated_at, 127) AS updated_at,
                    CONVERT(NVARCHAR(33), last_login_at, 127) AS last_login_at
                FROM dbo.qc_user_accounts
                ORDER BY id ASC
                """
            )
            rows = cursor.fetchall()
        return [self._public_record(self._row_to_dict(row)) for row in rows]

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        normalized = username.strip().lower()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT TOP 1
                    id,
                    username,
                    password_hash,
                    role,
                    is_active,
                    CONVERT(NVARCHAR(33), created_at, 127) AS created_at,
                    CONVERT(NVARCHAR(33), updated_at, 127) AS updated_at,
                    CONVERT(NVARCHAR(33), last_login_at, 127) AS last_login_at
                FROM dbo.qc_user_accounts
                WHERE LOWER(username) = ?
                """,
                normalized,
            )
            row = cursor.fetchone()
        return None if row is None else self._row_to_dict(row)

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT TOP 1
                    id,
                    username,
                    password_hash,
                    role,
                    is_active,
                    CONVERT(NVARCHAR(33), created_at, 127) AS created_at,
                    CONVERT(NVARCHAR(33), updated_at, 127) AS updated_at,
                    CONVERT(NVARCHAR(33), last_login_at, 127) AS last_login_at
                FROM dbo.qc_user_accounts
                WHERE id = ?
                """,
                int(user_id),
            )
            row = cursor.fetchone()
        return None if row is None else self._row_to_dict(row)

    def to_user_info(self, record: dict[str, Any] | None) -> UserInfo | None:
        if not record:
            return None
        return UserInfo(
            id=int(record["id"]),
            username=str(record["username"]),
            role=UserRole(str(record["role"])),
            is_active=bool(record.get("is_active", True)),
        )

    def get_user_info(self, user_id: int) -> UserInfo | None:
        return self.to_user_info(self.get_by_id(user_id))

    def authenticate(self, username: str, password: str) -> UserInfo | None:
        record = self.get_by_username(username)
        if not record or not record.get("is_active"):
            return None
        if not verify_password(password, str(record.get("password_hash") or "")):
            return None
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE dbo.qc_user_accounts
                SET last_login_at = ?, updated_at = ?
                WHERE id = ?
                """,
                now,
                now,
                int(record["id"]),
            )
            conn.commit()
        record["last_login_at"] = now
        record["updated_at"] = now
        return self.to_user_info(record)

    def create_user(self, username: str, password: str, role: str) -> dict[str, Any]:
        normalized_username = username.strip()
        if not normalized_username:
            raise ValueError("Username is required.")
        if self.get_by_username(normalized_username):
            raise ValueError("Username already exists.")
        role_enum = UserRole(role)
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dbo.qc_user_accounts (
                    username,
                    password_hash,
                    role,
                    is_active,
                    created_at,
                    updated_at
                )
                OUTPUT INSERTED.id
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                normalized_username,
                hash_password(password),
                role_enum.value,
                1,
                now,
                now,
            )
            inserted_id = int(cursor.fetchone()[0])
            conn.commit()
        return {
            "id": inserted_id,
            "username": normalized_username,
            "role": role_enum.value,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        }

    def set_active(self, user_id: int, is_active: bool) -> dict[str, Any]:
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE dbo.qc_user_accounts
                SET is_active = ?, updated_at = ?
                WHERE id = ?
                """,
                1 if is_active else 0,
                now,
                int(user_id),
            )
            conn.commit()
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)

    def set_role(self, user_id: int, role: str) -> dict[str, Any]:
        role_enum = UserRole(role)
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE dbo.qc_user_accounts
                SET role = ?, updated_at = ?
                WHERE id = ?
                """,
                role_enum.value,
                now,
                int(user_id),
            )
            conn.commit()
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)

    def set_password(self, user_id: int, new_password: str) -> dict[str, Any]:
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE dbo.qc_user_accounts
                SET password_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                hash_password(new_password),
                now,
                int(user_id),
            )
            conn.commit()
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)

    def _row_to_dict(self, row) -> dict[str, Any]:
        return {
            "id": int(row.id),
            "username": row.username,
            "password_hash": row.password_hash,
            "role": row.role,
            "is_active": bool(row.is_active),
            "created_at": str(row.created_at) if row.created_at is not None else None,
            "updated_at": str(row.updated_at) if row.updated_at is not None else None,
            "last_login_at": str(row.last_login_at) if row.last_login_at is not None else None,
        }

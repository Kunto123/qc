from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.security import hash_password, verify_password
from backend.app.repositories.base_json import JsonRepository
from shared.contracts.auth import UserInfo
from shared.contracts.enums import UserRole


def _seed_users() -> list[dict[str, Any]]:
    users = [
        (1, "admin", "admin123", UserRole.ADMIN),
        (2, "operator", "operator123", UserRole.OPERATOR),
        (3, "engineer", "engineer123", UserRole.ENGINEER),
    ]
    now = datetime.now(UTC).isoformat()
    return [
        {
            "id": user_id,
            "username": username,
            "password_hash": hash_password(password),
            "role": role.value,
            "is_active": True,
            "created_at": now,
        }
        for user_id, username, password, role in users
    ]


class UsersRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("users.json", _seed_users())

    def list_users(self) -> list[dict[str, Any]]:
        return self.load()

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        normalized = username.strip().lower()
        return next(
            (item for item in self.load() if item["username"].lower() == normalized),
            None,
        )

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        return next((item for item in self.load() if int(item["id"]) == int(user_id)), None)

    def authenticate(self, username: str, password: str) -> UserInfo | None:
        record = self.get_by_username(username)
        if not record or not record.get("is_active"):
            return None
        if not verify_password(password, str(record.get("password_hash") or "")):
            return None
        return UserInfo(
            id=int(record["id"]),
            username=str(record["username"]),
            role=UserRole(str(record["role"])),
            is_active=bool(record.get("is_active", True)),
        )

    def create_user(self, username: str, password: str, role: str) -> dict[str, Any]:
        users = self.load()
        if self.get_by_username(username):
            raise ValueError("Username already exists.")
        role_enum = UserRole(role)
        record = {
            "id": self.next_id(users),
            "username": username.strip(),
            "password_hash": hash_password(password),
            "role": role_enum.value,
            "is_active": True,
            "created_at": datetime.now(UTC).isoformat(),
        }
        users.append(record)
        self.save(users)
        return record

    def set_active(self, user_id: int, is_active: bool) -> dict[str, Any]:
        users = self.load()
        for item in users:
            if int(item["id"]) == int(user_id):
                item["is_active"] = bool(is_active)
                self.save(users)
                return item
        raise ValueError("User not found.")


from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass

from shared.contracts.auth import UserInfo


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _ = stored_hash.split("$", 1)
    except ValueError:
        return False
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


@dataclass(slots=True)
class TokenRecord:
    token: str
    user: UserInfo


class TokenStore:
    def __init__(self) -> None:
        self._items: dict[str, TokenRecord] = {}
        self._lock = threading.Lock()

    def issue(self, user: UserInfo) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._items[token] = TokenRecord(token=token, user=user)
        return token

    def get(self, token: str) -> UserInfo | None:
        with self._lock:
            record = self._items.get(token)
        return record.user if record else None

    def revoke(self, token: str) -> None:
        with self._lock:
            self._items.pop(token, None)


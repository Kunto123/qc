from __future__ import annotations

from functools import wraps
from typing import Callable

from flask import g, jsonify, request

from backend.app.core.container import token_store
from shared.contracts.enums import UserRole


def current_user():
    auth = str(request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    else:
        token = str(request.headers.get("X-Auth-Token") or "").strip()
    if not token:
        return None
    return token_store.get(token)


def require_auth(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return jsonify({"error": "Unauthorized"}), 401
        g.current_user = user
        return fn(*args, **kwargs)

    return wrapper


def require_roles(*allowed_roles: UserRole):
    def decorator(fn: Callable):
        @wraps(fn)
        @require_auth
        def wrapper(*args, **kwargs):
            user = g.current_user
            if user.role not in allowed_roles:
                return jsonify({"error": "Forbidden"}), 403
            return fn(*args, **kwargs)

        return wrapper

    return decorator


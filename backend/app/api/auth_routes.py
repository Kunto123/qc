from __future__ import annotations

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import token_store, users_repo
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import UserRole


auth_blueprint = Blueprint("auth", __name__, url_prefix="/auth")


@auth_blueprint.post("/login")
def login():
    payload = request.get_json(force=True) or {}
    user = users_repo.authenticate(
        str(payload.get("username") or ""),
        str(payload.get("password") or ""),
    )
    if user is None:
        return jsonify({"error": "Invalid credentials"}), 401
    token = token_store.issue(user)
    return jsonify({"token": token, "user": user.to_dict(), "expires_in": 86400})


@auth_blueprint.get("/me")
@require_auth
def me():
    return jsonify(g.current_user.to_dict())


@auth_blueprint.get("/users")
@require_roles(UserRole.ADMIN)
def list_users():
    return jsonify(users_repo.list_users())


@auth_blueprint.post("/users")
@require_roles(UserRole.ADMIN)
def create_user():
    payload = request.get_json(force=True) or {}
    try:
        record = users_repo.create_user(
            username=str(payload.get("username") or "").strip(),
            password=str(payload.get("password") or "").strip(),
            role=str(payload.get("role") or UserRole.OPERATOR.value).strip(),
        )
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(record), 201


@auth_blueprint.put("/users/<int:user_id>")
@require_roles(UserRole.ADMIN)
def update_user(user_id: int):
    payload = request.get_json(force=True) or {}
    try:
        record = users_repo.set_active(user_id, bool(payload.get("is_active", True)))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(record)


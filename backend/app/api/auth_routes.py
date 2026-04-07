from __future__ import annotations

import sys

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import audit_repo, token_store, users_repo
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import UserRole


auth_blueprint = Blueprint("auth", __name__, url_prefix="/auth")


def _client_ip() -> str:
    forwarded_for = str(request.headers.get("X-Forwarded-For") or "").strip()
    return forwarded_for.split(",", 1)[0].strip() if forwarded_for else (request.remote_addr or "")


def _client_name(payload: dict) -> str | None:
    value = str(payload.get("client_name") or request.headers.get("X-Client-Name") or "").strip()
    return value or None


def _try_audit(event_type: str, **kwargs) -> None:
    """Fire-and-forget audit log. Never raises so auth ops are not blocked."""
    try:
        audit_repo.log(event_type, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] WARNING: failed to write audit event '{event_type}': {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@auth_blueprint.post("/login")
def login():
    payload = request.get_json(force=True) or {}
    username_input = str(payload.get("username") or "").strip()
    ip = _client_ip()
    client = _client_name(payload)
    ua = request.user_agent.string or None

    user = users_repo.authenticate(username_input, str(payload.get("password") or ""))
    if user is None:
        _try_audit(
            "login_failure",
            username=username_input,
            ip_address=ip,
            client_name=client,
            details="Invalid credentials",
        )
        return jsonify({"error": "Invalid credentials"}), 401

    session = token_store.issue(user, ip_address=ip, client_name=client, user_agent=ua)
    _try_audit(
        "login_success",
        user_id=user.id,
        username=user.username,
        session_id=session.session_id,
        ip_address=ip,
        client_name=client,
    )
    return jsonify(
        {
            "token": session.token,
            "user": user.to_dict(),
            "expires_in": session.remaining_seconds(now=session.issued_at),
            "expires_at": session.expires_at.isoformat(),
            "session": session.to_dict(),
        }
    )


@auth_blueprint.get("/me")
@require_auth
def me():
    return jsonify(g.current_user.to_dict())


@auth_blueprint.post("/logout")
@require_auth
def logout():
    token = getattr(g, "current_token", "")
    session = getattr(g, "current_session", None)
    revoked = token_store.revoke(token)
    _try_audit(
        "logout",
        user_id=g.current_user.id,
        username=g.current_user.username,
        session_id=session.session_id if session else None,
        ip_address=_client_ip(),
    )
    return jsonify({"revoked": revoked})


@auth_blueprint.post("/logout-all")
@require_auth
def logout_all():
    revoked = token_store.revoke_user(g.current_user.id)
    _try_audit(
        "logout_all",
        user_id=g.current_user.id,
        username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"revoked {revoked} session(s)",
    )
    return jsonify({"revoked": revoked, "user_id": g.current_user.id})


@auth_blueprint.get("/sessions")
@require_auth
def list_sessions():
    current_session = getattr(g, "current_session", None)
    items = []
    for session in token_store.list_user_sessions(g.current_user.id):
        record = session.to_dict()
        record["is_current"] = bool(current_session and session.session_id == current_session.session_id)
        items.append(record)
    return jsonify(items)


# ---------------------------------------------------------------------------
# User management (admin only)
# ---------------------------------------------------------------------------

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
    _try_audit(
        "user_created",
        user_id=record.get("id"),
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"role={record.get('role')}",
    )
    return jsonify(record), 201


@auth_blueprint.put("/users/<int:user_id>")
@require_roles(UserRole.ADMIN)
def update_user(user_id: int):
    payload = request.get_json(force=True) or {}
    is_active = bool(payload.get("is_active", True))
    try:
        record = users_repo.set_active(user_id, is_active)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    if not is_active:
        token_store.revoke_user(user_id)
    _try_audit(
        "user_disabled" if not is_active else "user_enabled",
        user_id=user_id,
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
    )
    return jsonify(record)


@auth_blueprint.post("/users/<int:user_id>/role")
@require_roles(UserRole.ADMIN)
def change_user_role(user_id: int):
    payload = request.get_json(force=True) or {}
    new_role = str(payload.get("role") or "").strip()
    if not new_role:
        return jsonify({"error": "role is required"}), 400
    try:
        record = users_repo.set_role(user_id, new_role)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    _try_audit(
        "role_changed",
        user_id=user_id,
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"new_role={new_role}",
    )
    return jsonify(record)


@auth_blueprint.post("/users/<int:user_id>/reset-password")
@require_roles(UserRole.ADMIN)
def reset_user_password(user_id: int):
    payload = request.get_json(force=True) or {}
    new_password = str(payload.get("password") or "").strip()
    if len(new_password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    try:
        record = users_repo.set_password(user_id, new_password)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    # Revoke existing sessions so the user must re-authenticate
    token_store.revoke_user(user_id)
    _try_audit(
        "password_reset",
        user_id=user_id,
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
    )
    return jsonify({"ok": True, "user_id": user_id, "sessions_revoked": True})


@auth_blueprint.post("/users/<int:user_id>/revoke-sessions")
@require_roles(UserRole.ADMIN)
def revoke_user_sessions(user_id: int):
    target = users_repo.get_by_id(user_id)
    if target is None:
        return jsonify({"error": "User not found."}), 404
    revoked = token_store.revoke_user(user_id)
    _try_audit(
        "session_revoked",
        user_id=user_id,
        username=target.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"revoked {revoked} session(s)",
    )
    return jsonify({"user_id": user_id, "revoked": revoked})


# ---------------------------------------------------------------------------
# Audit log (admin only)
# ---------------------------------------------------------------------------

@auth_blueprint.get("/audit-log")
@require_roles(UserRole.ADMIN)
def get_audit_log():
    """Return recent auth audit events.

    Query params:
    - ``limit``   max entries to return (default 100, max 500)
    - ``user_id`` optional filter by user
    """
    try:
        limit = min(500, max(1, int(request.args.get("limit", 100))))
    except (ValueError, TypeError):
        limit = 100
    user_id_raw = request.args.get("user_id")
    user_id: int | None = None
    if user_id_raw is not None:
        try:
            user_id = int(user_id_raw)
        except (ValueError, TypeError):
            return jsonify({"error": "user_id must be an integer"}), 400
    return jsonify(audit_repo.list_recent(limit=limit, user_id=user_id))

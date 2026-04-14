from __future__ import annotations

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import templates_repo
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import UserRole


template_blueprint = Blueprint("templates", __name__, url_prefix="/templates")


@template_blueprint.get("")
@require_auth
def list_templates():
    return jsonify(templates_repo.list_summaries())


@template_blueprint.get("/<int:template_id>")
@require_auth
def get_template(template_id: int):
    detail = templates_repo.get_template_detail(template_id)
    if detail is None:
        return jsonify({"error": "Template not found"}), 404
    return jsonify(detail)


@template_blueprint.get("/versions/<int:version_id>")
@require_auth
def get_template_version(version_id: int):
    detail = templates_repo.get_version_detail(version_id)
    if detail is None:
        return jsonify({"error": "Template version not found"}), 404
    return jsonify(detail)


@template_blueprint.post("")
@require_roles(UserRole.ADMIN)
def create_template():
    payload = request.get_json(force=True) or {}
    try:
        record = templates_repo.create_template(payload)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(record), 201


@template_blueprint.put("/<int:template_id>")
@require_roles(UserRole.ADMIN)
def update_template(template_id: int):
    payload = request.get_json(force=True) or {}
    try:
        record = templates_repo.update_template(template_id, payload)
    except TypeError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(record)


@template_blueprint.delete("/<int:template_id>")
@require_roles(UserRole.ADMIN)
def delete_template(template_id: int):
    ok = templates_repo.delete_template(template_id)
    if not ok:
        return jsonify({"error": "Template not found"}), 404
    return jsonify({"deleted": True, "id": template_id})


@template_blueprint.get("/<int:template_id>/versions")
@require_auth
def list_template_versions(template_id: int):
    try:
        versions = templates_repo.list_versions(template_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(versions)


@template_blueprint.post("/<int:template_id>/transition")
@require_roles(UserRole.ADMIN)
def transition_template_lifecycle(template_id: int):
    payload = request.get_json(force=True) or {}
    new_status = str(payload.get("status") or "").strip().lower()
    if not new_status:
        return jsonify({"error": "status is required"}), 400
    change_note = str(payload.get("change_note") or "").strip() or None
    actor = getattr(g, "current_user", None)
    try:
        result = templates_repo.transition_lifecycle(
            template_id,
            new_status,
            actor_id=actor.id if actor else None,
            actor_username=actor.username if actor else None,
            change_note=change_note,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)


@template_blueprint.post("/<int:template_id>/rollback")
@require_roles(UserRole.ADMIN)
def rollback_template_version(template_id: int):
    payload = request.get_json(force=True) or {}
    version_id = payload.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
    try:
        result = templates_repo.rollback_version(template_id, int(version_id))
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)

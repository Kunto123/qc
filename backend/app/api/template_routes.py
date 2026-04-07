from __future__ import annotations

from flask import Blueprint, jsonify, request

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


@template_blueprint.post("")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_template():
    payload = request.get_json(force=True) or {}
    try:
        record = templates_repo.create_template(payload)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(record), 201


@template_blueprint.put("/<int:template_id>")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
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
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def delete_template(template_id: int):
    ok = templates_repo.delete_template(template_id)
    if not ok:
        return jsonify({"error": "Template not found"}), 404
    return jsonify({"deleted": True, "id": template_id})

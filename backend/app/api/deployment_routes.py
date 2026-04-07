from __future__ import annotations

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import deployments_repo, templates_repo
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import UserRole


deployment_blueprint = Blueprint("deployments", __name__, url_prefix="/deployments")


@deployment_blueprint.post("")
@require_roles(UserRole.ADMIN)
def deploy_template():
    payload = request.get_json(force=True) or {}
    template_id = int(payload.get("template_id") or 0)
    template_version_id = int(payload.get("template_version_id") or 0)
    line_id = str(payload.get("line_id") or "").strip()
    station_id = str(payload.get("station_id") or "").strip()
    if not template_id or not template_version_id or not line_id or not station_id:
        return jsonify({"error": "template_id, template_version_id, line_id, dan station_id wajib diisi"}), 400
    template = templates_repo.get_template(template_id)
    version = templates_repo.get_version(template_version_id)
    if template is None or version is None:
        return jsonify({"error": "Template or version not found"}), 404
    record = deployments_repo.deploy(
        template_id=template_id,
        template_version_id=template_version_id,
        line_id=line_id,
        station_id=station_id,
        deployed_by=None,
        template_name=template["name"],
        version_number=int(version["version_number"]),
    )
    return jsonify(record), 201


@deployment_blueprint.get("")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def list_deployments():
    return jsonify(deployments_repo.list_deployments())


@deployment_blueprint.get("/active")
@require_auth
def get_active_deployment():
    line_id = str(request.args.get("line_id") or "").strip()
    station_id = str(request.args.get("station_id") or "").strip()
    if not line_id or not station_id:
        return jsonify({"error": "line_id and station_id are required"}), 400
    record = deployments_repo.get_active(line_id, station_id)
    return jsonify({"deployment": record})


@deployment_blueprint.delete("/<int:deployment_id>")
@require_roles(UserRole.ADMIN)
def deactivate_deployment(deployment_id: int):
    ok = deployments_repo.deactivate(deployment_id)
    if not ok:
        return jsonify({"error": "Deployment not found"}), 404
    return jsonify({"id": deployment_id, "is_active": False})


@deployment_blueprint.post("/<int:deployment_id>/rollback")
@require_roles(UserRole.ADMIN)
def rollback_deployment(deployment_id: int):
    actor = getattr(g, "current_user", None)
    try:
        record = deployments_repo.rollback(
            deployment_id,
            rolled_back_by=actor.id if actor else None,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(record), 201

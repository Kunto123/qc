from __future__ import annotations

import csv
import io

from flask import Blueprint, Response, g, jsonify, request

from backend.app.core.container import inspection_results_repo, inspection_session_service
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import UserRole


inspection_blueprint = Blueprint("inspection", __name__)


@inspection_blueprint.post("/inspection/sessions/start")
@require_auth
def start_session():
    payload = request.get_json(force=True) or {}
    try:
        session = inspection_session_service.start_session(
            client_id=str(payload.get("client_id") or "").strip() or str(g.current_user.id),
            camera_index=int(payload.get("camera_index") or 0),
            template_version_id=int(payload.get("template_version_id") or 0),
            line_id=str(payload.get("line_id") or "").strip() or None,
            station_id=str(payload.get("station_id") or "").strip() or None,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(session), 201


@inspection_blueprint.post("/inspection/sessions/<session_id>/frame")
@require_auth
def push_frame(session_id: str):
    payload = request.get_json(force=True) or {}
    try:
        result = inspection_session_service.process_frame(
            session_id,
            image_b64=str(payload.get("image_b64") or ""),
            username=g.current_user.username,
            user_id=g.current_user.id,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@inspection_blueprint.post("/inspection/sessions/<session_id>/roi")
@require_auth
def update_roi(session_id: str):
    payload = request.get_json(force=True) or {}
    try:
        result = inspection_session_service.update_roi(session_id, payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@inspection_blueprint.post("/inspection/sessions/<session_id>/stop")
@require_auth
def stop_session(session_id: str):
    try:
        result = inspection_session_service.stop_session(session_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@inspection_blueprint.get("/inspection/latest-preview")
@require_auth
def latest_preview():
    result = inspection_session_service.get_latest_preview()
    if result is None:
        return jsonify({"error": "No active session with frames available."}), 404
    return jsonify(result)


@inspection_blueprint.get("/inspections")
@require_auth
def list_inspections():
    items = inspection_results_repo.list_results(
        line_id=request.args.get("line_id") or None,
        station_id=request.args.get("station_id") or None,
        part_name=request.args.get("part_name") or None,
        template_version_id=int(request.args["template_version_id"]) if request.args.get("template_version_id") else None,
        decision_code=request.args.get("decision_code") or None,
        push_status=request.args.get("push_status") or None,
        limit=min(int(request.args.get("limit") or 100), 1000),
        offset=int(request.args.get("offset") or 0),
    )
    return jsonify(items)


@inspection_blueprint.get("/inspections/<int:result_id>")
@require_auth
def get_inspection(result_id: int):
    item = inspection_results_repo.get_result(result_id)
    if item is None:
        return jsonify({"error": "Inspection result not found"}), 404
    return jsonify(item)


@inspection_blueprint.post("/inspections/<int:result_id>/retry-push")
@require_roles(UserRole.ADMIN)
def retry_inspection_push(result_id: int):
    try:
        item = inspection_results_repo.retry_result(result_id)
    except ValueError as exc:
        message = str(exc)
        status = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status
    return jsonify({"ok": item.get("push_status") == "sent", "result": item})


@inspection_blueprint.post("/inspections/retry-push")
@require_roles(UserRole.ADMIN)
def retry_failed_inspection_pushes():
    payload = request.get_json(silent=True) or {}
    raw_ids = payload.get("result_ids") or []
    result_ids = []
    if isinstance(raw_ids, list):
        for value in raw_ids:
            try:
                result_ids.append(int(value))
            except (TypeError, ValueError):
                continue
    limit = min(max(int(payload.get("limit") or 100), 1), 500)
    try:
        items = inspection_results_repo.retry_failed(result_ids=result_ids or None, limit=limit)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    succeeded = sum(1 for item in items if item.get("push_status") == "sent")
    return jsonify(
        {
            "attempted": len(items),
            "succeeded": succeeded,
            "failed": len(items) - succeeded,
            "items": items,
        }
    )


@inspection_blueprint.get("/inspections/export")
@require_auth
def export_inspections():
    items = inspection_results_repo.list_results(
        line_id=request.args.get("line_id") or None,
        station_id=request.args.get("station_id") or None,
        part_name=request.args.get("part_name") or None,
        template_version_id=int(request.args["template_version_id"]) if request.args.get("template_version_id") else None,
        decision_code=request.args.get("decision_code") or None,
        limit=10000,
        offset=0,
    )
    fields = [
        "id", "inspected_at", "line_id", "station_id", "part_name",
        "decision_code", "reject_reason_code",
        "part_ready_match_ratio", "sticker_confidence",
        "detected_class", "expected_class",
        "template_version_id", "push_status",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(items)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=inspections.csv"},
    )

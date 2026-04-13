from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.app.core.container import profiles_repo
from backend.app.core.http import require_auth, require_roles
from backend.app.services.calibration import CalibrationService
from shared.contracts.enums import UserRole


calibration_blueprint = Blueprint("calibration", __name__, url_prefix="/calibration")


@calibration_blueprint.post("/color-profile")
@require_roles(UserRole.ADMIN)
def compute_color_profile():
    payload = request.get_json(force=True) or {}
    try:
        image = CalibrationService.decode_image(str(payload.get("image_b64") or ""))
        roi = payload.get("roi")
        image = CalibrationService.apply_roi(image, roi)
        profile = CalibrationService.compute_color_profile(
            image,
            str(payload.get("colorspace") or "LAB"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(profile)


@calibration_blueprint.get("/profiles")
@require_auth
def list_profiles():
    return jsonify(profiles_repo.list_profiles())


@calibration_blueprint.post("/profiles")
@require_roles(UserRole.ADMIN)
def create_profile():
    payload = request.get_json(force=True) or {}
    expiry_raw = payload.get("expiry_interval_days")
    expiry_days: int | None = None
    if expiry_raw is not None:
        try:
            expiry_days = max(1, int(expiry_raw))
        except (ValueError, TypeError):
            return jsonify({"error": "expiry_interval_days must be a positive integer"}), 400
    record = profiles_repo.create(
        str(payload.get("name") or "").strip() or "Unnamed Profile",
        dict(payload.get("profile") or {}),
        scope_line_id=str(payload.get("scope_line_id") or "").strip() or None,
        scope_station_id=str(payload.get("scope_station_id") or "").strip() or None,
        scope_part_name=str(payload.get("scope_part_name") or "").strip() or None,
        expiry_interval_days=expiry_days,
    )
    return jsonify(record), 201


@calibration_blueprint.put("/profiles/<int:profile_id>")
@require_roles(UserRole.ADMIN)
def update_profile(profile_id: int):
    payload = request.get_json(force=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be an object"}), 400

    updates: dict = {}
    for field in ("name", "profile", "scope_line_id", "scope_station_id", "scope_part_name", "expiry_interval_days"):
        if field in payload:
            updates[field] = payload.get(field)

    if not updates:
        return jsonify({"error": "At least one field must be provided"}), 400

    try:
        record = profiles_repo.update(profile_id, **updates)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(record)


@calibration_blueprint.get("/profiles/active")
@require_auth
def get_active_profile():
    """Return the most recent non-expired profile for the given scope.

    Query params: line_id, station_id, part_name (all optional)
    """
    line_id = str(request.args.get("line_id") or "").strip() or None
    station_id = str(request.args.get("station_id") or "").strip() or None
    part_name = str(request.args.get("part_name") or "").strip() or None
    record = profiles_repo.get_active_for_scope(
        line_id=line_id,
        station_id=station_id,
        part_name=part_name,
    )
    if record is None:
        return jsonify({"error": "No active calibration profile found for the given scope"}), 404
    return jsonify(record)


@calibration_blueprint.delete("/profiles/<int:profile_id>")
@require_roles(UserRole.ADMIN)
def delete_profile(profile_id: int):
    ok = profiles_repo.delete(profile_id)
    if not ok:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify({"deleted": True, "id": profile_id})


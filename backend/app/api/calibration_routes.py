from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.app.core.container import profiles_repo
from backend.app.core.http import require_auth, require_roles
from backend.app.services.calibration import CalibrationService
from shared.contracts.enums import UserRole


calibration_blueprint = Blueprint("calibration", __name__, url_prefix="/calibration")


@calibration_blueprint.post("/color-profile")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
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
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_profile():
    payload = request.get_json(force=True) or {}
    record = profiles_repo.create(
        str(payload.get("name") or "").strip() or "Unnamed Profile",
        dict(payload.get("profile") or {}),
    )
    return jsonify(record), 201


@calibration_blueprint.delete("/profiles/<int:profile_id>")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def delete_profile(profile_id: int):
    ok = profiles_repo.delete(profile_id)
    if not ok:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify({"deleted": True, "id": profile_id})


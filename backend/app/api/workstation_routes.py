from __future__ import annotations

import base64
from datetime import UTC, datetime

from flask import Blueprint, jsonify, request

from backend.app.core.config import MODELS_DIR
from backend.app.core.container import datasets_repo, models_repo, training_service
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import UserRole


workstation_blueprint = Blueprint("workstation", __name__)


@workstation_blueprint.get("/datasets")
@require_auth
def list_datasets():
    return jsonify(datasets_repo.list_datasets())


@workstation_blueprint.post("/datasets")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_dataset():
    payload = request.get_json(force=True) or {}
    name = str(payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Dataset name is required"}), 400
    return jsonify(datasets_repo.create_dataset(name, str(payload.get("description") or ""))), 201


@workstation_blueprint.delete("/datasets/<dataset_id>")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def delete_dataset(dataset_id: str):
    ok = datasets_repo.delete_dataset(dataset_id)
    if not ok:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify({"deleted": True, "id": dataset_id})


@workstation_blueprint.get("/datasets/<dataset_id>/files")
@require_auth
def list_dataset_files(dataset_id: str):
    target = str(request.args.get("target") or "images")
    return jsonify(datasets_repo.list_files(dataset_id, target))


@workstation_blueprint.post("/datasets/<dataset_id>/upload")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def upload_dataset_file(dataset_id: str):
    payload = request.get_json(force=True) or {}
    file_name = str(payload.get("file_name") or "").strip()
    target = str(payload.get("target") or "images").strip()
    content_b64 = str(payload.get("content_b64") or "").strip()
    if not file_name or not content_b64:
        return jsonify({"error": "file_name and content_b64 are required"}), 400
    try:
        content = base64.b64decode(content_b64)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Invalid base64 content: {exc}"}), 400
    return jsonify(datasets_repo.save_file(dataset_id, target, file_name, content)), 201


@workstation_blueprint.get("/datasets/<dataset_id>/annotations/<image_name>")
@require_auth
def get_annotation(dataset_id: str, image_name: str):
    return jsonify(
        {
            "image_name": image_name,
            "labels": datasets_repo.get_annotation(dataset_id, image_name),
        }
    )


@workstation_blueprint.post("/datasets/<dataset_id>/annotations/<image_name>")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def save_annotation(dataset_id: str, image_name: str):
    payload = request.get_json(force=True) or {}
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400
    return jsonify(datasets_repo.save_annotation(dataset_id, image_name, labels))


@workstation_blueprint.get("/augment/jobs")
@require_auth
def list_augment_jobs():
    return jsonify(
        [
            {
                "id": "augment-demo-1",
                "dataset_id": "demo",
                "status": "completed",
                "created_at": datetime.now(UTC).isoformat(),
            }
        ]
    )


@workstation_blueprint.post("/augment/jobs")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_augment_job():
    payload = request.get_json(force=True) or {}
    return jsonify(
        {
            "id": f"augment-{datetime.now(UTC).strftime('%H%M%S')}",
            "dataset_id": payload.get("dataset_id"),
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "note": "Augmentation is scaffolded as a completed metadata job in this MVP.",
        }
    ), 201


@workstation_blueprint.get("/train/jobs")
@require_auth
def list_training_jobs():
    return jsonify(training_service.list_jobs())


@workstation_blueprint.post("/train/jobs")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_training_job():
    payload = request.get_json(force=True) or {}
    dataset_id = str(payload.get("dataset_id") or "").strip()
    base_model = str(payload.get("base_model") or "baseline").strip()
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    return jsonify(training_service.create_job(dataset_id, base_model, dict(payload))), 201


@workstation_blueprint.post("/train/jobs/<job_id>/cancel")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def cancel_training_job(job_id: str):
    try:
        result = training_service.cancel_job(job_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@workstation_blueprint.get("/models")
@require_auth
def list_models():
    return jsonify(models_repo.list_models())


@workstation_blueprint.post("/models/upload")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def upload_model():
    payload = request.get_json(force=True) or {}
    name = str(payload.get("name") or "").strip()
    file_name = str(payload.get("file_name") or "model.pt").strip()
    content_b64 = str(payload.get("content_b64") or "").strip()
    class_names_raw = payload.get("class_names") or []

    if not name or not content_b64:
        return jsonify({"error": "name and content_b64 are required"}), 400
    if not isinstance(class_names_raw, list):
        return jsonify({"error": "class_names must be a list"}), 400

    try:
        content = base64.b64decode(content_b64)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Invalid base64 content: {exc}"}), 400

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / file_name
    dest.write_bytes(content)

    model = models_repo.add_model(
        name,
        str(dest),
        "upload",
        runtime=str(payload.get("runtime") or "ultralytics").strip() or "ultralytics",
        task=str(payload.get("task") or "detection").strip() or "detection",
        class_names=list(class_names_raw),
        architecture_family=str(payload.get("architecture_family") or "").strip() or None,
        architecture_variant=str(payload.get("architecture_variant") or "").strip() or None,
    )
    return jsonify({**model, "saved_to": str(dest)}), 201


@workstation_blueprint.post("/models")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_model():
    payload = request.get_json(force=True) or {}
    name = str(payload.get("name") or "").strip()
    path = str(payload.get("path") or "").strip()
    if not name or not path:
        return jsonify({"error": "name and path are required"}), 400
    class_names = payload.get("class_names")
    if class_names is not None and not isinstance(class_names, list):
        return jsonify({"error": "class_names must be a list"}), 400
    return jsonify(
        models_repo.add_model(
            name,
            path,
            str(payload.get("source") or "manual"),
            meta_path=str(payload.get("meta_path") or "").strip() or None,
            runtime=str(payload.get("runtime") or "ultralytics").strip() or "ultralytics",
            task=str(payload.get("task") or "detection").strip() or "detection",
            class_names=list(class_names or []),
            architecture_family=str(payload.get("architecture_family") or "").strip() or None,
            architecture_variant=str(payload.get("architecture_variant") or "").strip() or None,
        )
    ), 201

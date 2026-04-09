from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path

from flask import Blueprint, g, jsonify, request, send_file

from backend.app.core.config import MODELS_DIR
from backend.app.core.container import (
    augment_repo,
    dataset_versions_repo,
    datasets_repo,
    models_repo,
    token_store,
    training_service,
    workstation_registry_repo,
)
from backend.app.core.model_catalog import list_base_models, resolve_base_model
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import UserRole


workstation_blueprint = Blueprint("workstation", __name__)


def _multipart_upload_files() -> list:
    files = request.files.getlist("files")
    if not files:
        files = request.files.getlist("files[]")
    if not files:
        files = request.files.getlist("file")
    return [item for item in files if getattr(item, "filename", "")]


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
    try:
        return jsonify(datasets_repo.list_files(dataset_id, target))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@workstation_blueprint.get("/datasets/<dataset_id>/files/<target>/<path:file_name>")
@require_auth
def download_dataset_file(dataset_id: str, target: str, file_name: str):
    allowed_targets = {"images", "labels", "exports"}
    if target not in allowed_targets:
        return jsonify({"error": f"Invalid dataset target '{target}'"}), 400
    file_path = datasets_repo.dataset_dir(dataset_id) / target / Path(file_name).name
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=False)


@workstation_blueprint.get("/datasets/<dataset_id>/versions")
@require_auth
def list_dataset_versions(dataset_id: str):
    if not datasets_repo.get_dataset(dataset_id):
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(dataset_versions_repo.list_versions(dataset_id))


@workstation_blueprint.post("/datasets/<dataset_id>/versions")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_dataset_version(dataset_id: str):
    payload = request.get_json(force=True) or {}
    try:
        return jsonify(dataset_versions_repo.create_version(dataset_id, payload)), 201
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code


@workstation_blueprint.get("/datasets/<dataset_id>/versions/<version_id>")
@require_auth
def get_dataset_version(dataset_id: str, version_id: str):
    version = dataset_versions_repo.get_version(version_id)
    if version is None or str(version.get("dataset_id")) != str(dataset_id):
        return jsonify({"error": "Dataset version not found"}), 404
    return jsonify(version)


@workstation_blueprint.post("/datasets/<dataset_id>/versions/<version_id>/export")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def export_dataset_version(dataset_id: str, version_id: str):
    try:
        return jsonify(dataset_versions_repo.export_version(dataset_id, version_id))
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code


@workstation_blueprint.post("/datasets/<dataset_id>/upload")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def upload_dataset_file(dataset_id: str):
    if request.files:
        target = str(request.form.get("target") or "images").strip()
        uploads = _multipart_upload_files()
        if not uploads:
            return jsonify({"error": "At least one file is required"}), 400
        batch: list[tuple[str, bytes]] = []
        for file_storage in uploads:
            file_name = str(file_storage.filename or "").strip()
            if not file_name:
                continue
            batch.append((file_name, file_storage.read()))
        if not batch:
            return jsonify({"error": "At least one file is required"}), 400
        try:
            saved = datasets_repo.save_files(dataset_id, target, batch)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"target": target, "count": len(saved), "items": saved}), 201

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
    try:
        item = datasets_repo.save_file(dataset_id, target, file_name, content)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"target": target, "count": 1, "items": [item]}), 201


@workstation_blueprint.get("/datasets/<dataset_id>/annotations/<image_name>")
@require_auth
def get_annotation(dataset_id: str, image_name: str):
    return jsonify(datasets_repo.get_annotation(dataset_id, image_name))


@workstation_blueprint.post("/datasets/<dataset_id>/annotations/<image_name>")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def save_annotation(dataset_id: str, image_name: str):
    payload = request.get_json(force=True)
    if payload is None:
        payload = {}
    if isinstance(payload, list):
        labels = payload
    elif isinstance(payload, dict):
        labels = payload.get("labels")
        if labels is None:
            labels = payload.get("annotations")
    else:
        labels = None
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400
    return jsonify(datasets_repo.save_annotation(dataset_id, image_name, labels))


@workstation_blueprint.get("/augment/jobs")
@require_auth
def list_augment_jobs():
    return jsonify(augment_repo.list_jobs())


@workstation_blueprint.get("/augment/jobs/<job_id>")
@require_auth
def get_augment_job(job_id: str):
    job = augment_repo.get_job(job_id)
    if job is None:
        return jsonify({"error": "Augment job not found"}), 404
    return jsonify(job)


@workstation_blueprint.post("/augment/jobs")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_augment_job():
    payload = request.get_json(force=True) or {}
    dataset_id = str(payload.get("dataset_id") or "").strip()
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    if not datasets_repo.get_dataset(dataset_id):
        return jsonify({"error": "Dataset not found"}), 404

    raw_transforms = payload.get("transforms")
    transforms: list[str] = list(raw_transforms) if isinstance(raw_transforms, list) else ["flip_h", "brightness", "blur"]
    multiplier = max(1, min(10, int(payload.get("multiplier") or 2)))

    job = augment_repo.create_job(
        dataset_id,
        transforms=transforms,
        multiplier=multiplier,
        params=dict(payload),
    )
    return jsonify(job), 201


@workstation_blueprint.post("/augment/jobs/<job_id>/cancel")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def cancel_augment_job(job_id: str):
    try:
        result = augment_repo.cancel_job(job_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)


@workstation_blueprint.get("/train/jobs")
@require_auth
def list_training_jobs():
    return jsonify(training_service.list_jobs())


@workstation_blueprint.get("/train/base-models")
@require_auth
def list_training_base_models():
    family = request.args.get("family")
    try:
        return jsonify(list_base_models(family))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@workstation_blueprint.post("/train/jobs")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def create_training_job():
    payload = request.get_json(force=True) or {}
    dataset_id = str(payload.get("dataset_id") or "").strip()
    base_model = str(payload.get("base_model") or "baseline").strip() or "baseline"
    base_model_family = str(payload.get("base_model_family") or "").strip() or None
    base_model_variant = str(payload.get("base_model_variant") or "").strip() or None
    device_mode = str(payload.get("device_mode") or payload.get("device") or "auto").strip().lower() or "auto"
    dataset_version_id = str(payload.get("dataset_version_id") or "").strip() or None
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    if device_mode not in {"auto", "gpu", "cpu"}:
        return jsonify({"error": "device_mode must be one of: auto, gpu, cpu"}), 400
    try:
        base_model_spec = resolve_base_model(base_model, family=base_model_family, variant=base_model_variant)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if dataset_version_id:
        dataset_version = dataset_versions_repo.get_version(dataset_version_id)
        if dataset_version is None or str(dataset_version.get("dataset_id")) != str(dataset_id):
            return jsonify({"error": "Dataset version not found"}), 404
        payload["dataset_version_id"] = dataset_version["id"]
        payload["dataset_version_number"] = dataset_version.get("version_number")
        payload["dataset_version_name"] = dataset_version.get("name")
        payload["dataset_version_display_label"] = dataset_version.get("display_label")
        payload["dataset_version_status"] = dataset_version.get("status")
        payload["dataset_version_export_format"] = dataset_version.get("export_format")
        payload["dataset_version_export_root"] = dataset_version.get("export_root")
        payload["dataset_version_manifest_path"] = dataset_version.get("manifest_path")
        payload["dataset_version_split_ratios"] = dataset_version.get("split_ratios")
    if base_model_spec is not None:
        resolved_base_model = base_model_spec["id"]
        payload["base_model"] = base_model_spec["id"]
        payload["base_model_catalog_id"] = base_model_spec["id"]
        payload["base_model_family"] = base_model_spec["family"]
        payload["base_model_variant"] = base_model_spec["variant"]
        payload["base_model_display_name"] = base_model_spec["display_name"]
        payload["base_model_weights_name"] = base_model_spec["weights_name"]
        payload["base_model_runtime"] = base_model_spec["runtime"]
        payload["base_model_task"] = base_model_spec["task"]
        payload["base_model_source"] = base_model_spec["source"]
        payload["base_model_spec"] = base_model_spec
    else:
        if base_model_family or base_model_variant:
            return jsonify({"error": "Unsupported base model family/variant"}), 400
        resolved_base_model = base_model
        payload["base_model"] = base_model
    payload["device_mode"] = device_mode
    return jsonify(training_service.create_job(dataset_id, resolved_base_model, dict(payload))), 201


@workstation_blueprint.get("/train/jobs/<job_id>")
@require_auth
def get_training_job(job_id: str):
    job = training_service.get_job(job_id)
    if job is None:
        return jsonify({"error": "Training job not found"}), 404
    return jsonify(job)


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


@workstation_blueprint.post("/models/<int:model_id>/transition")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def transition_model_lifecycle(model_id: int):
    payload = request.get_json(force=True) or {}
    new_status = str(payload.get("status") or "").strip().lower()
    if not new_status:
        return jsonify({"error": "status is required"}), 400
    note = str(payload.get("note") or "").strip() or None
    actor = getattr(g, "current_user", None)
    try:
        result = models_repo.transition_lifecycle(
            model_id,
            new_status,
            actor_id=actor.id if actor else None,
            note=note,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)


# ---------------------------------------------------------------------------
# Workstation registry + heartbeat
# ---------------------------------------------------------------------------

@workstation_blueprint.get("/workstations")
@require_roles(UserRole.ADMIN, UserRole.ENGINEER)
def list_workstations():
    """List all registered workstations and their last-seen timestamps."""
    return jsonify(workstation_registry_repo.list_workstations())


@workstation_blueprint.post("/workstations/heartbeat")
@require_auth
def workstation_heartbeat():
    """Register or update a workstation's identity and trigger stale session cleanup.

    Body (all optional except machine_id):
        machine_id     — unique identifier for this client machine (required)
        client_version — client application version string
        line_id        — production line this workstation is assigned to
        station_id     — station within the line
    """
    payload = request.get_json(force=True) or {}
    machine_id = str(payload.get("machine_id") or "").strip()
    if not machine_id:
        return jsonify({"error": "machine_id is required"}), 400

    forwarded_for = str(request.headers.get("X-Forwarded-For") or "").strip()
    ip_address = forwarded_for.split(",", 1)[0].strip() if forwarded_for else (request.remote_addr or "")

    record = workstation_registry_repo.heartbeat(
        machine_id=machine_id,
        client_version=str(payload.get("client_version") or "").strip() or None,
        line_id=str(payload.get("line_id") or "").strip() or None,
        station_id=str(payload.get("station_id") or "").strip() or None,
        ip_address=ip_address or None,
    )

    # Best-effort stale session cleanup on each heartbeat (non-blocking)
    try:
        purged = token_store.purge_expired()
    except Exception:  # noqa: BLE001
        purged = 0

    return jsonify({
        "ok": True,
        "workstation": record,
        "sessions_purged": purged,
        "server_time": datetime.now(UTC).isoformat(),
    })

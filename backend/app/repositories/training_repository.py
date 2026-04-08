from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from backend.app.repositories.base_json import JsonRepository

# Valid state machine transitions.
_TRANSITIONS: dict[str, set[str]] = {
    "queued":    {"running", "cancelled"},
    "running":   {"completed", "failed", "cancelled"},
    "completed": set(),
    "failed":    {"queued"},  # allow re-queue after failure
    "cancelled": {"queued"},  # allow re-queue after cancel
}


class TrainingRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("training_jobs.json", {"jobs": []})

    def list_jobs(self) -> list[dict]:
        return self.load()["jobs"]

    def get_job(self, job_id: str) -> dict | None:
        return next((j for j in self.list_jobs() if j["id"] == job_id), None)

    def create_job(self, dataset_id: str, base_model: str, params: dict) -> dict:
        payload = self.load()
        items = payload["jobs"]
        now = datetime.now(UTC).isoformat()
        request_params = dict(params or {})
        base_model_spec = request_params.get("base_model_spec")
        if not isinstance(base_model_spec, dict):
            base_model_spec = {}
        dataset_version_spec = request_params.get("dataset_version_spec")
        if not isinstance(dataset_version_spec, dict):
            dataset_version_spec = {}
        requested_base_model = str(base_model or request_params.get("base_model") or "").strip()
        requested_device_mode = str(request_params.get("device_mode") or "auto").strip().lower() or "auto"
        if requested_device_mode not in {"auto", "gpu", "cpu"}:
            requested_device_mode = "auto"
        request_params["device_mode"] = requested_device_mode
        base_model_catalog_id = str(base_model_spec.get("id") or "").strip() or None
        base_model_family = str(request_params.get("base_model_family") or base_model_spec.get("family") or "").strip().lower() or None
        base_model_variant = str(request_params.get("base_model_variant") or base_model_spec.get("variant") or "").strip().lower() or None
        base_model_display_name = str(request_params.get("base_model_display_name") or base_model_spec.get("display_name") or "").strip() or None
        base_model_weights_name = str(request_params.get("base_model_weights_name") or base_model_spec.get("weights_name") or "").strip() or None
        base_model_runtime = str(request_params.get("base_model_runtime") or base_model_spec.get("runtime") or "").strip().lower() or None
        base_model_task = str(request_params.get("base_model_task") or base_model_spec.get("task") or "").strip().lower() or None
        base_model_source = str(request_params.get("base_model_source") or base_model_spec.get("source") or ("catalog" if base_model_spec else "legacy")).strip().lower() or ("catalog" if base_model_spec else "legacy")
        dataset_version_id = str(request_params.get("dataset_version_id") or dataset_version_spec.get("id") or "").strip() or None
        dataset_version_number = request_params.get("dataset_version_number") or dataset_version_spec.get("version_number")
        dataset_version_name = str(request_params.get("dataset_version_name") or dataset_version_spec.get("name") or "").strip() or None
        dataset_version_display_label = str(
            request_params.get("dataset_version_display_label") or dataset_version_spec.get("display_label") or ""
        ).strip() or None
        dataset_version_status = str(request_params.get("dataset_version_status") or dataset_version_spec.get("status") or "").strip().lower() or None
        dataset_version_export_format = str(
            request_params.get("dataset_version_export_format") or dataset_version_spec.get("export_format") or ""
        ).strip().lower() or None
        dataset_version_export_root = str(
            request_params.get("dataset_version_export_root") or dataset_version_spec.get("export_root") or ""
        ).strip() or None
        dataset_version_manifest_path = str(
            request_params.get("dataset_version_manifest_path") or dataset_version_spec.get("manifest_path") or ""
        ).strip() or None
        dataset_version_split_ratios = request_params.get("dataset_version_split_ratios") or dataset_version_spec.get("split_ratios")
        record: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "dataset_id": dataset_id,
            "base_model": requested_base_model or base_model_catalog_id or "baseline",
            "base_model_catalog_id": base_model_catalog_id,
            "base_model_family": base_model_family,
            "base_model_variant": base_model_variant,
            "base_model_display_name": base_model_display_name,
            "base_model_weights_name": base_model_weights_name,
            "base_model_runtime": base_model_runtime,
            "base_model_task": base_model_task,
            "base_model_source": base_model_source,
            "dataset_version_id": dataset_version_id,
            "dataset_version_number": dataset_version_number,
            "dataset_version_name": dataset_version_name,
            "dataset_version_display_label": dataset_version_display_label,
            "dataset_version_status": dataset_version_status,
            "dataset_version_export_format": dataset_version_export_format,
            "dataset_version_export_root": dataset_version_export_root,
            "dataset_version_manifest_path": dataset_version_manifest_path,
            "dataset_version_split_ratios": dataset_version_split_ratios,
            "status": "queued",
            "trained_model_path": None,
            "params": request_params,
            "requested_device_mode": requested_device_mode,
            "effective_device": None,
            "device_backend": None,
            "device_fallback_reason": None,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "log": [
                f"Training job queued (device: {requested_device_mode}, base model: {base_model_display_name or base_model_catalog_id or requested_base_model or 'baseline'}"
                f"{', dataset version: ' + (dataset_version_display_label or dataset_version_name or dataset_version_id) if dataset_version_display_label or dataset_version_name or dataset_version_id else ''})."
            ],
        }
        items.append(record)
        self.save(payload)
        return record

    def transition(self, job_id: str, new_status: str, *, log_line: str | None = None, **extra_fields) -> dict:
        """Transition a job to new_status, raising ValueError if the transition is invalid."""
        payload = self.load()
        for item in payload["jobs"]:
            if item["id"] != job_id:
                continue
            current = item["status"]
            allowed = _TRANSITIONS.get(current, set())
            if new_status not in allowed:
                raise ValueError(
                    f"Cannot transition job '{job_id}' from '{current}' to '{new_status}'. "
                    f"Allowed: {sorted(allowed) or 'none'}"
                )
            now = datetime.now(UTC).isoformat()
            item["status"] = new_status
            if new_status == "running" and item.get("started_at") is None:
                item["started_at"] = now
            if new_status in {"completed", "failed", "cancelled"}:
                item["finished_at"] = now
            for key, value in extra_fields.items():
                item[key] = value
            if log_line:
                item.setdefault("log", []).append(log_line)
            self.save(payload)
            return dict(item)
        raise ValueError(f"Training job '{job_id}' not found.")

    def cancel_job(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job is None:
            raise ValueError("Training job not found.")
        if job["status"] not in {"queued", "running"}:
            raise ValueError(f"Cannot cancel job in status '{job['status']}'.")
        return self.transition(job_id, "cancelled", log_line="Job cancelled by user.")


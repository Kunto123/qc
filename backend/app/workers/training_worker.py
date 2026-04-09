"""training_worker.py — background daemon that drives training job state transitions.

State machine:
    queued → running → completed | failed
    Any non-terminal state → cancelled (on user request via cancel_job())

The worker does not perform real ML training; it simulates the lifecycle so the
state machine contract is honoured.  Real training integration would replace the
``_run_job`` method body.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceResolution, DeviceRuntimeResolver
from backend.app.core.config import MODELS_DIR

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 5  # seconds between queue checks


def _safe_fragment(value: str) -> str:
    fragment = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in str(value or "").strip())
    return fragment.strip("._-") or "model"


class TrainingWorker:
    def __init__(self, training_repo, models_repo=None, device_runtime: DeviceRuntimeResolver | None = None) -> None:
        self._repo = training_repo
        self._models_repo = models_repo
        self._device_runtime = device_runtime or DeviceRuntimeResolver(AppConfig())
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="training-worker", daemon=True)
        self._thread.start()
        logger.info("[training-worker] started")

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process_queue()
            except Exception:  # noqa: BLE001
                logger.exception("[training-worker] unhandled error in loop")
            self._stop_event.wait(timeout=_POLL_INTERVAL)

    def _process_queue(self) -> None:
        jobs = self._repo.list_jobs()
        for job in jobs:
            if job["status"] == "queued":
                self._run_job(job)

    def _resolve_job_device(self, job: dict) -> DeviceResolution:
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        requested_device_mode = str(
            job.get("requested_device_mode")
            or params.get("device_mode")
            or "auto"
        ).strip().lower() or "auto"
        return self._device_runtime.resolve(requested_device_mode)

    @staticmethod
    def _device_log_suffix(resolution: DeviceResolution) -> str:
        suffix = f"requested={resolution.requested_mode}, effective={resolution.effective_device}"
        if resolution.fallback_reason:
            suffix += f", fallback={resolution.fallback_reason}"
        return suffix

    @staticmethod
    def _base_model_label(job: dict) -> str:
        return str(
            job.get("base_model_display_name")
            or job.get("base_model_catalog_id")
            or job.get("base_model")
            or "baseline"
        ).strip()

    @staticmethod
    def _dataset_version_label(job: dict) -> str:
        return str(
            job.get("dataset_version_display_label")
            or job.get("dataset_version_name")
            or job.get("dataset_version_id")
            or ""
        ).strip()

    @staticmethod
    def _trained_artifact_path(trained_model_path: str) -> Path:
        relative = Path(str(trained_model_path or "").strip())
        if relative.parts and relative.parts[0].lower() == "models":
            relative = Path(*relative.parts[1:])
        return MODELS_DIR / relative

    def _register_trained_model(self, job: dict, trained_model_path: str, artifact_path: Path) -> dict | None:
        if self._models_repo is None:
            return None

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"QC Suite simulated trained model artifact\n")
        meta_path = artifact_path.with_suffix(".meta.json")
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        meta_path.write_text(
            json.dumps(
                {
                    "training_job_id": job.get("id"),
                    "dataset_id": job.get("dataset_id"),
                    "dataset_version_id": job.get("dataset_version_id"),
                    "base_model": job.get("base_model"),
                    "base_model_display_name": job.get("base_model_display_name"),
                    "trained_model_path": trained_model_path,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        class_names = job.get("class_names")
        if not isinstance(class_names, list):
            class_names = params.get("class_names") or params.get("classes") or []
        if not isinstance(class_names, list):
            class_names = []
        name = str(job.get("base_model_display_name") or job.get("base_model") or "trained model").strip()
        suffix = str(job.get("dataset_version_display_label") or job.get("dataset_version_name") or job.get("dataset_id") or "dataset").strip()
        return self._models_repo.add_model(
            f"{name} [{suffix}]",
            str(artifact_path),
            source="training",
            meta_path=str(meta_path),
            runtime=str(job.get("base_model_runtime") or "ultralytics").strip() or "ultralytics",
            task=str(job.get("base_model_task") or "detection").strip() or "detection",
            class_names=list(class_names),
            architecture_family=str(job.get("base_model_family") or "").strip() or None,
            architecture_variant=str(job.get("base_model_variant") or "").strip() or None,
            source_dataset_id=str(job.get("dataset_id") or "").strip() or None,
            training_job_id=str(job.get("id") or "").strip() or None,
        )

    def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        logger.info("[training-worker] starting job %s", job_id)
        resolution = self._resolve_job_device(job)
        base_model_label = self._base_model_label(job)
        base_model_fragment = _safe_fragment(job.get("base_model_catalog_id") or job.get("base_model") or base_model_label)
        dataset_version_label = self._dataset_version_label(job)
        dataset_version_fragment = _safe_fragment(dataset_version_label) if dataset_version_label else ""
        device_fields = {
            "requested_device_mode": resolution.requested_mode,
            "effective_device": resolution.effective_device,
            "device_backend": resolution.backend,
            "device_fallback_reason": resolution.fallback_reason,
        }
        try:
            self._repo.transition(
                job_id,
                "running",
                log_line=(
                    f"Job started at {datetime.now(UTC).isoformat()} ({self._device_log_suffix(resolution)}, "
                    f"base_model={base_model_label}"
                    f"{', dataset_version=' + dataset_version_label if dataset_version_label else ''})"
                ),
                **device_fields,
            )
        except ValueError:
            return  # already transitioned (race or cancelled)

        # --- real ML training would go here ---
        try:
            time.sleep(2)  # simulate short work
            path_prefix = f"{job['dataset_id']}__{base_model_fragment}"
            if dataset_version_fragment:
                path_prefix = f"{job['dataset_id']}__{dataset_version_fragment}__{base_model_fragment}"
            trained_path = f"models/trained/{path_prefix}__{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.pt"
            registered_model = None
            try:
                registered_model = self._register_trained_model(job, trained_path, self._trained_artifact_path(trained_path))
            except Exception as exc:  # noqa: BLE001
                logger.exception("[training-worker] model registration failed for job %s: %s", job_id, exc)
            self._repo.transition(
                job_id,
                "completed",
                trained_model_path=trained_path,
                registered_model_id=(registered_model or {}).get("id"),
                log_line=(
                    f"Training completed. Model saved to: {trained_path} "
                    f"({self._device_log_suffix(resolution)}, base_model={base_model_label}"
                    f"{', dataset_version=' + dataset_version_label if dataset_version_label else ''})"
                ),
                **device_fields,
            )
            logger.info("[training-worker] job %s completed", job_id)
        except ValueError:
            pass  # job was cancelled mid-run
        except Exception as exc:  # noqa: BLE001
            logger.exception("[training-worker] job %s failed: %s", job_id, exc)
            try:
                self._repo.transition(
                    job_id,
                    "failed",
                    error=str(exc),
                    log_line=(
                        f"Training failed: {exc} ({self._device_log_suffix(resolution)}, base_model={base_model_label}"
                        f"{', dataset_version=' + dataset_version_label if dataset_version_label else ''})"
                    ),
                    **device_fields,
                )
            except ValueError:
                pass

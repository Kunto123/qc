from __future__ import annotations

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceRuntimeResolver
from backend.app.repositories.training_repository import TrainingRepository
from backend.app.workers.training_worker import TrainingWorker


class TrainingService:
    def __init__(
        self,
        training_repo: TrainingRepository,
        models_repo=None,
        device_runtime: DeviceRuntimeResolver | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        self._training_repo = training_repo
        self._worker = TrainingWorker(
            training_repo,
            models_repo=models_repo,
            device_runtime=device_runtime,
            app_config=app_config,
        )
        self._worker.start()

    def list_jobs(self) -> list[dict]:
        return self._training_repo.list_jobs()

    def get_job(self, job_id: str) -> dict | None:
        return self._training_repo.get_job(job_id)

    def create_job(self, dataset_id: str, base_model: str, params: dict) -> dict:
        return self._training_repo.create_job(dataset_id, base_model, params)

    def cancel_job(self, job_id: str) -> dict:
        return self._training_repo.cancel_job(job_id)

    def delete_job(self, job_id: str) -> dict:
        return self._training_repo.delete_job(job_id)


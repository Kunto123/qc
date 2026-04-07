from __future__ import annotations

from backend.app.repositories.training_repository import TrainingRepository
from backend.app.workers.training_worker import TrainingWorker


class TrainingService:
    def __init__(self, training_repo: TrainingRepository) -> None:
        self._training_repo = training_repo
        self._worker = TrainingWorker(training_repo)
        self._worker.start()

    def list_jobs(self) -> list[dict]:
        return self._training_repo.list_jobs()

    def get_job(self, job_id: str) -> dict | None:
        return self._training_repo.get_job(job_id)

    def create_job(self, dataset_id: str, base_model: str, params: dict) -> dict:
        return self._training_repo.create_job(dataset_id, base_model, params)

    def cancel_job(self, job_id: str) -> dict:
        return self._training_repo.cancel_job(job_id)


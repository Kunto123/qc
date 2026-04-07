from __future__ import annotations

from datetime import UTC, datetime
import uuid

from backend.app.repositories.base_json import JsonRepository


class TrainingRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("training_jobs.json", {"jobs": []})

    def list_jobs(self) -> list[dict]:
        return self.load()["jobs"]

    def create_job(self, dataset_id: str, base_model: str, params: dict) -> dict:
        payload = self.load()
        items = payload["jobs"]
        now = datetime.now(UTC).isoformat()
        record = {
            "id": uuid.uuid4().hex[:12],
            "dataset_id": dataset_id,
            "base_model": base_model,
            "status": "completed",
            "trained_model_path": f"models/trained/{dataset_id}__{base_model}__latest.pt",
            "params": params,
            "created_at": now,
            "started_at": now,
            "finished_at": now,
            "log": [
                "Training job created.",
                "This MVP stores metadata and simulates an immediate successful run.",
            ],
        }
        items.append(record)
        self.save(payload)
        return record

    def cancel_job(self, job_id: str) -> dict:
        payload = self.load()
        for item in payload["jobs"]:
            if item["id"] == job_id:
                item["status"] = "cancelled"
                item["finished_at"] = datetime.now(UTC).isoformat()
                self.save(payload)
                return item
        raise ValueError("Training job not found.")


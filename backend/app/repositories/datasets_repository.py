from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.config import DATASETS_DIR, ensure_data_dirs
from backend.app.repositories.base_json import JsonRepository


class DatasetsRepository(JsonRepository):
    def __init__(self) -> None:
        ensure_data_dirs()
        super().__init__("datasets.json", {"datasets": []})

    def list_datasets(self) -> list[dict]:
        return self.load()["datasets"]

    def create_dataset(self, name: str, description: str = "") -> dict:
        payload = self.load()
        items = payload["datasets"]
        dataset_id = uuid.uuid4().hex[:12]
        dataset_dir = DATASETS_DIR / dataset_id
        for subdir in ("images", "labels", "exports"):
            (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)
        record = {
            "id": dataset_id,
            "name": name,
            "description": description,
            "folder_name": dataset_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        items.append(record)
        self.save(payload)
        return record

    def get_dataset(self, dataset_id: str) -> dict | None:
        return next((item for item in self.list_datasets() if item["id"] == dataset_id), None)

    def dataset_dir(self, dataset_id: str) -> Path:
        return DATASETS_DIR / dataset_id

    def list_files(self, dataset_id: str, target: str) -> list[dict]:
        folder = self.dataset_dir(dataset_id) / target
        folder.mkdir(parents=True, exist_ok=True)
        return [
            {"name": item.name, "path": str(item), "size": item.stat().st_size}
            for item in sorted(folder.iterdir())
            if item.is_file()
        ]

    def save_file(self, dataset_id: str, target: str, file_name: str, content: bytes) -> dict:
        folder = self.dataset_dir(dataset_id) / target
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / Path(file_name).name
        out_path.write_bytes(content)
        return {"name": out_path.name, "path": str(out_path), "size": out_path.stat().st_size}

    def save_annotation(self, dataset_id: str, image_name: str, labels: list[dict]) -> dict:
        folder = self.dataset_dir(dataset_id) / "labels"
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / f"{Path(image_name).stem}.json"
        out_path.write_text(json.dumps(labels, ensure_ascii=True, indent=2), encoding="utf-8")
        return {
            "image_name": image_name,
            "labels": labels,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def get_annotation(self, dataset_id: str, image_name: str) -> list[dict]:
        out_path = self.dataset_dir(dataset_id) / "labels" / f"{Path(image_name).stem}.json"
        if not out_path.exists():
            return []
        return json.loads(out_path.read_text(encoding="utf-8"))

    def delete_dataset(self, dataset_id: str) -> bool:
        payload = self.load()
        before = len(payload["datasets"])
        payload["datasets"] = [item for item in payload["datasets"] if item["id"] != dataset_id]
        if len(payload["datasets"]) == before:
            return False
        self.save(payload)
        shutil.rmtree(self.dataset_dir(dataset_id), ignore_errors=True)
        return True

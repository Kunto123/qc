from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.core.config import DEFAULT_STICKER_MODEL_META_PATH, DEFAULT_STICKER_MODEL_PATH
from backend.app.repositories.base_json import JsonRepository


def _load_default_meta() -> dict[str, Any]:
    path = Path(DEFAULT_STICKER_MODEL_META_PATH)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_models_payload() -> dict[str, Any]:
    meta = _load_default_meta()
    now = datetime.now(UTC).isoformat()
    if not DEFAULT_STICKER_MODEL_PATH:
        return {"models": []}
    return {
        "models": [
            {
                "id": 1,
                "name": "AKH Sticker Detector",
                "path": DEFAULT_STICKER_MODEL_PATH,
                "meta_path": DEFAULT_STICKER_MODEL_META_PATH or None,
                "source": "seeded-default",
                "runtime": "ultralytics",
                "task": "detection",
                "architecture_family": meta.get("architecture_family") or "yolov5",
                "architecture_variant": meta.get("architecture_variant") or "unknown",
                "class_names": list(meta.get("class_names") or []),
                "created_at": now,
            }
        ]
    }


class ModelsRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("models.json", _default_models_payload())

    def list_models(self) -> list[dict]:
        return self.load()["models"]

    def get_model(self, model_id: int) -> dict | None:
        return next((item for item in self.list_models() if int(item["id"]) == int(model_id)), None)

    def find_by_path(self, path: str) -> dict | None:
        normalized = str(path or "").strip().lower()
        return next(
            (item for item in self.list_models() if str(item.get("path") or "").strip().lower() == normalized),
            None,
        )

    def add_model(
        self,
        name: str,
        path: str,
        source: str = "manual",
        *,
        meta_path: str | None = None,
        runtime: str = "ultralytics",
        task: str = "detection",
        class_names: list[str] | None = None,
        architecture_family: str | None = None,
        architecture_variant: str | None = None,
    ) -> dict:
        payload = self.load()
        items = payload["models"]
        record = {
            "id": self.next_id(items),
            "name": name,
            "path": path,
            "meta_path": meta_path,
            "source": source,
            "runtime": runtime,
            "task": task,
            "class_names": list(class_names or []),
            "architecture_family": architecture_family,
            "architecture_variant": architecture_variant,
            "created_at": datetime.now(UTC).isoformat(),
        }
        items.append(record)
        self.save(payload)
        return record

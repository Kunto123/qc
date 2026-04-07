from __future__ import annotations

from datetime import UTC, datetime

from backend.app.repositories.base_json import JsonRepository


class DeploymentsRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("deployments.json", {"deployments": []})

    def list_deployments(self) -> list[dict]:
        return self.load()["deployments"]

    def deploy(
        self,
        *,
        template_id: int,
        template_version_id: int,
        line_id: str,
        station_id: str,
        deployed_by: int | None,
        template_name: str,
        version_number: int,
    ) -> dict:
        payload = self.load()
        items = payload["deployments"]
        now = datetime.now(UTC).isoformat()
        for item in items:
            if item["line_id"] == line_id and item["station_id"] == station_id and item["is_active"]:
                item["is_active"] = False
                item["effective_until"] = now
        record = {
            "id": self.next_id(items),
            "template_id": int(template_id),
            "template_version_id": int(template_version_id),
            "line_id": line_id,
            "station_id": station_id,
            "is_active": True,
            "deployed_by": deployed_by,
            "effective_from": now,
            "effective_until": None,
            "created_at": now,
            "template_name": template_name,
            "version_number": version_number,
        }
        items.append(record)
        self.save(payload)
        return record

    def get_active(self, line_id: str, station_id: str) -> dict | None:
        return next(
            (
                item
                for item in reversed(self.list_deployments())
                if item["line_id"] == line_id
                and item["station_id"] == station_id
                and item["is_active"]
            ),
            None,
        )

    def deactivate(self, deployment_id: int) -> bool:
        payload = self.load()
        now = datetime.now(UTC).isoformat()
        for item in payload["deployments"]:
            if int(item["id"]) == int(deployment_id) and item["is_active"]:
                item["is_active"] = False
                item["effective_until"] = now
                self.save(payload)
                return True
        return False


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

    def rollback(self, deployment_id: int, *, rolled_back_by: int | None = None) -> dict:
        """Deactivate deployment_id and re-deploy the previous version for the same line/station."""
        payload = self.load()
        items = payload["deployments"]
        now = datetime.now(UTC).isoformat()

        target = next((d for d in items if int(d["id"]) == int(deployment_id)), None)
        if target is None:
            raise ValueError(f"Deployment {deployment_id} not found.")

        line_id = target["line_id"]
        station_id = target["station_id"]

        # Find the most recent *other* deployment for the same line/station
        previous = next(
            (
                d for d in reversed(items)
                if d["line_id"] == line_id
                and d["station_id"] == station_id
                and int(d["id"]) != int(deployment_id)
            ),
            None,
        )
        if previous is None:
            raise ValueError("No previous deployment found to roll back to.")

        # Deactivate all active deployments for this line/station
        for item in items:
            if item["line_id"] == line_id and item["station_id"] == station_id and item["is_active"]:
                item["is_active"] = False
                item["effective_until"] = now

        # Create a new deployment record based on the previous one
        record = {
            "id": self.next_id(items),
            "template_id": previous["template_id"],
            "template_version_id": previous["template_version_id"],
            "line_id": line_id,
            "station_id": station_id,
            "is_active": True,
            "deployed_by": rolled_back_by,
            "effective_from": now,
            "effective_until": None,
            "created_at": now,
            "template_name": previous.get("template_name"),
            "version_number": previous.get("version_number"),
            "rollback_from_deployment_id": int(deployment_id),
        }
        items.append(record)
        self.save(payload)
        return record


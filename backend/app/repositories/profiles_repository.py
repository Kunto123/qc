from __future__ import annotations

from datetime import UTC, datetime

from backend.app.repositories.base_json import JsonRepository


class ProfilesRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("profiles.json", {"profiles": []})

    def list_profiles(self) -> list[dict]:
        return self.load()["profiles"]

    def get(self, profile_id: int) -> dict | None:
        return next((item for item in self.list_profiles() if int(item["id"]) == int(profile_id)), None)

    def create(self, name: str, profile: dict) -> dict:
        payload = self.load()
        items = payload["profiles"]
        record = {
            "id": self.next_id(items),
            "name": name,
            "profile": profile,
            "created_at": datetime.now(UTC).isoformat(),
        }
        items.append(record)
        self.save(payload)
        return record

    def delete(self, profile_id: int) -> bool:
        payload = self.load()
        before = len(payload["profiles"])
        payload["profiles"] = [
            item for item in payload["profiles"] if int(item["id"]) != int(profile_id)
        ]
        if len(payload["profiles"]) == before:
            return False
        self.save(payload)
        return True


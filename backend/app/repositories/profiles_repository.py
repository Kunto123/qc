from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from backend.app.repositories.base_json import JsonRepository


def _is_expired(record: dict) -> bool:
    expires_at = record.get("expires_at")
    if not expires_at:
        return False
    try:
        exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= exp_dt
    except (ValueError, TypeError):
        return False


class ProfilesRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("profiles.json", {"profiles": []})

    def list_profiles(self) -> list[dict]:
        profiles = self.load()["profiles"]
        now = datetime.now(UTC)
        result = []
        for item in profiles:
            record = dict(item)
            record["is_expired"] = _is_expired(record)
            result.append(record)
        return result

    def get(self, profile_id: int) -> dict | None:
        item = next((item for item in self.load()["profiles"] if int(item["id"]) == int(profile_id)), None)
        if item is None:
            return None
        record = dict(item)
        record["is_expired"] = _is_expired(record)
        return record

    def get_active_for_scope(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
    ) -> dict | None:
        """Return the most recent non-expired profile matching the given scope.

        Scope matching is hierarchical: an exact match beats a partial match.
        A None field on the stored profile means "any" (wildcard).
        """
        profiles = self.load()["profiles"]
        candidates = []
        for item in profiles:
            if _is_expired(item):
                continue
            scope_line = item.get("scope_line_id")
            scope_station = item.get("scope_station_id")
            scope_part = item.get("scope_part_name")
            # each scope field: None = wildcard, value = must match
            if scope_line is not None and scope_line != line_id:
                continue
            if scope_station is not None and scope_station != station_id:
                continue
            if scope_part is not None and scope_part != part_name:
                continue
            # specificity score: how many fields matched exactly
            score = (
                (1 if scope_line == line_id else 0)
                + (1 if scope_station == station_id else 0)
                + (1 if scope_part == part_name else 0)
            )
            candidates.append((score, item.get("version", 1), item))

        if not candidates:
            return None
        # prefer highest specificity, then latest version
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = candidates[0][2]
        result = dict(best)
        result["is_expired"] = False
        return result

    def create(
        self,
        name: str,
        profile: dict,
        *,
        scope_line_id: str | None = None,
        scope_station_id: str | None = None,
        scope_part_name: str | None = None,
        expiry_interval_days: int | None = None,
    ) -> dict:
        payload = self.load()
        items = payload["profiles"]

        # Version numbering per scope
        scope_items = [
            item for item in items
            if item.get("scope_line_id") == scope_line_id
            and item.get("scope_station_id") == scope_station_id
            and item.get("scope_part_name") == scope_part_name
        ]
        version = max((int(item.get("version") or 1) for item in scope_items), default=0) + 1

        now = datetime.now(UTC)
        expires_at: str | None = None
        if expiry_interval_days and expiry_interval_days > 0:
            expires_at = (now + timedelta(days=expiry_interval_days)).isoformat()

        record: dict[str, Any] = {
            "id": self.next_id(items),
            "name": name,
            "profile": profile,
            "version": version,
            "scope_line_id": scope_line_id,
            "scope_station_id": scope_station_id,
            "scope_part_name": scope_part_name,
            "expires_at": expires_at,
            "created_at": now.isoformat(),
        }
        items.append(record)
        self.save(payload)
        result = dict(record)
        result["is_expired"] = False
        return result

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


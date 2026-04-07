from __future__ import annotations

import requests


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token: str | None = None
        self.session = requests.Session()

    def set_token(self, token: str | None) -> None:
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request_json(self, method: str, path: str, *, params: dict | None = None, payload: dict | None = None, timeout: int = 20):
        response = self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            json=payload,
            headers=self._headers(),
            timeout=timeout,
        )
        if not response.ok:
            detail = response.text
            try:
                detail = response.json().get("error") or detail
            except ValueError:
                pass
            raise RuntimeError(f"{response.status_code}: {detail}")
        return response.json()

    def _get(self, path: str, params: dict | None = None):
        return self._request_json("GET", path, params=params, timeout=15)

    def _post(self, path: str, payload: dict):
        return self._request_json("POST", path, payload=payload, timeout=20)

    def _put(self, path: str, payload: dict):
        return self._request_json("PUT", path, payload=payload, timeout=20)

    def _delete(self, path: str):
        return self._request_json("DELETE", path, timeout=20)

    def login(self, username: str, password: str) -> dict:
        return self._post("/auth/login", {"username": username, "password": password})

    def me(self) -> dict:
        return self._get("/auth/me")

    def list_templates(self) -> list[dict]:
        return self._get("/templates")

    def get_template(self, template_id: int) -> dict:
        return self._get(f"/templates/{template_id}")

    def create_template(self, payload: dict) -> dict:
        return self._post("/templates", payload)

    def update_template(self, template_id: int, payload: dict) -> dict:
        return self._put(f"/templates/{template_id}", payload)

    def delete_template(self, template_id: int) -> dict:
        return self._delete(f"/templates/{template_id}")

    def list_deployments(self) -> list[dict]:
        return self._get("/deployments")

    def deploy_template(self, payload: dict) -> dict:
        return self._post("/deployments", payload)

    def get_active_deployment(self, line_id: str, station_id: str) -> dict:
        return self._get("/deployments/active", {"line_id": line_id, "station_id": station_id})

    def deactivate_deployment(self, deployment_id: int) -> dict:
        return self._delete(f"/deployments/{deployment_id}")

    def create_session(self, payload: dict) -> dict:
        return self._post("/inspection/sessions/start", payload)

    def update_roi(self, session_id: str, payload: dict) -> dict:
        return self._post(f"/inspection/sessions/{session_id}/roi", payload)

    def update_rois(
        self,
        session_id: str,
        *,
        part_ready_roi: dict | None = None,
        sticker_roi: dict | None = None,
    ) -> dict:
        payload: dict[str, dict] = {}
        if part_ready_roi is not None:
            payload["part_ready_roi"] = part_ready_roi
        if sticker_roi is not None:
            payload["sticker_roi"] = sticker_roi
        return self._post(f"/inspection/sessions/{session_id}/roi", payload)

    def get_latest_preview(self) -> dict:
        return self._get("/inspection/latest-preview")

    def stop_session(self, session_id: str) -> dict:
        return self._post(f"/inspection/sessions/{session_id}/stop", {})

    def push_frame(self, session_id: str, image_b64: str) -> dict:
        return self._post(f"/inspection/sessions/{session_id}/frame", {"image_b64": image_b64})

    def list_inspections(self, params: dict | None = None) -> list[dict]:
        return self._get("/inspections", params)

    def export_inspections_csv(self, params: dict | None = None) -> str:
        """Return raw CSV text from the server export endpoint."""
        response = self.session.request(
            method="GET",
            url=f"{self.base_url}/inspections/export",
            params=params or {},
            headers=self._headers(),
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"{response.status_code}: {response.text}")
        return response.text

    def get_inspection(self, result_id: int) -> dict:
        return self._get(f"/inspections/{result_id}")

    def dashboard_summary(self, params: dict | None = None) -> dict:
        return self._get("/dashboard/summary", params)

    def dashboard_buckets(self, params: dict | None = None) -> list[dict]:
        return self._get("/dashboard/buckets", params)

    def list_users(self) -> list[dict]:
        return self._get("/auth/users")

    def create_user(self, payload: dict) -> dict:
        return self._post("/auth/users", payload)

    def set_user_active(self, user_id: int, is_active: bool) -> dict:
        return self._put(f"/auth/users/{user_id}", {"is_active": is_active})

    def list_profiles(self) -> list[dict]:
        return self._get("/calibration/profiles")

    def compute_color_profile(self, payload: dict) -> dict:
        return self._post("/calibration/color-profile", payload)

    def save_profile(self, payload: dict) -> dict:
        return self._post("/calibration/profiles", payload)

    def delete_profile(self, profile_id: int) -> dict:
        return self._delete(f"/calibration/profiles/{profile_id}")

    def list_datasets(self) -> list[dict]:
        return self._get("/datasets")

    def create_dataset(self, payload: dict) -> dict:
        return self._post("/datasets", payload)

    def delete_dataset(self, dataset_id: str) -> dict:
        return self._delete(f"/datasets/{dataset_id}")

    def list_dataset_files(self, dataset_id: str, target: str = "images") -> list[dict]:
        return self._get(f"/datasets/{dataset_id}/files", {"target": target})

    def upload_dataset_file(self, dataset_id: str, payload: dict) -> dict:
        return self._post(f"/datasets/{dataset_id}/upload", payload)

    def get_annotation(self, dataset_id: str, image_name: str) -> dict:
        return self._get(f"/datasets/{dataset_id}/annotations/{image_name}")

    def save_annotation(self, dataset_id: str, image_name: str, labels: list[dict]) -> dict:
        return self._post(f"/datasets/{dataset_id}/annotations/{image_name}", {"labels": labels})

    def list_augment_jobs(self) -> list[dict]:
        return self._get("/augment/jobs")

    def create_augment_job(self, payload: dict) -> dict:
        return self._post("/augment/jobs", payload)

    def list_training_jobs(self) -> list[dict]:
        return self._get("/train/jobs")

    def create_training_job(self, payload: dict) -> dict:
        return self._post("/train/jobs", payload)

    def cancel_training_job(self, job_id: str) -> dict:
        return self._post(f"/train/jobs/{job_id}/cancel", {})

    def list_models(self) -> list[dict]:
        return self._get("/models")

    def create_model(self, payload: dict) -> dict:
        return self._post("/models", payload)

    def upload_model_file(self, payload: dict) -> dict:
        return self._post("/models/upload", payload)

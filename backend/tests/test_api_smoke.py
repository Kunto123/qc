from __future__ import annotations

import atexit
import base64
from io import BytesIO
import os
import shutil
import time
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from werkzeug.datastructures import MultiDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="qc-suite-tests-"))
atexit.register(lambda: shutil.rmtree(TEST_DATA_ROOT, ignore_errors=True))
os.environ["QC_SUITE_DATA_ROOT"] = str(TEST_DATA_ROOT)
os.environ["QC_SUITE_STICKER_INFERENCE_MODE"] = "classic"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.factory import create_app


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _sample_image_b64() -> str:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (270, 190), (370, 290), (255, 255, 255), thickness=-1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _blank_image_b64() -> str:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _two_roi_ready_image_b64() -> str:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (180, 160), (40, 180, 40), thickness=-1)
    cv2.rectangle(image, (260, 160), (380, 300), (255, 255, 255), thickness=-1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _presence_image_b64() -> str:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (150, 120), (235, 200), (255, 255, 255), thickness=-1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class ApiSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.testing = True
        cls.client = cls.app.test_client()
        cls.admin_token = cls._login("admin", "admin123")
        cls.operator_token = cls._login("operator", "operator123")
        cls.engineer_token = cls._login("engineer", "engineer123")

    @classmethod
    def _login(cls, username: str, password: str) -> str:
        response = cls.client.post("/auth/login", json={"username": username, "password": password})
        assert response.status_code == 200, response.get_json()
        payload = response.get_json()
        return str(payload["token"])

    def test_00_template_detail_exposes_two_rois(self) -> None:
        response = self.client.get("/templates/1", headers=_headers(self.admin_token))
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertIn("part_ready_roi", payload)
        self.assertIn("sticker_roi", payload)
        self.assertEqual(payload["part_ready_roi"]["x"], 0.2)
        self.assertEqual(payload["sticker_roi"]["w"], 0.6)
        self.assertIn("model_meta_path", payload["vision"])

    def test_00b_seeded_model_registry_contains_default_model(self) -> None:
        response = self.client.get("/models", headers=_headers(self.engineer_token))
        self.assertEqual(response.status_code, 200, response.get_json())
        models = response.get_json()
        self.assertTrue(models)
        self.assertEqual(models[0]["name"], "AKH Sticker Detector")
        self.assertEqual(models[0]["runtime"], "ultralytics")

    def test_00b2_base_model_catalog_contains_yolo_variants(self) -> None:
        response = self.client.get("/train/base-models", headers=_headers(self.engineer_token))
        self.assertEqual(response.status_code, 200, response.get_json())
        catalog = response.get_json()
        self.assertTrue(any(item["id"] == "yolov5s" for item in catalog))
        self.assertTrue(any(item["id"] == "yolov11x" for item in catalog))
        self.assertTrue(all(item["task"] == "detection" for item in catalog))

    def test_00c_auth_logout_revokes_current_token(self) -> None:
        login_response = self.client.post(
            "/auth/login",
            json={"username": "operator", "password": "operator123", "client_name": "smoke-auth"},
        )
        self.assertEqual(login_response.status_code, 200, login_response.get_json())
        login_payload = login_response.get_json()
        self.assertIn("expires_at", login_payload)
        self.assertIn("session", login_payload)
        token = str(login_payload["token"])

        sessions_response = self.client.get("/auth/sessions", headers=_headers(token))
        self.assertEqual(sessions_response.status_code, 200, sessions_response.get_json())
        sessions = sessions_response.get_json()
        self.assertTrue(any(item.get("is_current") for item in sessions))
        self.assertTrue(any(item.get("client_name") == "smoke-auth" for item in sessions))

        logout_response = self.client.post("/auth/logout", headers=_headers(token))
        self.assertEqual(logout_response.status_code, 200, logout_response.get_json())
        self.assertTrue(logout_response.get_json()["revoked"])

        me_response = self.client.get("/auth/me", headers=_headers(token))
        self.assertEqual(me_response.status_code, 401, me_response.get_json())

    def test_00e_admin_user_list_does_not_expose_password_hash(self) -> None:
        users_response = self.client.get("/auth/users", headers=_headers(self.admin_token))
        self.assertEqual(users_response.status_code, 200, users_response.get_json())
        users = users_response.get_json()
        self.assertTrue(users)
        self.assertTrue(all("password_hash" not in item for item in users))

    def test_00e2_legacy_engineer_login_is_mapped_to_supported_role(self) -> None:
        login_response = self.client.post(
            "/auth/login",
            json={"username": "engineer", "password": "engineer123"},
        )
        self.assertEqual(login_response.status_code, 200, login_response.get_json())
        user = login_response.get_json().get("user") or {}
        self.assertEqual(user.get("role"), "admin")

    def test_00e3_admin_cannot_create_engineer_role(self) -> None:
        response = self.client.post(
            "/auth/users",
            json={"username": f"blocked_{uuid4().hex[:8]}", "password": "secret123", "role": "engineer"},
            headers=_headers(self.admin_token),
        )
        self.assertEqual(response.status_code, 400, response.get_json())
        payload = response.get_json() or {}
        self.assertIn("must be one of", str(payload.get("error") or ""))

    def test_00e4_admin_cannot_change_role_to_engineer(self) -> None:
        create_response = self.client.post(
            "/auth/users",
            json={"username": f"rolecheck_{uuid4().hex[:8]}", "password": "secret123", "role": "operator"},
            headers=_headers(self.admin_token),
        )
        self.assertEqual(create_response.status_code, 201, create_response.get_json())
        user = create_response.get_json()

        change_response = self.client.post(
            f"/auth/users/{user['id']}/role",
            json={"role": "engineer"},
            headers=_headers(self.admin_token),
        )
        self.assertEqual(change_response.status_code, 400, change_response.get_json())
        payload = change_response.get_json() or {}
        self.assertIn("must be one of", str(payload.get("error") or ""))

    def test_00f_admin_can_retry_failed_push_via_api(self) -> None:
        from backend.app.api import inspection_routes as inspection_routes_module

        class _RetryRepo:
            def retry_result(self, result_id: int) -> dict:
                return {
                    "id": result_id,
                    "push_status": "sent",
                    "retry_count": 1,
                    "last_push_error": None,
                    "sql_mirror_id": 321,
                }

        original_repo = inspection_routes_module.inspection_results_repo
        inspection_routes_module.inspection_results_repo = _RetryRepo()
        try:
            response = self.client.post("/inspections/77/retry-push", headers=_headers(self.admin_token))
        finally:
            inspection_routes_module.inspection_results_repo = original_repo

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["id"], 77)
        self.assertEqual(payload["result"]["push_status"], "sent")

    def test_00g_operator_cannot_retry_failed_push_via_api(self) -> None:
        response = self.client.post("/inspections/77/retry-push", headers=_headers(self.operator_token))
        self.assertEqual(response.status_code, 403, response.get_json())

    def test_00h_admin_can_batch_retry_failed_pushes_via_api(self) -> None:
        from backend.app.api import inspection_routes as inspection_routes_module

        class _BatchRetryRepo:
            def retry_failed(self, *, result_ids=None, limit: int = 100) -> list[dict]:
                return [
                    {"id": 11, "push_status": "sent"},
                    {"id": 12, "push_status": "failed"},
                ]

        original_repo = inspection_routes_module.inspection_results_repo
        inspection_routes_module.inspection_results_repo = _BatchRetryRepo()
        try:
            response = self.client.post(
                "/inspections/retry-push",
                json={"result_ids": [11, 12], "limit": 10},
                headers=_headers(self.admin_token),
            )
        finally:
            inspection_routes_module.inspection_results_repo = original_repo

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["attempted"], 2)
        self.assertEqual(payload["succeeded"], 1)
        self.assertEqual(payload["failed"], 1)

    def test_00c2_login_failure_is_recorded_in_audit_log(self) -> None:
        bad_login = self.client.post("/auth/login", json={"username": "nobody", "password": "wrong"})
        self.assertEqual(bad_login.status_code, 401)

        audit_response = self.client.get("/auth/audit-log", headers=_headers(self.admin_token))
        self.assertEqual(audit_response.status_code, 200, audit_response.get_json())
        events = audit_response.get_json()
        failures = [e for e in events if e.get("event_type") == "login_failure"]
        self.assertTrue(failures, "Expected at least one login_failure audit event")
        self.assertTrue(any(e.get("username") == "nobody" for e in failures))

    def test_00c3_successful_login_is_recorded_in_audit_log(self) -> None:
        # A fresh login so we have a guaranteed event in the log
        fresh_login = self.client.post(
            "/auth/login",
            json={"username": "engineer", "password": "engineer123", "client_name": "audit-smoke"},
        )
        self.assertEqual(fresh_login.status_code, 200)

        audit_response = self.client.get("/auth/audit-log", headers=_headers(self.admin_token))
        self.assertEqual(audit_response.status_code, 200, audit_response.get_json())
        events = audit_response.get_json()
        successes = [e for e in events if e.get("event_type") == "login_success"]
        self.assertTrue(successes, "Expected at least one login_success audit event")

    def test_00c4_audit_log_is_admin_only(self) -> None:
        response = self.client.get("/auth/audit-log", headers=_headers(self.operator_token))
        self.assertEqual(response.status_code, 403, response.get_json())

    def test_00c5_audit_log_limit_param_is_respected(self) -> None:
        audit_response = self.client.get(
            "/auth/audit-log?limit=2",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(audit_response.status_code, 200, audit_response.get_json())
        events = audit_response.get_json()
        self.assertLessEqual(len(events), 2)

    def test_00d_disabling_user_revokes_existing_sessions(self) -> None:
        username = f"revoke_{uuid4().hex[:8]}"
        create_user_response = self.client.post(
            "/auth/users",
            json={"username": username, "password": "phase8pass", "role": "operator"},
            headers=_headers(self.admin_token),
        )
        self.assertEqual(create_user_response.status_code, 201, create_user_response.get_json())
        user = create_user_response.get_json()

        login_response = self.client.post("/auth/login", json={"username": username, "password": "phase8pass"})
        self.assertEqual(login_response.status_code, 200, login_response.get_json())
        user_token = str(login_response.get_json()["token"])

        me_response = self.client.get("/auth/me", headers=_headers(user_token))
        self.assertEqual(me_response.status_code, 200, me_response.get_json())

        disable_response = self.client.put(
            f"/auth/users/{user['id']}",
            json={"is_active": False},
            headers=_headers(self.admin_token),
        )
        self.assertEqual(disable_response.status_code, 200, disable_response.get_json())
        self.assertFalse(disable_response.get_json()["is_active"])

        me_after_disable_response = self.client.get("/auth/me", headers=_headers(user_token))
        self.assertEqual(me_after_disable_response.status_code, 401, me_after_disable_response.get_json())

    def test_01_operator_flow_accepts_centered_detection(self) -> None:
        templates_response = self.client.get("/templates", headers=_headers(self.operator_token))
        self.assertEqual(templates_response.status_code, 200)
        templates = templates_response.get_json()
        self.assertTrue(templates)

        deploy_response = self.client.post(
            "/deployments",
            json={
                "template_id": 1,
                "template_version_id": 1,
                "line_id": "LINE-A",
                "station_id": "ST-01",
            },
            headers=_headers(self.admin_token),
        )
        self.assertEqual(deploy_response.status_code, 201, deploy_response.get_json())

        active_response = self.client.get(
            "/deployments/active?line_id=LINE-A&station_id=ST-01",
            headers=_headers(self.operator_token),
        )
        self.assertEqual(active_response.status_code, 200)
        self.assertIsNotNone(active_response.get_json().get("deployment"))

        session_response = self.client.post(
            "/inspection/sessions/start",
            json={
                "client_id": "operator-smoke",
                "camera_index": 0,
                "template_version_id": 1,
                "line_id": "LINE-A",
                "station_id": "ST-01",
            },
            headers=_headers(self.operator_token),
        )
        self.assertEqual(session_response.status_code, 201, session_response.get_json())
        session_payload = session_response.get_json()

        frame_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/frame",
            json={"image_b64": _sample_image_b64()},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(frame_response.status_code, 200, frame_response.get_json())
        frame_payload = frame_response.get_json()
        self.assertEqual(frame_payload["validation"]["decision"], "ACCEPT")
        self.assertTrue(frame_payload["count_committed"])
        self.assertTrue(frame_payload["db_write"]["written"])
        self.assertEqual(frame_payload["counters"]["session_total"], 1)
        self.assertEqual(frame_payload["counters"]["session_accept"], 1)
        self.assertEqual(frame_payload["counters"]["session_reject"], 0)
        self.assertEqual(frame_payload["sticker_detection"]["backend"], "classic")
        self.assertTrue(frame_payload["sticker_detection"]["model_path"])
        result_id = frame_payload["db_write"]["result_id"]

        duplicate_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/frame",
            json={"image_b64": _sample_image_b64()},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(duplicate_response.status_code, 200, duplicate_response.get_json())
        duplicate_payload = duplicate_response.get_json()
        self.assertFalse(duplicate_payload["count_committed"])
        self.assertEqual(duplicate_payload["event_state"], "cooldown")
        self.assertEqual(duplicate_payload["counters"]["session_total"], 1)

        empty_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/frame",
            json={"image_b64": _blank_image_b64()},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(empty_response.status_code, 200, empty_response.get_json())
        self.assertEqual(empty_response.get_json()["event_state"], "idle")

        second_part_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/frame",
            json={"image_b64": _sample_image_b64()},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(second_part_response.status_code, 200, second_part_response.get_json())
        second_part_payload = second_part_response.get_json()
        self.assertTrue(second_part_payload["count_committed"])
        self.assertEqual(second_part_payload["counters"]["session_total"], 2)

        inspections_response = self.client.get("/inspections", headers=_headers(self.admin_token))
        self.assertEqual(inspections_response.status_code, 200)
        self.assertGreaterEqual(len(inspections_response.get_json()), 2)

        result_response = self.client.get(f"/inspections/{result_id}", headers=_headers(self.admin_token))
        self.assertEqual(result_response.status_code, 200, result_response.get_json())
        result_payload = result_response.get_json()
        self.assertEqual(result_payload["station_id"], "ST-01")
        self.assertEqual(result_payload["sticker_backend"], "classic")
        self.assertIsInstance(result_payload["validation_details"], dict)
        self.assertIsInstance(result_payload["part_ready_roi_meta"], dict)
        self.assertIsInstance(result_payload["sticker_roi_meta"], dict)

        dashboard_response = self.client.get("/dashboard/summary", headers=_headers(self.admin_token))
        self.assertEqual(dashboard_response.status_code, 200)
        summary = dashboard_response.get_json()
        self.assertGreaterEqual(summary["total_inspections"], 2)
        self.assertIn("backend_classic", summary)
        self.assertIn("total_part_ready", summary)
        self.assertGreaterEqual(summary["backend_classic"], 1)

        filtered_summary_response = self.client.get(
            "/dashboard/summary?line_id=LINE-A&station_id=ST-01",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(filtered_summary_response.status_code, 200)
        filtered_summary = filtered_summary_response.get_json()
        self.assertGreaterEqual(filtered_summary["total_inspections"], 1)

        bucket_response = self.client.get(
            "/dashboard/buckets?line_id=LINE-A&station_id=ST-01&granularity=hour",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(bucket_response.status_code, 200)
        buckets = bucket_response.get_json()
        self.assertTrue(buckets)
        self.assertIn("station_id", buckets[0])
        self.assertIn("backend_classic", buckets[0])

    def test_02_part_ready_color_gate_blocks_commit_until_match(self) -> None:
        calibration_response = self.client.post(
            "/calibration/color-profile",
            json={
                "image_b64": _two_roi_ready_image_b64(),
                "colorspace": "LAB",
                "roi": {"x": 0.09, "y": 0.12, "w": 0.19, "h": 0.22},
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(calibration_response.status_code, 200, calibration_response.get_json())
        profile_record = self.client.post(
            "/calibration/profiles",
            json={
                "name": "sample-white-square",
                "profile": calibration_response.get_json(),
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(profile_record.status_code, 201, profile_record.get_json())
        profile_id = profile_record.get_json()["id"]

        template_response = self.client.post(
            "/templates",
            json={
                "name": "Two ROI Gate",
                "description": "Part-ready gate using LAB profile.",
                "is_active": True,
                "camera": {"camera_index": 0, "width": 640, "height": 480, "fps": 15},
                "part_ready_roi": {"x": 0.09, "y": 0.12, "w": 0.19, "h": 0.22},
                "sticker_roi": {"x": 0.35, "y": 0.3, "w": 0.3, "h": 0.32},
                "vision": {"model_path": "models/dummy.pt", "classes": ["sample-sticker"]},
                "part_ready": {
                    "enabled": True,
                    "color_profile_id": profile_id,
                    "colorspace": "LAB",
                    "min_match_ratio": 0.7,
                },
                "sticker": {
                    "part_name": "Sample Part",
                    "expected_class": "sample-sticker",
                    "line": "LINE-B",
                    "enabled": True,
                    "validator_mode": "ml_detection",
                    "min_roi_confidence": 0.0,
                    "max_offset_x": 80,
                    "max_offset_y": 80,
                },
                "persistence": {"write_to_db": True},
                "metadata": {"scenario": "two-roi-gate"},
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(template_response.status_code, 201, template_response.get_json())
        created_template = template_response.get_json()
        self.assertIn("part_ready_roi", created_template)
        self.assertIn("sticker_roi", created_template)

        session_response = self.client.post(
            "/inspection/sessions/start",
            json={
                "client_id": "gate-smoke",
                "camera_index": 0,
                "template_version_id": created_template["version_id"],
                "line_id": "LINE-B",
                "station_id": "ST-02",
            },
            headers=_headers(self.operator_token),
        )
        self.assertEqual(session_response.status_code, 201, session_response.get_json())
        session_payload = session_response.get_json()

        not_ready_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/frame",
            json={"image_b64": _blank_image_b64()},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(not_ready_response.status_code, 200, not_ready_response.get_json())
        not_ready_payload = not_ready_response.get_json()
        self.assertFalse(not_ready_payload["part_ready"]["part_ready"])
        self.assertFalse(not_ready_payload["count_committed"])
        self.assertEqual(not_ready_payload["db_write"]["reason"], "not_committed")

        ready_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/frame",
            json={"image_b64": _two_roi_ready_image_b64()},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(ready_response.status_code, 200, ready_response.get_json())
        ready_payload = ready_response.get_json()
        self.assertTrue(ready_payload["part_ready"]["part_ready"])
        self.assertTrue(ready_payload["count_committed"])
        self.assertEqual(ready_payload["validation"]["decision"], "ACCEPT")
        self.assertEqual(ready_payload["counters"]["session_total"], 1)

    def test_03_validator_prefers_expected_class_candidate(self) -> None:
        from backend.app.core.container import inspection_session_service

        session_response = self.client.post(
            "/inspection/sessions/start",
            json={
                "client_id": "validator-pick",
                "camera_index": 0,
                "template_version_id": 1,
                "line_id": "LINE-A",
                "station_id": "ST-03",
            },
            headers=_headers(self.operator_token),
        )
        self.assertEqual(session_response.status_code, 201, session_response.get_json())
        session_payload = session_response.get_json()

        original_predict = inspection_session_service._sticker_inference.predict

        def fake_predict(*_args, **_kwargs):
            return {
                "backend": "patched",
                "model_path": "patched.pt",
                "meta_path": "patched.meta.json",
                "class_names": ["K0W-HB0", "K1Z-FA0"],
                "fallback_reason": None,
                "detections": [
                    {
                        "label": "K1Z-FA0",
                        "confidence": 0.97,
                        "class_confidence": 0.97,
                        "position": {"x1": 140.0, "y1": 80.0, "x2": 240.0, "y2": 180.0},
                    },
                    {
                        "label": "K0W-HB0",
                        "confidence": 0.72,
                        "class_confidence": 0.72,
                        "position": {"x1": 150.0, "y1": 90.0, "x2": 230.0, "y2": 170.0},
                    },
                ],
            }

        inspection_session_service._sticker_inference.predict = fake_predict
        try:
            frame_response = self.client.post(
                f"/inspection/sessions/{session_payload['session_id']}/frame",
                json={"image_b64": _presence_image_b64()},
                headers=_headers(self.operator_token),
            )
        finally:
            inspection_session_service._sticker_inference.predict = original_predict

        self.assertEqual(frame_response.status_code, 200, frame_response.get_json())
        payload = frame_response.get_json()
        self.assertEqual(payload["validation"]["decision"], "ACCEPT")
        self.assertEqual(payload["validation"]["detected_class"], "K0W-HB0")
        self.assertEqual(payload["validation"]["sticker_backend"], "patched")
        self.assertEqual(payload["validation"]["validation_details"]["candidate_source"], "expected_class")
        self.assertEqual(payload["validation"]["validation_details"]["selected_candidate"]["label"], "K0W-HB0")

    def test_04_validator_returns_out_of_position_deterministically(self) -> None:
        from backend.app.core.container import inspection_session_service

        session_response = self.client.post(
            "/inspection/sessions/start",
            json={
                "client_id": "validator-offset",
                "camera_index": 0,
                "template_version_id": 1,
                "line_id": "LINE-A",
                "station_id": "ST-04",
            },
            headers=_headers(self.operator_token),
        )
        self.assertEqual(session_response.status_code, 201, session_response.get_json())
        session_payload = session_response.get_json()

        original_predict = inspection_session_service._sticker_inference.predict

        def fake_predict(*_args, **_kwargs):
            return {
                "backend": "patched",
                "model_path": "patched.pt",
                "meta_path": "patched.meta.json",
                "class_names": ["K0W-HB0"],
                "fallback_reason": None,
                "detections": [
                    {
                        "label": "K0W-HB0",
                        "confidence": 0.91,
                        "class_confidence": 0.91,
                        "position": {"x1": 320.0, "y1": 40.0, "x2": 382.0, "y2": 120.0},
                    }
                ],
            }

        inspection_session_service._sticker_inference.predict = fake_predict
        try:
            frame_response = self.client.post(
                f"/inspection/sessions/{session_payload['session_id']}/frame",
                json={"image_b64": _presence_image_b64()},
                headers=_headers(self.operator_token),
            )
        finally:
            inspection_session_service._sticker_inference.predict = original_predict

        self.assertEqual(frame_response.status_code, 200, frame_response.get_json())
        payload = frame_response.get_json()
        self.assertEqual(payload["validation"]["decision"], "REJECT")
        self.assertEqual(payload["validation"]["reject_reason_code"], "OUT_OF_POSITION")
        self.assertEqual(payload["validation"]["validation_details"]["candidate_source"], "expected_class")
        self.assertEqual(payload["validation"]["validation_details"]["status"], "out_of_position")

    def test_05_engineer_workstation_endpoints(self) -> None:
        dataset_name = f"dataset-{uuid4().hex[:8]}"
        dataset_response = self.client.post(
            "/datasets",
            json={"name": dataset_name, "description": "smoke"},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(dataset_response.status_code, 201, dataset_response.get_json())
        dataset = dataset_response.get_json()

        upload_response = self.client.post(
            f"/datasets/{dataset['id']}/upload",
            json={
                "file_name": "sample.jpg",
                "target": "images",
                "content_b64": _sample_image_b64(),
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(upload_response.status_code, 201, upload_response.get_json())

        batch_upload_response = self.client.post(
            f"/datasets/{dataset['id']}/upload",
            data=MultiDict(
                [
                    ("target", "images"),
                    ("files", (BytesIO(base64.b64decode(_sample_image_b64())), "batch-1.jpg")),
                    ("files", (BytesIO(base64.b64decode(_blank_image_b64())), "batch-2.jpg")),
                ]
            ),
            content_type="multipart/form-data",
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(batch_upload_response.status_code, 201, batch_upload_response.get_json())
        batch_upload_payload = batch_upload_response.get_json()
        self.assertEqual(batch_upload_payload["count"], 2)
        self.assertEqual(len(batch_upload_payload["items"]), 2)

        invalid_target_response = self.client.post(
            f"/datasets/{dataset['id']}/upload",
            json={
                "file_name": "bad.jpg",
                "target": "not-a-target",
                "content_b64": _sample_image_b64(),
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(invalid_target_response.status_code, 400, invalid_target_response.get_json())

        annotation_response = self.client.post(
            f"/datasets/{dataset['id']}/annotations/sample.jpg",
            json={
                "labels": [
                    {
                        "type": "bbox",
                        "class": "sample-sticker",
                        "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
                    }
                ]
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(annotation_response.status_code, 200, annotation_response.get_json())
        annotation_payload = annotation_response.get_json()
        self.assertEqual(annotation_payload["schema_version"], 1)
        self.assertEqual(annotation_payload["label_count"], 1)

        image_browser_response = self.client.get(
            f"/datasets/{dataset['id']}/files?target=images",
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(image_browser_response.status_code, 200, image_browser_response.get_json())
        image_browser_items = image_browser_response.get_json()
        self.assertTrue(any(item.get("annotation_exists") for item in image_browser_items if item["name"] == "sample.jpg"))

        dataset_list_response = self.client.get("/datasets", headers=_headers(self.engineer_token))
        self.assertEqual(dataset_list_response.status_code, 200, dataset_list_response.get_json())
        dataset_list = dataset_list_response.get_json()
        refreshed_dataset = next(item for item in dataset_list if item["id"] == dataset["id"])
        self.assertEqual(refreshed_dataset["image_count"], 3)
        self.assertEqual(refreshed_dataset["annotated_image_count"], 1)

        version_response = self.client.post(
            f"/datasets/{dataset['id']}/versions",
            json={
                "name": "Smoke snapshot",
                "description": "versioned export",
                "split_ratios": {"train": 0.6, "valid": 0.2, "test": 0.2},
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(version_response.status_code, 201, version_response.get_json())
        version_payload = version_response.get_json()
        self.assertEqual(version_payload["dataset_id"], dataset["id"])
        self.assertTrue(version_payload["display_label"].startswith("v"))
        self.assertTrue(Path(version_payload["export_root"]).exists())

        version_list_response = self.client.get(
            f"/datasets/{dataset['id']}/versions",
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(version_list_response.status_code, 200, version_list_response.get_json())
        version_list = version_list_response.get_json()
        self.assertEqual(len(version_list), 1)
        self.assertEqual(version_list[0]["id"], version_payload["id"])

        version_detail_response = self.client.get(
            f"/datasets/{dataset['id']}/versions/{version_payload['id']}",
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(version_detail_response.status_code, 200, version_detail_response.get_json())
        self.assertEqual(version_detail_response.get_json()["id"], version_payload["id"])

        version_export_response = self.client.post(
            f"/datasets/{dataset['id']}/versions/{version_payload['id']}/export",
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(version_export_response.status_code, 200, version_export_response.get_json())
        self.assertEqual(version_export_response.get_json()["id"], version_payload["id"])

        augment_response = self.client.post(
            "/augment/jobs",
            json={"dataset_id": dataset["id"]},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(augment_response.status_code, 201, augment_response.get_json())

        train_response = self.client.post(
            "/train/jobs",
            json={"dataset_id": dataset["id"], "dataset_version_id": version_payload["id"], "base_model": "yolov5s"},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(train_response.status_code, 201, train_response.get_json())
        train_payload = train_response.get_json()
        self.assertEqual(train_payload["requested_device_mode"], "auto")
        self.assertEqual(train_payload["params"]["device_mode"], "auto")
        self.assertEqual(train_payload["base_model"], "yolov5s")
        self.assertEqual(train_payload["base_model_catalog_id"], "yolov5s")
        self.assertEqual(train_payload["base_model_family"], "yolov5")
        self.assertEqual(train_payload["base_model_variant"], "s")
        self.assertEqual(train_payload["base_model_display_name"], "YOLOv5 Small")
        self.assertEqual(train_payload["dataset_version_id"], version_payload["id"])
        self.assertEqual(train_payload["dataset_version_display_label"], version_payload["display_label"])

        deadline = time.time() + 10.0
        completed_job = None
        while time.time() < deadline:
            jobs_response = self.client.get("/train/jobs", headers=_headers(self.engineer_token))
            self.assertEqual(jobs_response.status_code, 200, jobs_response.get_json())
            jobs = jobs_response.get_json()
            completed_job = next((item for item in jobs if item.get("id") == train_payload["id"]), None)
            if completed_job and completed_job.get("status") == "completed" and completed_job.get("registered_model_id") is not None:
                break
            time.sleep(0.2)

        self.assertIsNotNone(completed_job)
        self.assertEqual(completed_job.get("status"), "completed")
        self.assertIsNotNone(completed_job.get("registered_model_id"))

        models_response = self.client.get("/models", headers=_headers(self.engineer_token))
        self.assertEqual(models_response.status_code, 200, models_response.get_json())
        models = models_response.get_json()
        self.assertTrue(any(item.get("provenance", {}).get("training_job_id") == train_payload["id"] for item in models))

        model_response = self.client.post(
            "/models",
            json={"name": "smoke-model", "path": "models/smoke.pt", "source": "manual"},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(model_response.status_code, 201, model_response.get_json())

        calibration_response = self.client.post(
            "/calibration/color-profile",
            json={"image_b64": _sample_image_b64(), "colorspace": "LAB"},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(calibration_response.status_code, 200, calibration_response.get_json())
        profile_payload = calibration_response.get_json()
        self.assertEqual(profile_payload["colorspace"], "LAB")

    def test_05a_engineer_dataset_download_and_delete(self) -> None:
        dataset_name = f"dataset-delete-{uuid4().hex[:8]}"
        dataset_response = self.client.post(
            "/datasets",
            json={"name": dataset_name, "description": "delete smoke"},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(dataset_response.status_code, 201, dataset_response.get_json())
        dataset = dataset_response.get_json()
        dataset_dir = TEST_DATA_ROOT / "datasets" / str(dataset["id"])

        upload_response = self.client.post(
            f"/datasets/{dataset['id']}/upload",
            json={
                "file_name": "delete-me.jpg",
                "target": "images",
                "content_b64": _sample_image_b64(),
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(upload_response.status_code, 201, upload_response.get_json())

        download_response = self.client.get(
            f"/datasets/{dataset['id']}/files/images/delete-me.jpg",
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(download_response.status_code, 200, download_response.get_json())
        self.assertGreater(len(download_response.data), 0)
        download_response.close()

        delete_response = self.client.delete(f"/datasets/{dataset['id']}", headers=_headers(self.engineer_token))
        self.assertEqual(delete_response.status_code, 200, delete_response.get_json())
        self.assertTrue(delete_response.get_json()["deleted"])
        self.assertFalse(dataset_dir.exists())

        dataset_list_response = self.client.get("/datasets", headers=_headers(self.engineer_token))
        self.assertEqual(dataset_list_response.status_code, 200, dataset_list_response.get_json())
        dataset_list = dataset_list_response.get_json()
        self.assertFalse(any(item["id"] == dataset["id"] for item in dataset_list))

    def test_06_session_accepts_independent_dual_roi_updates(self) -> None:
        session_response = self.client.post(
            "/inspection/sessions/start",
            json={
                "client_id": "roi-update",
                "camera_index": 0,
                "template_version_id": 1,
                "line_id": "LINE-ROI",
                "station_id": "ST-ROI",
            },
            headers=_headers(self.operator_token),
        )
        self.assertEqual(session_response.status_code, 201, session_response.get_json())
        session_payload = session_response.get_json()

        roi_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/roi",
            json={
                "part_ready_roi": {"x": 0.05, "y": 0.08, "w": 0.10, "h": 0.12},
                "sticker_roi": {"x": 0.60, "y": 0.25, "w": 0.20, "h": 0.30},
            },
            headers=_headers(self.operator_token),
        )
        self.assertEqual(roi_response.status_code, 200, roi_response.get_json())
        roi_payload = roi_response.get_json()
        self.assertEqual(roi_payload["part_ready_roi"]["x"], 0.05)
        self.assertEqual(roi_payload["part_ready_roi"]["w"], 0.10)
        self.assertEqual(roi_payload["sticker_roi"]["x"], 0.60)
        self.assertEqual(roi_payload["sticker_roi"]["h"], 0.30)
        self.assertNotEqual(roi_payload["part_ready_roi"]["x"], roi_payload["sticker_roi"]["x"])

        frame_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/frame",
            json={"image_b64": _blank_image_b64()},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(frame_response.status_code, 200, frame_response.get_json())
        frame_payload = frame_response.get_json()
        self.assertEqual(frame_payload["part_ready_roi_meta"]["x"], 32)
        self.assertEqual(frame_payload["part_ready_roi_meta"]["width"], 64)
        self.assertEqual(frame_payload["sticker_roi_meta"]["x"], 384)
        self.assertEqual(frame_payload["sticker_roi_meta"]["width"], 128)

    def test_07_admin_template_versioning_deployment_lifecycle_and_user_activation(self) -> None:
        create_response = self.client.post(
            "/templates",
            json={
                "name": "Phase8 Admin Template",
                "description": "created by phase 8 regression",
                "is_active": True,
                "camera": {"camera_index": 1, "width": 640, "height": 480, "fps": 15},
                "part_ready_roi": {"x": 0.11, "y": 0.12, "w": 0.20, "h": 0.18},
                "sticker_roi": {"x": 0.41, "y": 0.32, "w": 0.28, "h": 0.20},
                "vision": {
                    "model_path": "models/admin-template.pt",
                    "model_meta_path": "models/admin-template.meta.json",
                    "runtime": "ultralytics",
                    "conf_threshold": 0.31,
                    "stream_fps": 11,
                    "inference_fps": 5,
                    "imgsz": 640,
                    "classes": ["K0W-HB0", "K1Z-FA0"],
                },
                "part_ready": {
                    "enabled": True,
                    "color_profile_id": None,
                    "colorspace": "LAB",
                    "distance_threshold": 13.0,
                    "min_match_ratio": 0.81,
                },
                "sticker": {
                    "part_name": "Phase8 Part",
                    "expected_class": "K0W-HB0",
                    "line": "LINE-ADMIN",
                    "enabled": True,
                    "validator_mode": "ml_detection",
                    "min_roi_confidence": 0.2,
                    "min_class_confidence": 0.55,
                    "max_offset_x": 24,
                    "max_offset_y": 18,
                },
                "persistence": {"write_to_db": True},
                "metadata": {"family": "phase8"},
            },
            headers=_headers(self.admin_token),
        )
        self.assertEqual(create_response.status_code, 201, create_response.get_json())
        created = create_response.get_json()
        self.assertEqual(created["version_number"], 1)

        update_response = self.client.put(
            f"/templates/{created['id']}",
            json={
                **created,
                "description": "updated by phase 8 regression",
                "part_ready_roi": {"x": 0.15, "y": 0.18, "w": 0.18, "h": 0.14},
                "sticker_roi": {"x": 0.48, "y": 0.22, "w": 0.22, "h": 0.24},
                "part_ready": {
                    **created["part_ready"],
                    "min_match_ratio": 0.9,
                },
                "sticker": {
                    **created["sticker"],
                    "max_offset_x": 12,
                    "max_offset_y": 12,
                },
            },
            headers=_headers(self.admin_token),
        )
        self.assertEqual(update_response.status_code, 200, update_response.get_json())
        updated = update_response.get_json()
        self.assertGreater(updated["version_id"], created["version_id"])
        self.assertEqual(updated["version_number"], 2)
        self.assertEqual(updated["part_ready_roi"]["x"], 0.15)
        self.assertEqual(updated["sticker_roi"]["x"], 0.48)
        self.assertEqual(updated["part_ready"]["min_match_ratio"], 0.9)

        list_response = self.client.get("/templates", headers=_headers(self.admin_token))
        self.assertEqual(list_response.status_code, 200, list_response.get_json())
        summaries = list_response.get_json()
        summary = next(item for item in summaries if int(item["id"]) == int(created["id"]))
        self.assertEqual(summary["version_number"], 2)
        self.assertEqual(summary["version_id"], updated["version_id"])

        deploy_response = self.client.post(
            "/deployments",
            json={
                "template_id": created["id"],
                "template_version_id": updated["version_id"],
                "line_id": "LINE-ADMIN",
                "station_id": "ST-ADMIN",
            },
            headers=_headers(self.admin_token),
        )
        self.assertEqual(deploy_response.status_code, 201, deploy_response.get_json())
        deployment = deploy_response.get_json()

        active_response = self.client.get(
            "/deployments/active?line_id=LINE-ADMIN&station_id=ST-ADMIN",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(active_response.status_code, 200, active_response.get_json())
        self.assertEqual(active_response.get_json()["deployment"]["template_version_id"], updated["version_id"])

        deactivate_response = self.client.delete(
            f"/deployments/{deployment['id']}",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(deactivate_response.status_code, 200, deactivate_response.get_json())

        active_after_response = self.client.get(
            "/deployments/active?line_id=LINE-ADMIN&station_id=ST-ADMIN",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(active_after_response.status_code, 200, active_after_response.get_json())
        self.assertIsNone(active_after_response.get_json()["deployment"])

        new_username = f"user_{uuid4().hex[:8]}"
        create_user_response = self.client.post(
            "/auth/users",
            json={"username": new_username, "password": "phase8pass", "role": "operator"},
            headers=_headers(self.admin_token),
        )
        self.assertEqual(create_user_response.status_code, 201, create_user_response.get_json())
        user = create_user_response.get_json()
        self.assertNotIn("password_hash", user)

        disable_response = self.client.put(
            f"/auth/users/{user['id']}",
            json={"is_active": False},
            headers=_headers(self.admin_token),
        )
        self.assertEqual(disable_response.status_code, 200, disable_response.get_json())
        disabled = disable_response.get_json()
        self.assertFalse(disabled["is_active"])
        self.assertNotIn("password_hash", disabled)

        login_disabled_response = self.client.post("/auth/login", json={"username": new_username, "password": "phase8pass"})
        self.assertEqual(login_disabled_response.status_code, 401, login_disabled_response.get_json())

        enable_response = self.client.put(
            f"/auth/users/{user['id']}",
            json={"is_active": True},
            headers=_headers(self.admin_token),
        )
        self.assertEqual(enable_response.status_code, 200, enable_response.get_json())
        self.assertTrue(enable_response.get_json()["is_active"])

        login_enabled_response = self.client.post("/auth/login", json={"username": new_username, "password": "phase8pass"})
        self.assertEqual(login_enabled_response.status_code, 200, login_enabled_response.get_json())

    def test_08_engineer_metadata_roundtrip_and_filtered_queries(self) -> None:
        dataset_name = f"phase8-dataset-{uuid4().hex[:8]}"
        dataset_response = self.client.post(
            "/datasets",
            json={"name": dataset_name, "description": "phase8 metadata"},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(dataset_response.status_code, 201, dataset_response.get_json())
        dataset = dataset_response.get_json()

        model_name = f"phase8-model-{uuid4().hex[:8]}"
        model_response = self.client.post(
            "/models",
            json={
                "name": model_name,
                "path": "models/phase8.pt",
                "meta_path": "models/phase8.meta.json",
                "source": "manual",
                "runtime": "ultralytics",
                "task": "detection",
                "class_names": ["K0W-HB0", "K1Z-FA0"],
                "architecture_family": "yolov5",
                "architecture_variant": "mu",
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(model_response.status_code, 201, model_response.get_json())
        model_payload = model_response.get_json()
        self.assertEqual(model_payload["runtime"], "ultralytics")
        self.assertEqual(model_payload["class_names"], ["K0W-HB0", "K1Z-FA0"])
        self.assertEqual(model_payload["architecture_variant"], "mu")

        list_models_response = self.client.get("/models", headers=_headers(self.engineer_token))
        self.assertEqual(list_models_response.status_code, 200, list_models_response.get_json())
        list_models = list_models_response.get_json()
        created_model = next(item for item in list_models if item["name"] == model_name)
        self.assertEqual(created_model["meta_path"], "models/phase8.meta.json")

        compute_response = self.client.post(
            "/calibration/color-profile",
            json={
                "image_b64": _two_roi_ready_image_b64(),
                "colorspace": "LAB",
                "roi": {"x": 0.09, "y": 0.12, "w": 0.19, "h": 0.22},
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(compute_response.status_code, 200, compute_response.get_json())
        save_profile_response = self.client.post(
            "/calibration/profiles",
            json={"name": "phase8-profile", "profile": compute_response.get_json()},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(save_profile_response.status_code, 201, save_profile_response.get_json())
        profile_id = save_profile_response.get_json()["id"]

        list_profiles_response = self.client.get("/calibration/profiles", headers=_headers(self.engineer_token))
        self.assertEqual(list_profiles_response.status_code, 200, list_profiles_response.get_json())
        profiles = list_profiles_response.get_json()
        self.assertTrue(any(int(item["id"]) == int(profile_id) for item in profiles))

        delete_profile_response = self.client.delete(
            f"/calibration/profiles/{profile_id}",
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(delete_profile_response.status_code, 200, delete_profile_response.get_json())

        profiles_after_response = self.client.get("/calibration/profiles", headers=_headers(self.engineer_token))
        self.assertEqual(profiles_after_response.status_code, 200, profiles_after_response.get_json())
        profiles_after = profiles_after_response.get_json()
        self.assertFalse(any(int(item["id"]) == int(profile_id) for item in profiles_after))

        session_response = self.client.post(
            "/inspection/sessions/start",
            json={
                "client_id": "phase8-filter",
                "camera_index": 0,
                "template_version_id": 1,
                "line_id": "LINE-FILTER",
                "station_id": "ST-FILTER",
            },
            headers=_headers(self.operator_token),
        )
        self.assertEqual(session_response.status_code, 201, session_response.get_json())
        session_payload = session_response.get_json()
        frame_response = self.client.post(
            f"/inspection/sessions/{session_payload['session_id']}/frame",
            json={"image_b64": _sample_image_b64()},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(frame_response.status_code, 200, frame_response.get_json())
        self.assertEqual(frame_response.get_json()["validation"]["decision"], "ACCEPT")

        filtered_results_response = self.client.get(
            "/inspections?line_id=LINE-FILTER&station_id=ST-FILTER&decision_code=ACCEPT&template_version_id=1",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(filtered_results_response.status_code, 200, filtered_results_response.get_json())
        filtered_results = filtered_results_response.get_json()
        self.assertTrue(filtered_results)
        self.assertTrue(
            all(
                item.get("line_id") == "LINE-FILTER"
                and item.get("station_id") == "ST-FILTER"
                and item.get("decision_code") == "ACCEPT"
                for item in filtered_results
            )
        )

    def test_08a_training_job_request_records_metadata(self) -> None:
        dataset_name = f"phase8-train-{uuid4().hex[:8]}"
        dataset_response = self.client.post(
            "/datasets",
            json={"name": dataset_name, "description": "phase8 train metadata"},
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(dataset_response.status_code, 201, dataset_response.get_json())
        dataset = dataset_response.get_json()

        training_response = self.client.post(
            "/train/jobs",
            json={
                "dataset_id": dataset["id"],
                "base_model": "yolov11m",
                "device_mode": "cpu",
                "classes": ["K0W-HB0", "K1Z-FA0"],
                "note": "phase8 regression",
            },
            headers=_headers(self.engineer_token),
        )
        self.assertEqual(training_response.status_code, 201, training_response.get_json())
        training_payload = training_response.get_json()
        self.assertEqual(training_payload["params"]["classes"], ["K0W-HB0", "K1Z-FA0"])
        self.assertEqual(training_payload["requested_device_mode"], "cpu")
        self.assertEqual(training_payload["params"]["device_mode"], "cpu")
        self.assertEqual(training_payload["base_model"], "yolov11m")
        self.assertEqual(training_payload["base_model_family"], "yolov11")
        self.assertEqual(training_payload["base_model_variant"], "m")
        self.assertEqual(training_payload["base_model_display_name"], "YOLOv11 Medium")

        filtered_dashboard_response = self.client.get(
            "/dashboard/summary?line_id=LINE-FILTER&station_id=ST-FILTER&template_version_id=1",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(filtered_dashboard_response.status_code, 200, filtered_dashboard_response.get_json())
        filtered_summary = filtered_dashboard_response.get_json()
        self.assertGreaterEqual(filtered_summary["total_inspections"], 1)

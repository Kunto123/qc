from __future__ import annotations

import unittest

import numpy as np

try:
    import tkinter as tk
except Exception:  # noqa: BLE001
    tk = None

from client_tk.app.screens.admin.view import AdminScreen
from client_tk.app.screens.engineer.view import EngineerScreen
from client_tk.app.screens.operator.view import OperatorScreen
from client_tk.app.services.session_state import SessionState


class _StubApi:
    def list_templates(self):
        return [{"id": 1, "name": "QC Line A", "version_id": 1, "version_number": 1}]

    def get_template(self, template_id: int):
        return {
            "id": template_id,
            "version_id": 1,
            "version_number": 1,
            "name": "QC Line A",
            "description": "stub",
            "is_active": True,
            "camera": {"camera_index": 0, "width": 640, "height": 480, "fps": 15},
            "part_ready_roi": {"x": 0.2, "y": 0.2, "w": 0.25, "h": 0.25},
            "sticker_roi": {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6},
            "vision": {
                "model_path": "models/dummy.pt",
                "model_meta_path": None,
                "runtime": "ultralytics",
                "conf_threshold": 0.25,
                "stream_fps": 10,
                "inference_fps": 4,
                "imgsz": 640,
                "classes": ["K0W-HB0"],
            },
            "part_ready": {
                "enabled": True,
                "color_profile_id": None,
                "colorspace": "LAB",
                "distance_threshold": None,
                "min_match_ratio": 0.75,
            },
            "sticker": {
                "part_name": "Sample Part",
                "expected_class": "K0W-HB0",
                "line": "LINE-A",
                "enabled": True,
                "validator_mode": "ml_detection",
                "min_roi_confidence": 0.0,
                "min_class_confidence": None,
                "max_offset_x": 80,
                "max_offset_y": 80,
            },
            "persistence": {"write_to_db": True},
            "metadata": {},
        }

    def list_models(self):
        return []

    def list_profiles(self):
        return []

    def list_deployments(self):
        return []

    def list_users(self):
        return []

    def list_inspections(self, params=None):
        return []

    def retry_inspection_push(self, result_id: int):
        return {"ok": True, "result": {"id": result_id, "push_status": "sent"}}

    def retry_failed_inspection_pushes(self, result_ids=None, limit: int = 100):
        return {"attempted": 0, "succeeded": 0, "failed": 0, "items": []}

    def dashboard_summary(self, params=None):
        return {}

    def dashboard_buckets(self, params=None):
        return []

    def list_datasets(self):
        return []

    def list_augment_jobs(self):
        return []

    def list_training_jobs(self):
        return []


@unittest.skipIf(tk is None, "Tkinter is not available in this environment")
class UiSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk root unavailable: {exc}")
        self.root.withdraw()
        self.api = _StubApi()
        self.state = SessionState(base_url="http://127.0.0.1:8100")
        self.state.user = {"id": 1, "username": "tester"}

    def tearDown(self) -> None:
        if getattr(self, "root", None) is not None:
            self.root.update_idletasks()
            self.root.destroy()

    def test_operator_screen_initializes(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        self.assertTrue(screen.winfo_exists())
        self.assertEqual(str(screen.template_selector["state"]), "readonly")
        self.assertTrue(screen.template_context.get().startswith("Template: -"))
        screen.destroy()

    def test_operator_local_dual_views_refresh(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        frame[:, :] = (20, 90, 180)
        screen.part_ready_roi_x_value.set("0.1")
        screen.part_ready_roi_y_value.set("0.1")
        screen.part_ready_roi_w_value.set("0.2")
        screen.part_ready_roi_h_value.set("0.3")
        screen.sticker_roi_x_value.set("0.4")
        screen.sticker_roi_y_value.set("0.2")
        screen.sticker_roi_w_value.set("0.3")
        screen.sticker_roi_h_value.set("0.4")
        screen._update_local_roi_previews(frame)
        screen.update_idletasks()
        self.assertIsNotNone(screen.part_ready_preview._photo)
        self.assertIsNotNone(screen.main_view._photo)
        self.assertIn("Sticker ROI", screen.display_source.get())
        screen.destroy()

    def test_admin_screen_initializes(self) -> None:
        screen = AdminScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        self.assertTrue(screen.winfo_exists())
        self.assertIsNotNone(screen.template_form)
        screen.destroy()

    def test_engineer_screen_initializes(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        self.assertTrue(screen.winfo_exists())
        self.assertIsNotNone(screen.training_summary)
        screen.destroy()

    def test_engineer_calibration_roi_preview_refreshes(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen.calibration_image = np.zeros((120, 200, 3), dtype=np.uint8)
        screen.calibration_image[:, :] = (10, 120, 220)
        for entry, value in (
            (screen.calib_roi_x, "0.1"),
            (screen.calib_roi_y, "0.2"),
            (screen.calib_roi_w, "0.3"),
            (screen.calib_roi_h, "0.4"),
        ):
            entry.delete(0, "end")
            entry.insert(0, value)
        screen._refresh_calibration_preview()
        screen.update_idletasks()
        self.assertIn("crop", screen.calibration_preview_info.get().lower())
        self.assertIsNotNone(screen.calibration_source_preview._photo)
        self.assertIsNotNone(screen.calibration_crop_preview._photo)
        screen.destroy()

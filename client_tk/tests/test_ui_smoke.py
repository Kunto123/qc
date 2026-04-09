from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import cv2
from tkinter import ttk

try:
    import tkinter as tk
except Exception:  # noqa: BLE001
    tk = None

from client_tk.app.screens.admin.view import AdminScreen
from client_tk.app.components.live_view import LiveView
from client_tk.app.components.annotation_canvas import AnnotationCanvas
from client_tk.app.screens.engineer.view import EngineerScreen
from client_tk.app.screens.operator.view import OperatorScreen
from client_tk.app.components.scrollable_frame import ScrollableFrame, _dispatch_mousewheel
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
    def list_dataset_versions(self, dataset_id: str):
        return []

    def create_dataset_version(self, dataset_id: str, payload: dict):
        return {}

    def export_dataset_version(self, dataset_id: str, version_id: str):
        return {}

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

    def list_dataset_files(self, dataset_id: str, target: str = "images"):
        return []

    def get_annotation(self, dataset_id: str, image_name: str):
        return {"labels": []}

    def save_annotation(self, dataset_id: str, image_name: str, labels: list[dict]):
        return {"labels": labels}

    def delete_dataset(self, dataset_id: str):
        return {"deleted": True, "id": dataset_id}

    def download_dataset_image(self, dataset_id: str, image_name: str):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        if not ok:
            return b""
        return buffer.tobytes()

    def list_augment_jobs(self):
        return []

    def list_training_jobs(self):
        return []

    def create_training_job(self, payload: dict):
        return {"id": "train-stub", **payload}

    def list_base_models(self):
        return [
            {"id": "yolov5s", "display_label": "YOLOv5 Small (yolov5s)", "display_name": "YOLOv5 Small", "family": "yolov5", "family_label": "YOLOv5", "variant": "s", "variant_label": "Small", "runtime": "ultralytics", "weights_name": "yolov5s.pt"},
            {"id": "yolov11m", "display_label": "YOLOv11 Medium (yolov11m)", "display_name": "YOLOv11 Medium", "family": "yolov11", "family_label": "YOLOv11", "variant": "m", "variant_label": "Medium", "runtime": "ultralytics", "weights_name": "yolov11m.pt"},
        ]


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
        self._async_patchers = [
            mock.patch("client_tk.app.screens.admin.view.run_async", new=self._run_async_sync),
        ]
        for patcher in self._async_patchers:
            patcher.start()

    def _run_async_sync(self, widget, func, *, callback=None, args=(), kwargs=None):
        try:
            result = func(*args, **(kwargs or {}))
        except Exception as exc:  # noqa: BLE001
            if callback is not None:
                callback(None, exc)
            return None
        if callback is not None:
            callback(result, None)
        return None

    def tearDown(self) -> None:
        for patcher in getattr(self, "_async_patchers", []):
            patcher.stop()
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

    def test_operator_layout_switches_to_compact(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(screen, "winfo_width", return_value=1000), mock.patch.object(
            screen.winfo_toplevel(),
            "winfo_width",
            return_value=1000,
        ):
            screen._apply_responsive_layout()
        self.assertTrue(screen._is_compact_layout)
        self.assertEqual(int(screen.action_buttons[0].grid_info()["row"]), 0)
        self.assertEqual(int(screen.action_buttons[3].grid_info()["row"]), 1)
        self.assertEqual(int(screen.template_box.grid_info()["row"]), 1)
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

    def test_admin_layout_switches_to_compact(self) -> None:
        screen = AdminScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(screen, "winfo_width", return_value=1000), mock.patch.object(
            screen.winfo_toplevel(),
            "winfo_width",
            return_value=1000,
        ):
            screen._apply_responsive_layout()
        self.assertTrue(screen._layout_compact)
        self.assertEqual(int(screen.templates_left.grid_info()["row"]), 0)
        self.assertEqual(int(screen.templates_right.grid_info()["row"]), 1)
        self.assertEqual(int(screen.deployments_left.grid_info()["row"]), 0)
        self.assertEqual(int(screen.deployments_right.grid_info()["row"]), 1)
        self.assertEqual(int(screen.users_left.grid_info()["row"]), 0)
        self.assertEqual(int(screen.users_right.grid_info()["row"]), 1)
        self.assertEqual(int(screen.results_left.grid_info()["row"]), 0)
        self.assertEqual(int(screen.results_right.grid_info()["row"]), 1)
        self.assertEqual(int(screen.admin_cards["templates"].grid_info()["row"]), 0)
        self.assertEqual(int(screen.admin_cards["users"].grid_info()["row"]), 1)
        screen.destroy()

    def test_admin_overview_cards_hide_on_short_height(self) -> None:
        screen = AdminScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(screen, "winfo_height", return_value=700):
            screen._apply_responsive_layout()
        self.assertFalse(screen._overview_cards_visible)
        screen.destroy()

    def test_engineer_screen_initializes(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        self.assertTrue(screen.winfo_exists())
        self.assertIsNotNone(screen.training_summary)
        self.assertIsNotNone(screen.annotation_canvas)
        self.assertEqual(str(screen.annot_dataset["state"]), "readonly")
        self.assertEqual(screen.train_device.get(), "auto")
        self.assertEqual(str(screen.train_device["state"]), "readonly")
        self.assertTrue(screen.train_base_model.get())
        self.assertEqual(str(screen.train_base_model["state"]), "readonly")
        self.assertEqual(str(screen.train_dataset_version["state"]), "readonly")
        self.assertEqual(str(screen.annot_shape["state"]), "readonly")
        self.assertEqual(str(screen.annot_class["state"]), "normal")
        self.assertEqual(str(screen.annot_apply_class_button["state"]), "disabled")
        screen.destroy()

    def test_engineer_training_summary_shows_key_metrics(self) -> None:
        jobs = [
            {
                "id": "train-001",
                "dataset_id": "ds-train",
                "status": "completed",
                "base_model": "yolov11m",
                "base_model_display_name": "YOLOv11 Medium",
                "metrics": {
                    "accuracy": 0.9234,
                    "map": 0.8123,
                },
                "evaluation": {
                    "r2": 0.7845,
                    "rmse": 0.0456,
                },
                "trained_model_path": "models/trained/train-001.pt",
            }
        ]

        with mock.patch.object(self.api, "list_training_jobs", return_value=jobs):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            screen.train_jobs.selection_clear(0, "end")
            screen.train_jobs.selection_set(0)
            screen.on_training_selected()

            self.assertEqual(screen.training_summary._labels["base_model"].cget("text"), "YOLOv11 Medium")
            self.assertEqual(screen.training_summary._labels["status"].cget("text"), "completed")
            self.assertEqual(screen.training_summary._labels["accuracy"].cget("text"), "92.34%")
            self.assertEqual(screen.training_summary._labels["map_score"].cget("text"), "81.23%")
            self.assertEqual(screen.training_summary._labels["r2_score"].cget("text"), "0.785")
            self.assertEqual(screen.training_summary._labels["error"].cget("text"), "RMSE 0.0456")

            screen.destroy()

    def test_engineer_training_job_uses_active_dataset_version_when_combo_is_blank(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        captured_payloads: list[dict] = []

        def create_training_job(payload: dict):
            captured_payloads.append(dict(payload))
            return {"id": "train-x", **payload}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-vers", "name": "Dataset Versioned"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[
                {
                    "id": "ver-1",
                    "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                    "version_number": 1,
                    "name": "Snapshot v1",
                    "status": "ready",
                    "export_format": "yolo",
                    "export_root": "data/export/ver-1",
                    "image_count": 1,
                    "annotated_image_count": 1,
                    "coverage_percent": 100.0,
                    "class_names": ["K0W-HB0"],
                }
            ],
        ), mock.patch.object(
            self.api,
            "create_training_job",
            side_effect=create_training_job,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            screen.train_dataset_version.set("")
            screen._active_dataset_version_id = "ver-1"
            screen.create_training_job()

            self.assertTrue(captured_payloads)
            self.assertEqual(captured_payloads[-1].get("dataset_version_id"), "ver-1")

            screen.destroy()

    def test_engineer_annotation_dataset_follows_selected_dataset(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen._dataset_cache = [{"id": "ds-123", "name": "Sample Dataset"}]
        screen.dataset_list.delete(0, "end")
        screen.dataset_list.insert("end", "ds-123 | Sample Dataset | 0 imgs / 0 ann / 0 aug")
        screen._sync_annotation_dataset_selector()
        screen.dataset_list.selection_clear(0, "end")
        screen.dataset_list.selection_set(0)
        screen.on_dataset_selected()
        self.assertEqual(screen.annot_dataset_var.get(), "ds-123")
        self.assertEqual(screen._annotation_dataset_id, "ds-123")
        self.assertEqual(str(screen.annot_dataset["state"]), "readonly")
        self.assertEqual(tuple(screen.annot_dataset["values"]), ("ds-123",))
        screen.destroy()

    def test_engineer_annotation_dataset_dropdown_selects_dataset(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen._dataset_cache = [
            {"id": "ds-1", "name": "Dataset One"},
            {"id": "ds-2", "name": "Dataset Two"},
        ]
        screen.dataset_list.delete(0, "end")
        screen.dataset_list.insert("end", "ds-1 | Dataset One | 0 imgs / 0 ann / 0 aug")
        screen.dataset_list.insert("end", "ds-2 | Dataset Two | 0 imgs / 0 ann / 0 aug")
        screen._sync_annotation_dataset_selector()

        self.assertEqual(tuple(screen.annot_dataset["values"]), ("ds-1", "ds-2"))

        screen.annot_dataset_var.set("ds-2")
        screen._on_annotation_dataset_selected()

        self.assertEqual(screen._annotation_dataset_id, "ds-2")
        self.assertEqual(screen.annot_dataset_var.get(), "ds-2")
        self.assertEqual(screen.upload_dataset_id.get().strip(), "ds-2")
        self.assertEqual(screen._selected_dataset_id(), "ds-2")
        screen.destroy()

    def test_engineer_empty_dataset_selection_keeps_annotation_context(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-safe", "name": "Dataset Safe"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            self.assertEqual(screen._annotation_dataset_id, "ds-safe")
            self.assertEqual(screen.annot_dataset_var.get(), "ds-safe")

            screen.dataset_list.selection_clear(0, "end")
            screen.on_dataset_selected()
            screen.update_idletasks()

            self.assertEqual(screen._annotation_dataset_id, "ds-safe")
            self.assertEqual(screen.annot_dataset_var.get(), "ds-safe")
            self.assertEqual(screen.annot_image_var.get(), "sample.png")
            self.assertIsNotNone(screen.annotation_canvas._source_frame)
            self.assertIsNotNone(screen.annotation_canvas._photo)
            self.assertEqual(screen.dataset_list.curselection(), (0,))

        screen.destroy()

    def test_engineer_annotation_toolbar_changes_keep_context_visible(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-keep", "name": "Dataset Keep"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            self.assertEqual(screen.annot_dataset_var.get(), "ds-keep")
            self.assertEqual(screen._selected_dataset_id(), "ds-keep")
            self.assertEqual(screen.annot_image_var.get(), "sample.png")
            self.assertIsNotNone(screen.annotation_canvas._photo)

            screen.annot_shape.set("polygon")
            screen._sync_annotation_mode()
            screen.update_idletasks()
            self.assertEqual(screen.annot_dataset_var.get(), "ds-keep")
            self.assertEqual(screen._selected_dataset_id(), "ds-keep")
            self.assertIsNotNone(screen.annotation_canvas._photo)

            screen.annot_class_var.set("manual-class")
            screen._on_annotation_class_input()
            screen.update_idletasks()
            self.assertEqual(screen.annot_dataset_var.get(), "ds-keep")
            self.assertEqual(screen._selected_dataset_id(), "ds-keep")
            self.assertIsNotNone(screen.annotation_canvas._photo)

        screen.destroy()

    def test_engineer_manual_save_annotation_button_persists_current_labels(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        save_calls: list[tuple[str, str, list[dict]]] = []

        def save_annotation(dataset_id: str, image_name: str, labels: list[dict]):
            save_calls.append((dataset_id, image_name, labels))
            return {"labels": labels}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-save", "name": "Dataset Save"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ), mock.patch.object(
            self.api,
            "save_annotation",
            side_effect=save_annotation,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            save_calls.clear()
            screen.annotation_canvas._labels = [
                {
                    "type": "bbox",
                    "shape_type": "bbox",
                    "class_name": "manual-save",
                    "class": "manual-save",
                    "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                    "normalized": True,
                }
            ]
            screen.save_current_annotation()

            self.assertTrue(save_calls)
            self.assertEqual(save_calls[-1][0], "ds-save")
            self.assertEqual(save_calls[-1][1], "sample.png")
            self.assertEqual(save_calls[-1][2][0].get("class_name"), "manual-save")
            self.assertEqual(screen.annotation_status.get(), "Saved sample.png")

        screen.destroy()

    def test_engineer_annotation_class_input_auto_applies_to_selected_label(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        save_calls: list[list[dict]] = []

        def save_annotation(dataset_id: str, image_name: str, labels: list[dict]):
            save_calls.append(labels)
            return {"labels": labels}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-auto", "name": "Dataset Auto"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={
                "labels": [
                    {
                        "type": "bbox",
                        "shape_type": "bbox",
                        "class_name": "object",
                        "class": "object",
                        "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                        "normalized": True,
                    }
                ]
            },
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ), mock.patch.object(
            self.api,
            "save_annotation",
            side_effect=save_annotation,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            self.assertTrue(screen.annotation_canvas.set_selected_label_index(0))
            screen.annot_class_var.set("K0W-HB0")
            screen._on_annotation_class_input()
            screen.update_idletasks()

            self.assertTrue(save_calls)
            self.assertEqual(screen.annotation_canvas.get_labels()[0].get("class_name"), "K0W-HB0")
            self.assertEqual(screen.annotation_canvas.get_selected_label_index(), 0)
            self.assertEqual(screen.annot_class_var.get(), "K0W-HB0")

        screen.destroy()

    def test_engineer_refresh_auto_selects_first_dataset(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-abc", "name": "Dataset A"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            wraps=self.api.download_dataset_image,
        ) as download_mock:
            screen.refresh_datasets()
        self.assertEqual(screen.annot_dataset_var.get(), "ds-abc")
        self.assertEqual(screen.annot_image_var.get(), "sample.png")
        self.assertEqual(screen._annotation_dataset_id, "ds-abc")
        self.assertIsNotNone(screen.annotation_canvas._source_frame)
        self.assertIsNotNone(screen.annotation_canvas._photo)
        self.assertIn("loaded via backend", screen.annotation_status.get())
        download_mock.assert_called_once_with("ds-abc", "sample.png")
        screen.destroy()

    def test_engineer_data_scroll_event_restores_annotation_canvas(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-scroll", "name": "Dataset Scroll"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            self.assertIsNotNone(screen.annotation_canvas._photo)

            # Simulate a render-loss edge case, then confirm scroll event path restores it.
            screen.annotation_canvas._photo = None
            screen.annotation_canvas._needs_redraw_on_map = True
            data_scroller = screen._tab_scrollers.get("data")
            self.assertIsNotNone(data_scroller)
            assert data_scroller is not None
            data_scroller.event_generate("<<ScrollableFrameScrolled>>")
            screen.update_idletasks()

            self.assertIsNotNone(screen.annotation_canvas._photo)

        screen.destroy()

    def test_engineer_annotation_class_combines_sources_and_edits_selected_label(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        versions = [
            {
                "id": "ver-1",
                "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                "version_number": 1,
                "name": "Snapshot v1",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-1",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0", "K1Z-FA0"],
            }
        ]

        save_calls: list[list[dict]] = []

        def list_dataset_files(dataset_id: str, target: str = "images"):
            if target == "images":
                return [{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}]
            return []

        def save_annotation(dataset_id: str, image_name: str, labels: list[dict]):
            save_calls.append(labels)
            return {"labels": labels}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-123", "name": "Dataset A"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            side_effect=list_dataset_files,
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=versions,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={
                "labels": [
                    {
                        "type": "bbox",
                        "class_name": "manual-old",
                        "class": "manual-old",
                        "bbox": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
                        "normalized": True,
                    }
                ]
            },
        ), mock.patch.object(
            self.api,
            "save_annotation",
            side_effect=save_annotation,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            class_values = list(screen.annot_class["values"])
            self.assertIn("K0W-HB0", class_values)
            self.assertIn("K1Z-FA0", class_values)
            self.assertIn("manual-old", class_values)

            self.assertTrue(screen.annotation_canvas.set_selected_label_index(0))
            screen.update_idletasks()
            self.assertEqual(str(screen.annot_apply_class_button["state"]), "normal")

            screen.annot_class_var.set("manual-new")
            screen._apply_class_to_selected_annotation()
            screen.update_idletasks()

            self.assertTrue(save_calls)
            self.assertEqual(save_calls[-1][0].get("class_name"), "manual-new")
            self.assertEqual(screen.annotation_canvas.get_labels()[0].get("class_name"), "manual-new")
            self.assertIn("manual-new", list(screen.annot_class["values"]))

        screen.destroy()

    def test_engineer_create_dataset_version_focuses_latest(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        versions = [
            {
                "id": "ver-1",
                "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                "version_number": 1,
                "name": "Snapshot v1",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-1",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0"],
            }
        ]

        def list_dataset_versions(dataset_id: str):
            return [dict(item) for item in versions]

        def create_dataset_version(dataset_id: str, payload: dict):
            created = {
                "id": "ver-2",
                "display_label": "v2 | Snapshot v2 | ready | 1/1 ann",
                "version_number": 2,
                "name": payload.get("name") or "Snapshot v2",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-2",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0", "K1Z-FA0"],
            }
            versions.insert(0, dict(created))
            return created

        def list_dataset_files(dataset_id: str, target: str = "images"):
            if target == "images":
                return [{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}]
            return []

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-1", "name": "Dataset One"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            side_effect=list_dataset_files,
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            side_effect=list_dataset_versions,
        ), mock.patch.object(
            self.api,
            "create_dataset_version",
            side_effect=create_dataset_version,
        ), mock.patch(
            "client_tk.app.screens.engineer.view.messagebox.showinfo",
            return_value=None,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            self.assertEqual(screen._active_dataset_version_id, "ver-1")
            self.assertIsNotNone(screen.annotation_canvas._source_frame)

            # Simulate render/state drop before create version; implementation should restore active image.
            screen.annotation_canvas.clear()
            self.assertIsNone(screen.annotation_canvas._source_frame)

            screen.version_name.delete(0, "end")
            screen.version_name.insert(0, "Snapshot v2")
            screen.create_dataset_version()
            screen.update_idletasks()

            self.assertEqual(screen._active_dataset_version_id, "ver-2")
            selected = screen._selected_dataset_version_record()
            self.assertIsNotNone(selected)
            self.assertEqual(selected.get("id"), "ver-2")
            self.assertEqual(screen.train_dataset_version.get(), "v2 | Snapshot v2 | ready | 1/1 ann")
            self.assertEqual(screen.annot_dataset_var.get(), "ds-1")
            self.assertEqual(screen.annot_image_var.get(), "sample.png")
            self.assertIsNotNone(screen.annotation_canvas._source_frame)
            self.assertIsNotNone(screen.annotation_canvas._photo)

        screen.destroy()

    def test_annotation_canvas_right_click_resize_selected_bbox(self) -> None:
        host = ttk.Frame(self.root, width=220, height=220)
        host.pack(fill="both", expand=False)
        canvas = AnnotationCanvas(host, title="Test Canvas", size=(200, 200))
        canvas.pack(fill="both", expand=False)
        self.root.update_idletasks()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas.load_bgr(frame, image_name="sample.png")
        canvas.set_labels(
            [
                {
                    "type": "bbox",
                    "shape_type": "bbox",
                    "class_name": "object",
                    "class": "object",
                    "bbox": {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3},
                    "normalized": True,
                }
            ]
        )
        self.assertTrue(canvas.set_selected_label_index(0))

        # Start resize from top-left handle, then drag outward.
        canvas._on_right_click(SimpleNamespace(x=40, y=40))
        canvas._on_right_drag(SimpleNamespace(x=20, y=20))
        canvas._on_right_release(SimpleNamespace(x=20, y=20))
        self.root.update_idletasks()

        labels = canvas.get_labels()
        self.assertEqual(len(labels), 1)
        bbox = labels[0].get("bbox") or {}
        self.assertLess(float(bbox.get("x", 1.0)), 0.2)
        self.assertLess(float(bbox.get("y", 1.0)), 0.2)
        self.assertGreater(float(bbox.get("w", 0.0)), 0.3)
        self.assertGreater(float(bbox.get("h", 0.0)), 0.3)

        canvas.destroy()
        host.destroy()

    def test_engineer_delete_dataset_reselects_remaining_dataset(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        datasets = [
            {"id": "ds-1", "name": "Dataset One"},
            {"id": "ds-2", "name": "Dataset Two"},
        ]

        def list_datasets():
            return [dict(item) for item in datasets]

        def list_dataset_files(dataset_id: str, target: str = "images"):
            image_name = "first.png" if dataset_id == "ds-1" else "second.png"
            return [{"name": image_name, "path": f"Z:/missing/{image_name}", "size": 123}]

        def download_dataset_image(dataset_id: str, image_name: str):
            return image_bytes

        def delete_dataset(dataset_id: str):
            datasets[:] = [item for item in datasets if item["id"] != dataset_id]
            return {"deleted": True, "id": dataset_id}

        with mock.patch.object(self.api, "list_datasets", side_effect=list_datasets), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ), mock.patch.object(self.api, "list_dataset_files", side_effect=list_dataset_files), mock.patch.object(
            self.api,
            "download_dataset_image",
            side_effect=download_dataset_image,
        ), mock.patch.object(self.api, "delete_dataset", side_effect=delete_dataset) as delete_mock, mock.patch(
            "client_tk.app.screens.engineer.view.messagebox.askyesno",
            return_value=True,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            self.assertEqual(screen._selected_dataset_id(), "ds-1")
            self.assertEqual(screen.annot_dataset_var.get(), "ds-1")
            self.assertEqual(screen.annot_image_var.get(), "first.png")
            self.assertIsNotNone(screen.annotation_canvas._source_frame)

            screen.dataset_list.selection_clear(0, "end")
            self.assertIsNone(screen._selected_dataset_id())

            screen.delete_dataset()

        delete_mock.assert_called_once_with("ds-1")
        self.assertEqual(screen._selected_dataset_id(), "ds-2")
        self.assertEqual(screen.annot_dataset_var.get(), "ds-2")
        self.assertEqual(screen.annot_image_var.get(), "second.png")
        self.assertIsNotNone(screen.annotation_canvas._source_frame)
        self.assertEqual(screen.dataset_files.size(), 1)
        screen.destroy()

    def test_engineer_layout_switches_to_compact(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(screen, "winfo_width", return_value=1000), mock.patch.object(
            screen.winfo_toplevel(),
            "winfo_width",
            return_value=1000,
        ):
            screen._apply_responsive_layout()
        self.assertTrue(screen._layout_compact)
        self.assertEqual(int(screen.data_top_container.grid_info()["row"]), 0)
        self.assertEqual(int(screen.data_annotation_shell.grid_info()["row"]), 1)
        self.assertEqual(int(screen.dataset_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.upload_panel.grid_info()["row"]), 1)
        self.assertEqual(int(screen.augment_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.train_panel.grid_info()["row"]), 1)
        self.assertEqual(int(screen.training_jobs_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.training_detail_panel.grid_info()["row"]), 1)
        self.assertEqual(int(screen.models_left_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.models_right_panel.grid_info()["row"]), 1)
        self.assertEqual(int(screen.calibration_left_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.calibration_right_outer.grid_info()["row"]), 1)
        self.assertEqual(int(screen.annotation_canvas.grid_info()["row"]), 2)
        self.assertEqual(int(screen.annotation_canvas.grid_info()["column"]), 0)
        self.assertEqual(screen.annot_prev_button.pack_info()["side"], "left")
        self.assertEqual(screen.annot_next_button.pack_info()["side"], "right")
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

    def test_scrollable_frame_dispatches_to_nested_body(self) -> None:
        container = ttk.Frame(self.root, width=240, height=140)
        container.pack_propagate(False)
        container.pack(fill="both", expand=False)

        outer = ScrollableFrame(container)
        outer.pack(fill="both", expand=True)
        for index in range(6):
            ttk.Label(outer.body, text=f"Outer Row {index}").pack(anchor="w")

        inner = ScrollableFrame(outer.body)
        inner.pack(fill="both", expand=True)
        for index in range(40):
            ttk.Label(inner.body, text=f"Inner Row {index}").pack(anchor="w")

        for index in range(6, 12):
            ttk.Label(outer.body, text=f"Outer Row {index}").pack(anchor="w")

        self.root.update_idletasks()
        outer_before = outer.canvas.yview()
        inner_before = inner.canvas.yview()

        _dispatch_mousewheel(SimpleNamespace(widget=inner.body, num=5, delta=0))
        self.root.update_idletasks()

        outer_after = outer.canvas.yview()
        inner_after = inner.canvas.yview()

        self.assertGreater(inner_after[0], inner_before[0])
        self.assertEqual(outer_after, outer_before)

    def test_live_view_keeps_fixed_size_after_image_load(self) -> None:
        container = ttk.Frame(self.root, width=360, height=260)
        container.pack_propagate(False)
        container.pack(fill="both", expand=False)

        live_view = LiveView(container, "Preview", size=(320, 200))
        live_view.pack(fill="both", expand=False)
        self.root.update_idletasks()

        initial_size = (live_view.winfo_width(), live_view.winfo_height())
        image = np.zeros((1200, 1600, 3), dtype=np.uint8)
        live_view.update_bgr(image)
        self.root.update_idletasks()

        self.assertEqual((live_view.winfo_width(), live_view.winfo_height()), initial_size)

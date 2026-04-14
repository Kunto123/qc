from __future__ import annotations

import json
import random
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.repositories.base_json import JsonRepository
from backend.app.repositories.datasets_repository import DatasetsRepository


_ALLOWED_EXPORT_FORMATS = {"yolo", "yolo-detection"}
_DEFAULT_SPLIT_RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MUTABLE_VERSION_STATUSES = {"draft", "ready", "archived"}
_UNSET = object()

# Regex to detect augmented filenames: must end with _augNNN (3 digits) before extension.
_AUG_SUFFIX_RE = re.compile(r"_aug\d{3}$", re.IGNORECASE)


def _original_stem(stem: str) -> str | None:
    """Return the original image stem if *stem* carries an ``_augNNN`` suffix, else None."""
    m = _AUG_SUFFIX_RE.search(stem)
    return stem[: m.start()] if m else None


def _coerce_ratio(value: Any, default: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def _normalize_split_ratios(raw: Any) -> dict[str, float]:
    payload = raw if isinstance(raw, dict) else {}
    ratios = {
        "train": _coerce_ratio(payload.get("train"), _DEFAULT_SPLIT_RATIOS["train"]),
        "valid": _coerce_ratio(payload.get("valid", payload.get("val")), _DEFAULT_SPLIT_RATIOS["valid"]),
        "test": _coerce_ratio(payload.get("test"), _DEFAULT_SPLIT_RATIOS["test"]),
    }
    total = sum(ratios.values())
    if total <= 0:
        return dict(_DEFAULT_SPLIT_RATIOS)
    return {key: round(value / total, 6) for key, value in ratios.items()}


def _split_items(items: list[Path], ratios: dict[str, float], seed: str) -> dict[str, list[Path]]:
    if not items:
        return {"train": [], "valid": [], "test": []}

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    weighted_counts = {key: total * ratios.get(key, 0.0) for key in ("train", "valid", "test")}
    split_counts = {key: int(weighted_counts[key]) for key in weighted_counts}
    remaining = total - sum(split_counts.values())

    if remaining > 0:
        order = sorted(
            weighted_counts,
            key=lambda key: weighted_counts[key] - split_counts[key],
            reverse=True,
        )
        for key in order:
            if remaining <= 0:
                break
            split_counts[key] += 1
            remaining -= 1

    splits: dict[str, list[Path]] = {}
    cursor = 0
    for key in ("train", "valid", "test"):
        next_cursor = cursor + split_counts[key]
        splits[key] = shuffled[cursor:next_cursor]
        cursor = next_cursor
    return splits


class DatasetVersionRepository(JsonRepository):
    def __init__(self, datasets_repo: DatasetsRepository) -> None:
        super().__init__("dataset_versions.json", {"versions": []})
        self._datasets_repo = datasets_repo

    def list_versions(self, dataset_id: str | None = None) -> list[dict]:
        items = self.load()["versions"]
        if dataset_id:
            items = [item for item in items if str(item.get("dataset_id")) == str(dataset_id)]
        items = sorted(
            items,
            key=lambda item: (
                int(item.get("version_number") or 0),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )
        return [self._enrich_version(dict(item)) for item in items]

    def get_version(self, version_id: str) -> dict | None:
        if not version_id:
            return None
        for item in self.load()["versions"]:
            if str(item.get("id")) == str(version_id):
                return self._enrich_version(dict(item))
        return None

    def update_version(
        self,
        dataset_id: str,
        version_id: str,
        *,
        name: Any = _UNSET,
        description: Any = _UNSET,
        status: Any = _UNSET,
    ) -> dict:
        if self._datasets_repo.get_dataset(dataset_id) is None:
            raise ValueError("Dataset not found.")
        if name is _UNSET and description is _UNSET and status is _UNSET:
            raise ValueError("At least one mutable field is required.")

        payload = self.load()
        for item in payload["versions"]:
            if str(item.get("id")) != str(version_id) or str(item.get("dataset_id")) != str(dataset_id):
                continue

            if name is not _UNSET:
                normalized_name = str(name or "").strip()
                if not normalized_name:
                    raise ValueError("name must not be empty")
                item["name"] = normalized_name

            if description is not _UNSET:
                item["description"] = str(description or "").strip()

            if status is not _UNSET:
                normalized_status = str(status or "").strip().lower()
                if normalized_status not in _MUTABLE_VERSION_STATUSES:
                    allowed = ", ".join(sorted(_MUTABLE_VERSION_STATUSES))
                    raise ValueError(f"status must be one of: {allowed}")
                if normalized_status == "ready" and not list(item.get("class_names") or []):
                    raise ValueError("status 'ready' requires at least one class in export.")
                item["status"] = normalized_status
                item["ready_for_training"] = normalized_status == "ready"

            item["updated_at"] = datetime.now(UTC).isoformat()
            self.save(payload)
            self._write_manifest(item)
            return self._enrich_version(dict(item))

        raise ValueError("Dataset version not found.")

    def create_version(
        self,
        dataset_id: str,
        params: dict | None = None,
        *,
        augment_jobs: list[dict] | None = None,
    ) -> dict:
        if self._datasets_repo.get_dataset(dataset_id) is None:
            raise ValueError("Dataset not found.")

        payload = dict(params or {})
        export_format = str(payload.get("export_format") or "yolo").strip().lower() or "yolo"
        if export_format not in _ALLOWED_EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format '{export_format}'.")

        source_images = self._datasets_repo.list_files(dataset_id, "images")
        if not source_images:
            raise ValueError("Dataset has no images to version.")

        version_number = self._next_version_number(dataset_id)
        version_id = uuid.uuid4().hex[:12]
        version_name = str(payload.get("name") or "").strip() or f"Version {version_number}"
        description = str(payload.get("description") or "").strip()
        split_ratios = _normalize_split_ratios(payload.get("split_ratios") or payload.get("splits") or payload.get("split"))

        version_root = self.version_dir(dataset_id, version_id)
        source_images_dir = version_root / "source" / "images"
        source_labels_dir = version_root / "source" / "labels"
        export_root = version_root / "export"

        if version_root.exists():
            shutil.rmtree(version_root, ignore_errors=True)

        source_stats = self._snapshot_source(dataset_id, source_images, source_images_dir, source_labels_dir)

        # Integrate augmented images into the snapshot (photometric-only, annotation copied verbatim).
        augmented_image_count = 0
        selected_augment_job_ids: list[str] = []
        if augment_jobs:
            aug_stats = self._snapshot_augmented(dataset_id, augment_jobs, source_images_dir, source_labels_dir)
            augmented_image_count = aug_stats["augmented_image_count"]
            selected_augment_job_ids = [str(j.get("id") or "") for j in augment_jobs]

        export_stats = self._build_export_from_snapshot(
            source_images_dir=source_images_dir,
            source_labels_dir=source_labels_dir,
            export_root=export_root,
            split_ratios=split_ratios,
            seed=f"{dataset_id}:{version_number}:{version_id}",
        )

        total_image_count = source_stats["image_count"] + augmented_image_count

        now = datetime.now(UTC).isoformat()
        record = {
            "id": version_id,
            "dataset_id": dataset_id,
            "version_number": version_number,
            "name": version_name,
            "description": description,
            "export_format": export_format,
            "status": "ready" if export_stats["class_names"] else "draft",
            "ready_for_training": bool(export_stats["class_names"]),
            "split_ratios": split_ratios,
            "image_count": total_image_count,
            "label_count": source_stats["label_count"],
            "annotated_image_count": source_stats["annotated_image_count"],
            "annotation_coverage": source_stats["annotation_coverage"],
            "class_names": export_stats["class_names"],
            "skipped_label_count": export_stats["skipped_label_count"],
            "split_counts": export_stats["split_counts"],
            "source_root": str(version_root / "source"),
            "export_root": str(export_root),
            "data_yaml_path": str(export_stats["data_yaml_path"]),
            "classes_path": str(export_stats["classes_path"]),
            "manifest_path": str(version_root / "manifest.json"),
            "created_at": now,
            "exported_at": now,
            # Lineage metadata for augment integration
            "selected_augment_job_ids": selected_augment_job_ids,
            "augmented_image_count_in_version": augmented_image_count,
            "total_source_image_count": source_stats["image_count"],
        }

        self._append_version(record)
        self._write_manifest(record)
        return self._enrich_version(record)

    def export_version(self, dataset_id: str, version_id: str) -> dict:
        payload = self.load()
        for item in payload["versions"]:
            if str(item.get("id")) != str(version_id) or str(item.get("dataset_id")) != str(dataset_id):
                continue

            version_root = self.version_dir(dataset_id, version_id)
            source_images_dir = version_root / "source" / "images"
            source_labels_dir = version_root / "source" / "labels"
            export_root = version_root / "export"
            if not source_images_dir.exists() or not source_labels_dir.exists():
                raise ValueError("Dataset version snapshot is missing.")

            export_stats = self._build_export_from_snapshot(
                source_images_dir=source_images_dir,
                source_labels_dir=source_labels_dir,
                export_root=export_root,
                split_ratios=_normalize_split_ratios(item.get("split_ratios")),
                seed=f'{dataset_id}:{item.get("version_number") or 0}:{version_id}',
            )

            item.update(
                {
                    "export_format": str(item.get("export_format") or "yolo").strip().lower() or "yolo",
                    "class_names": export_stats["class_names"],
                    "skipped_label_count": export_stats["skipped_label_count"],
                    "split_counts": export_stats["split_counts"],
                    "data_yaml_path": str(export_stats["data_yaml_path"]),
                    "classes_path": str(export_stats["classes_path"]),
                    "export_root": str(export_root),
                    "status": "ready" if export_stats["class_names"] else "draft",
                    "ready_for_training": bool(export_stats["class_names"]),
                    "exported_at": datetime.now(UTC).isoformat(),
                }
            )
            self.save(payload)
            self._write_manifest(item)
            return self._enrich_version(dict(item))

        raise ValueError("Dataset version not found.")

    def version_dir(self, dataset_id: str, version_id: str) -> Path:
        return self._datasets_repo.dataset_dir(dataset_id) / "versions" / version_id

    def _append_version(self, record: dict) -> None:
        payload = self.load()
        payload["versions"].append(record)
        self.save(payload)

    def _next_version_number(self, dataset_id: str) -> int:
        versions = [item for item in self.load()["versions"] if str(item.get("dataset_id")) == str(dataset_id)]
        if not versions:
            return 1
        return max(int(item.get("version_number") or 0) for item in versions) + 1

    def _snapshot_source(self, dataset_id: str, source_images: list[dict], source_images_dir: Path, source_labels_dir: Path) -> dict[str, Any]:
        source_images_dir.mkdir(parents=True, exist_ok=True)
        source_labels_dir.mkdir(parents=True, exist_ok=True)

        class_names: list[str] = []
        annotated_image_count = 0
        label_count = 0
        image_count = 0

        for item in source_images:
            source_image_path = Path(str(item.get("path") or ""))
            if not source_image_path.exists():
                continue
            image_count += 1
            shutil.copy2(source_image_path, source_images_dir / source_image_path.name)

            annotation = self._datasets_repo.get_annotation(dataset_id, source_image_path.name)
            (source_labels_dir / f"{source_image_path.stem}.json").write_text(
                json.dumps(annotation, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )

            labels = annotation.get("labels") if isinstance(annotation, dict) else []
            if not isinstance(labels, list):
                labels = []
            label_count += len(labels)
            if labels:
                annotated_image_count += 1
            for label in labels:
                class_name = self._label_class_name(label)
                if class_name and class_name not in class_names:
                    class_names.append(class_name)

        annotation_coverage = round(annotated_image_count / image_count, 4) if image_count else 0.0

        return {
            "image_count": image_count,
            "label_count": label_count,
            "annotated_image_count": annotated_image_count,
            "annotation_coverage": annotation_coverage,
            "class_names": class_names,
        }

    def _snapshot_augmented(
        self,
        dataset_id: str,
        augment_jobs: list[dict],
        source_images_dir: Path,
        source_labels_dir: Path,
    ) -> dict[str, int]:
        """Copy augmented images and their original annotations into the version snapshot.

        For each augmented image ``{stem}_augNNN{ext}`` the annotation from the
        corresponding original image ``{stem}{ext}`` is copied verbatim (safe because
        only photometric transforms preserve object geometry).

        Returns ``{"augmented_image_count": N}`` — the number of images actually copied.
        """
        augmented_count = 0
        for job in augment_jobs:
            output_dir_str = str(job.get("output_dataset_id") or "")
            if not output_dir_str:
                continue
            output_dir = Path(output_dir_str)
            if not output_dir.exists() or not output_dir.is_dir():
                continue

            for aug_file in sorted(output_dir.iterdir()):
                if not aug_file.is_file() or aug_file.suffix.lower() not in _IMAGE_EXTS:
                    continue

                orig_stem = _original_stem(aug_file.stem)
                if orig_stem is None:
                    continue  # not a recognized augmented filename

                # Copy augmented image into the snapshot images dir.
                dest_image = source_images_dir / aug_file.name
                shutil.copy2(aug_file, dest_image)

                # Retrieve the original annotation (try same extension first, then others).
                annotation: dict = {}
                for ext in (aug_file.suffix.lower(), ".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    candidate_name = f"{orig_stem}{ext}"
                    ann = self._datasets_repo.get_annotation(dataset_id, candidate_name)
                    if isinstance(ann, dict) and ann.get("labels"):
                        annotation = ann
                        break

                aug_annotation = dict(annotation)
                aug_annotation["image_name"] = aug_file.name

                (source_labels_dir / f"{aug_file.stem}.json").write_text(
                    json.dumps(aug_annotation, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
                augmented_count += 1

        return {"augmented_image_count": augmented_count}

    def _build_export_from_snapshot(
        self,
        *,
        source_images_dir: Path,
        source_labels_dir: Path,
        export_root: Path,
        split_ratios: dict[str, float],
        seed: str,
    ) -> dict[str, Any]:
        try:
            import cv2
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenCV is required to export dataset versions: {exc}") from exc

        if export_root.exists():
            shutil.rmtree(export_root, ignore_errors=True)

        export_images_root = export_root / "images"
        export_labels_root = export_root / "labels"
        for split in ("train", "valid", "test"):
            (export_images_root / split).mkdir(parents=True, exist_ok=True)
            (export_labels_root / split).mkdir(parents=True, exist_ok=True)

        source_images = [
            item
            for item in sorted(source_images_dir.iterdir())
            if item.is_file() and item.suffix.lower() in _IMAGE_EXTS
        ]
        annotations: dict[str, dict] = {}
        class_names: list[str] = []

        for image_path in source_images:
            annotation_path = source_labels_dir / f"{image_path.stem}.json"
            if annotation_path.exists():
                try:
                    annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    annotation_payload = {}
            else:
                annotation_payload = {}
            if not isinstance(annotation_payload, dict):
                annotation_payload = {}
            annotation_payload.setdefault("image_name", image_path.name)
            labels = annotation_payload.get("labels")
            if not isinstance(labels, list):
                labels = []
                annotation_payload["labels"] = labels
            annotations[image_path.name] = annotation_payload
            for label in labels:
                class_name = self._label_class_name(label)
                if class_name and class_name not in class_names:
                    class_names.append(class_name)

        class_index = {name: index for index, name in enumerate(class_names)}
        split_assignments = _split_items(source_images, split_ratios, seed)

        skipped_label_count = 0
        yolo_label_count = 0
        split_counts = {key: len(value) for key, value in split_assignments.items()}

        for split_name, split_items in split_assignments.items():
            for image_path in split_items:
                image_dest = export_images_root / split_name / image_path.name
                label_dest = export_labels_root / split_name / f"{image_path.stem}.txt"
                shutil.copy2(image_path, image_dest)

                image_size = self._image_size(cv2, image_path)
                annotation = annotations.get(image_path.name, {})
                label_lines, skipped = self._annotation_to_yolo(annotation, class_index, image_size)
                skipped_label_count += skipped
                yolo_label_count += len(label_lines)
                label_dest.write_text("\n".join(label_lines), encoding="utf-8")

        data_yaml_path = export_root / "data.yaml"
        classes_path = export_root / "classes.txt"
        data_yaml_path.write_text(self._build_data_yaml(class_names, export_root=export_root), encoding="utf-8")
        classes_path.write_text("\n".join(class_names), encoding="utf-8")

        return {
            "class_names": class_names,
            "split_counts": split_counts,
            "skipped_label_count": skipped_label_count,
            "yolo_label_count": yolo_label_count,
            "data_yaml_path": data_yaml_path,
            "classes_path": classes_path,
        }

    def _write_manifest(self, record: dict) -> None:
        manifest_path = Path(str(record["manifest_path"]))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")

    def _enrich_version(self, record: dict) -> dict:
        enriched = dict(record)
        dataset = self._datasets_repo.get_dataset(str(record.get("dataset_id") or ""))
        dataset_name = str(dataset.get("name") or "") if dataset else ""
        annotated = int(enriched.get("annotated_image_count") or 0)
        image_count = int(enriched.get("image_count") or 0)
        coverage = float(enriched.get("annotation_coverage") or 0.0)
        status = str(enriched.get("status") or "draft")
        enriched.update(
            {
                "dataset_name": dataset_name,
                "coverage_percent": round(coverage * 100, 2),
                "display_label": (
                    f"v{enriched.get('version_number')} | {enriched.get('name')} | "
                    f"{status} | {annotated}/{image_count} ann"
                ),
            }
        )
        return enriched

    @staticmethod
    def _label_class_name(label: Any) -> str | None:
        if not isinstance(label, dict):
            return None
        value = str(label.get("class_name") or label.get("class") or label.get("label") or "").strip()
        return value or None

    @staticmethod
    def _load_box_values(source: dict[str, Any]) -> tuple[float, float, float, float] | None:
        try:
            x = float(source.get("x"))
            y = float(source.get("y"))
            width = float(source.get("w"))
            height = float(source.get("h"))
        except (TypeError, ValueError):
            return None
        return x, y, width, height

    @staticmethod
    def _image_size(cv2_module, image_path: Path) -> tuple[int, int] | None:  # type: ignore[no-untyped-def]
        image = cv2_module.imread(str(image_path))
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height

    def _annotation_to_yolo(
        self,
        annotation: dict,
        class_index: dict[str, int],
        image_size: tuple[int, int] | None,
    ) -> tuple[list[str], int]:
        labels = annotation.get("labels") if isinstance(annotation, dict) else []
        if not isinstance(labels, list):
            labels = []

        lines: list[str] = []
        skipped = 0
        for label in labels:
            line = self._label_to_yolo_line(label, class_index, image_size)
            if line is None:
                skipped += 1
                continue
            lines.append(line)
        return lines, skipped

    def _label_to_yolo_line(
        self,
        label: Any,
        class_index: dict[str, int],
        image_size: tuple[int, int] | None,
    ) -> str | None:
        if not isinstance(label, dict):
            return None

        class_name = self._label_class_name(label)
        if class_name is None:
            return None
        class_id = class_index.get(class_name)
        if class_id is None:
            return None

        normalized = bool(label.get("normalized"))
        box = None

        if isinstance(label.get("bbox"), dict):
            box = self._load_box_values(label["bbox"])
        elif all(key in label for key in ("x", "y", "w", "h")):
            box = self._load_box_values(label)

        if box is not None:
            x, y, width, height = box
            if not normalized and image_size is not None and max(abs(x), abs(y), abs(width), abs(height)) > 1.0:
                image_width, image_height = image_size
                if image_width <= 0 or image_height <= 0:
                    return None
                x /= float(image_width)
                y /= float(image_height)
                width /= float(image_width)
                height /= float(image_height)
            return self._format_yolo_line(class_id, x, y, width, height)

        raw_points = label.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            return None

        normalized_points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            try:
                px = float(point.get("x"))
                py = float(point.get("y"))
            except (TypeError, ValueError):
                continue
            if not normalized and image_size is not None and max(abs(px), abs(py)) > 1.0:
                image_width, image_height = image_size
                if image_width <= 0 or image_height <= 0:
                    return None
                px /= float(image_width)
                py /= float(image_height)
            normalized_points.append((px, py))

        if len(normalized_points) < 3:
            return None

        xs = [point[0] for point in normalized_points]
        ys = [point[1] for point in normalized_points]
        x = min(xs)
        y = min(ys)
        width = max(xs) - x
        height = max(ys) - y
        return self._format_yolo_line(class_id, x, y, width, height)

    @staticmethod
    def _format_yolo_line(class_id: int, x: float, y: float, width: float, height: float) -> str | None:
        if width <= 0 or height <= 0:
            return None
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        width = min(1.0, max(0.0, width))
        height = min(1.0, max(0.0, height))
        return f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"

    @staticmethod
    def _build_data_yaml(class_names: list[str], *, export_root: Path) -> str:
        dataset_root = export_root.resolve().as_posix()
        if class_names:
            names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
            names_section = f"names:\n{names_block}\n"
        else:
            names_section = "names: []\n"

        return (
            f"path: {dataset_root}\n"
            "train: images/train\n"
            "val: images/valid\n"
            "test: images/test\n"
            f"nc: {len(class_names)}\n"
            f"{names_section}"
        )
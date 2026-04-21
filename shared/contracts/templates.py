from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CameraDefaults:
    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None


@dataclass(slots=True)
class RoiGeometry:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class VisionConfig:
    model_path: str = "models/dummy.pt"
    model_meta_path: str | None = None
    runtime: str = "ultralytics"
    conf_threshold: float = 0.25
    stream_fps: float = 10.0
    inference_fps: float = 4.0
    imgsz: int = 640
    classes: list[str] = field(default_factory=list)
    enable_ergonomic_check: bool = False
    ergonomic_pose_model_path: str | None = None
    ergonomic_min_keypoint_conf: float = 0.35


@dataclass(slots=True)
class PartReadyConfig:
    enabled: bool = True
    color_profile_id: int | None = None
    colorspace: str = "LAB"
    distance_threshold: float | None = None
    min_match_ratio: float | None = None


@dataclass(slots=True)
class StickerRule:
    part_name: str
    expected_class: str
    line: str
    enabled: bool = True
    validator_mode: str = "ml_detection"
    min_roi_confidence: float = 0.0
    min_class_confidence: float | None = None
    max_offset_x: float | None = None
    max_offset_y: float | None = None
    expected_center_x: float | None = None
    expected_center_y: float | None = None
    expected_tilt_degrees: float = 0.0
    max_tilt_degrees: float | None = None
    # Tilt gate toggle: when False (default) the OUT_OF_ANGLE decision is never
    # raised — tilt telemetry is still calculated and forwarded as observability data.
    # Set True to make max_tilt_degrees an active reject gate.
    tilt_gate_enabled: bool = False
    # Legacy field — kept for backward compatibility with older templates and API
    # payloads only. Runtime commit timing is now controlled exclusively by
    # part_ready_settle_ms and no longer depends on this field.
    commit_stable_frames: int = 1
    # Primary runtime timing knob: controls both the inference hold (settle) and the
    # commit window after the first stable post-ready result.
    # None  = use system-wide default (QC_SUITE_PART_READY_SETTLE_MS env var).
    # 0     = bypass debounce for this template regardless of env.
    # > 0   = explicit ms value; overrides env default.
    part_ready_settle_ms: int | None = None


@dataclass(slots=True)
class PersistenceConfig:
    write_to_db: bool = True


@dataclass(slots=True)
class InspectionTemplate:
    id: int | None
    version_id: int | None
    version_number: int
    name: str
    description: str
    is_active: bool
    camera: CameraDefaults
    part_ready_roi: RoiGeometry
    sticker_roi: RoiGeometry
    vision: VisionConfig
    part_ready: PartReadyConfig
    sticker: StickerRule
    persistence: PersistenceConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def roi(self) -> RoiGeometry:
        # Transitional alias for code paths that still reference the legacy single ROI.
        return self.sticker_roi

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version_id": self.version_id,
            "version_number": self.version_number,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "camera": asdict(self.camera),
            "part_ready_roi": asdict(self.part_ready_roi),
            "sticker_roi": asdict(self.sticker_roi),
            "vision": asdict(self.vision),
            "part_ready": asdict(self.part_ready),
            "sticker": asdict(self.sticker),
            "persistence": asdict(self.persistence),
            "metadata": dict(self.metadata),
        }


def _pick_roi_payload(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def template_from_dict(payload: dict[str, Any]) -> InspectionTemplate:
    part_ready_roi_payload = _pick_roi_payload(payload, "part_ready_roi", "roi", "sticker_roi")
    sticker_roi_payload = _pick_roi_payload(payload, "sticker_roi", "roi", "part_ready_roi")
    return InspectionTemplate(
        id=payload.get("id"),
        version_id=payload.get("version_id"),
        version_number=int(payload.get("version_number") or 1),
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or "").strip(),
        is_active=bool(payload.get("is_active", True)),
        camera=CameraDefaults(**(payload.get("camera") or {})),
        part_ready_roi=RoiGeometry(**part_ready_roi_payload),
        sticker_roi=RoiGeometry(**sticker_roi_payload),
        vision=VisionConfig(**(payload.get("vision") or {})),
        part_ready=PartReadyConfig(**(payload.get("part_ready") or {})),
        sticker=StickerRule(**(payload.get("sticker") or {})),
        persistence=PersistenceConfig(**(payload.get("persistence") or {})),
        metadata=dict(payload.get("metadata") or {}),
    )

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CameraDefaults:
    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    rotation_degrees: float = 0.0


@dataclass(slots=True)
class RoiGeometry:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    rotation: float = 0.0
    width: int | None = None


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
    text_anchor_class: str = "text_anchor"
    center_dot_class: str = "center_dot"
    anchor_crop_padding_ratio: float = 0.08
    anchor_crop_scale: float = 2.0


@dataclass(slots=True)
class PartReadyConfig:
    enabled: bool = True
    method: str = "gap_template_match"
    gap_match_threshold: float = 0.85
    gap_ref_path: str | None = None
    logo_ref_path: str | None = None
    gap_ref_type: str = "raw"
    gap_hsv_lower: list[int] = field(default_factory=lambda: [90, 50, 50])
    gap_hsv_upper: list[int] = field(default_factory=lambda: [130, 255, 255])
    gap_padding_px: int = 20
    color_profile_id: int | None = None
    colorspace: str = "LAB"
    distance_threshold: float | None = None
    min_match_ratio: float | None = None
    hsv_lower: list[int] = field(default_factory=lambda: [0, 0, 0])
    hsv_upper: list[int] = field(default_factory=lambda: [180, 255, 80])
    stable_ms: int = 500
    release_ms: int = 300
    ema_alpha: float = 0.3
    hsv_adaptive: bool = False
    hsv_adaptive_alpha: float = 0.1
    hsv_adaptive_min_ratio: float = 0.85


@dataclass(slots=True)
class ComponentClassTarget:
    class_name: str
    count: int


@dataclass(slots=True)
class ComponentRoiRule:
    name: str
    roi: RoiGeometry
    classes: list[ComponentClassTarget]
    strict_foreign_class: bool = False


@dataclass(slots=True)
class StickerRule:
    part_name: str
    expected_class: str
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
    tilt_gate_enabled: bool = False
    edge_roi_tolerance_px: int = 10
    edge_search_padding_ratio: float = 0.10
    morph_kernel_width: int = 40
    morph_kernel_height: int = 5
    min_text_aspect_ratio: float = 3.0
    commit_stable_frames: int = 1
    part_ready_settle_ms: int | None = None
    part_ready_settle_frames: int = 3
    white_hsv_lower: list[int] = field(default_factory=lambda: [0, 0, 160])
    white_hsv_upper: list[int] = field(default_factory=lambda: [180, 70, 255])
    min_text_contour_area_ratio: float = 0.002


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
    component_rois: list[ComponentRoiRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def roi(self) -> RoiGeometry:
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
            "component_rois": [
                {
                    "name": cr.name,
                    "roi": asdict(cr.roi),
                    "classes": [asdict(c) for c in cr.classes],
                    "strict_foreign_class": cr.strict_foreign_class,
                }
                for cr in self.component_rois
            ],
            "metadata": dict(self.metadata),
        }


_ROI_ALLOWED = {"x", "y", "w", "h", "rotation", "width"}


def _pick_roi_payload(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


_VALID_STICKER_FIELDS = {
    "part_name", "expected_class", "enabled", "validator_mode", "min_roi_confidence",
    "min_class_confidence", "max_offset_x", "max_offset_y", "expected_center_x",
    "expected_center_y", "expected_tilt_degrees", "max_tilt_degrees",
    "tilt_gate_enabled", "edge_roi_tolerance_px",
    "edge_search_padding_ratio", "morph_kernel_width", "morph_kernel_height",
    "min_text_aspect_ratio", "commit_stable_frames", "part_ready_settle_ms",
    "part_ready_settle_frames", "white_hsv_lower", "white_hsv_upper",
    "min_text_contour_area_ratio",
}


def _parse_component_rois(payload: dict[str, Any]) -> list[ComponentRoiRule]:
    """Parse component_rois from template payload dict."""
    raw_rois = payload.get("component_rois") or []
    if not isinstance(raw_rois, list):
        return []
    result = []
    for raw_roi in raw_rois:
        if not isinstance(raw_roi, dict):
            continue
        roi_raw = raw_roi.get("roi") or {}
        roi_geom = RoiGeometry(**{k: v for k, v in roi_raw.items() if k in _ROI_ALLOWED})
        raw_classes = raw_roi.get("classes") or []
        classes = []
        for rc in raw_classes:
            if not isinstance(rc, dict):
                continue
            cn = str(rc.get("class_name") or "").strip()
            if not cn:
                continue
            try:
                cnt = int(rc.get("count") or 0)
            except (TypeError, ValueError):
                cnt = 0
            if cnt <= 0:
                continue
            classes.append(ComponentClassTarget(class_name=cn, count=cnt))
        if not classes:
            continue
        result.append(ComponentRoiRule(
            name=str(raw_roi.get("name") or "ROI").strip() or "ROI",
            roi=roi_geom,
            classes=classes,
            strict_foreign_class=bool(raw_roi.get("strict_foreign_class", False)),
        ))
    return result


def template_from_dict(payload: dict[str, Any]) -> InspectionTemplate:
    part_ready_roi_payload = _pick_roi_payload(payload, "part_ready_roi", "roi", "sticker_roi")
    sticker_roi_payload = _pick_roi_payload(payload, "sticker_roi", "roi", "part_ready_roi")
    # Strip unknown keys that RoiGeometry doesn't accept (e.g. 'height' from old DB data)
    part_ready_roi_payload = {k: v for k, v in part_ready_roi_payload.items() if k in _ROI_ALLOWED}
    sticker_roi_payload = {k: v for k, v in sticker_roi_payload.items() if k in _ROI_ALLOWED}
    _sticker_raw = dict(payload.get("sticker") or {})
    _vision_raw = payload.get("vision") or {}
    _vision_filtered = {k: v for k, v in _vision_raw.items() if k in VisionConfig.__slots__}
    _sticker_filtered = {k: v for k, v in _sticker_raw.items() if k in _VALID_STICKER_FIELDS}
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
        vision=VisionConfig(**_vision_filtered),
        part_ready=PartReadyConfig(**(payload.get("part_ready") or {})),
        sticker=StickerRule(**_sticker_filtered),
        persistence=PersistenceConfig(**(payload.get("persistence") or {})),
        component_rois=_parse_component_rois(payload),
        metadata=dict(payload.get("metadata") or {}),
    )

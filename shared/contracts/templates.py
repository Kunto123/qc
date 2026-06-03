from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CameraDefaults:
    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    rotation_degrees: float = 0.0  # Free rotation: 0, 90, 180, 270, or any angle


@dataclass(slots=True)
class RoiGeometry:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    rotation: float = 0.0
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
    ocr_engine: str = "default"
    ocr_language: str = "eng"
    ocr_psm: int = 7
    ocr_allowlist: str = ""
    text_anchor_class: str = "text_anchor"
    center_dot_class: str = "center_dot"
    anchor_crop_padding_ratio: float = 0.08
    anchor_crop_scale: float = 2.0


@dataclass(slots=True)
class PartReadyConfig:
    enabled: bool = True
    # Method: "gap_template_match" (new default) or "color_profile_match" (legacy)
    method: str = "gap_template_match"
    # Gap detection via template matching
    gap_match_threshold: float = 0.85  # min cv2.matchTemplate score for part_ready=True
    gap_ref_path: str | None = None    # relative path to reference PNG on disk
    gap_hsv_lower: list[int] = field(default_factory=lambda: [90, 50, 50])  # blue clamp HSV lower
    gap_hsv_upper: list[int] = field(default_factory=lambda: [130, 255, 255])  # blue clamp HSV upper
    gap_padding_px: int = 20  # px padding around clamp mask to extract gap patch
    # Legacy color profile fields (kept for backward compatibility)
    color_profile_id: int | None = None
    colorspace: str = "LAB"
    distance_threshold: float | None = None
    min_match_ratio: float | None = None
    hsv_lower: list[int] = field(default_factory=lambda: [0, 0, 0])
    hsv_upper: list[int] = field(default_factory=lambda: [180, 255, 80])
    stable_ms: int = 500
    release_ms: int = 300


@dataclass(slots=True)
class StickerRule:
    part_name: str
    expected_class: str
    line: str
    station: str = ""
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
    use_ocr: bool = False
    ocr_expected_code: str = ""
    ocr_flip_fallback: bool = True
    ocr_mode: str | None = None
    ocr_expected_text: str | None = None
    ocr_min_confidence: float | None = None
    ocr_regex: str | None = None
    ocr_canonical_map: dict[str, str] = field(default_factory=dict)
    anchor_min_confidence: float | None = None
    dot_min_confidence: float | None = None
    expected_dot_x: float | None = None
    expected_dot_y: float | None = None
    max_anchor_offset_x: float | None = None
    max_anchor_offset_y: float | None = None
    # Tilt gate toggle: when False (default) the OUT_OF_ANGLE decision is never
    # raised — tilt telemetry is still calculated and forwarded as observability data.
    # Set True to make max_tilt_degrees an active reject gate.
    tilt_gate_enabled: bool = False
    # Edge/text-band analysis config for OUT_OF_ANGLE gate
    edge_roi_tolerance_px: int = 10  # px tolerance for text-band edge sticking out of ROI
    edge_search_padding_ratio: float = 0.10  # expand sticker ROI by this ratio for edge search
    morph_kernel_width: int = 40  # horizontal kernel width for morphological closing
    morph_kernel_height: int = 5  # horizontal kernel height for morphological closing
    min_text_aspect_ratio: float = 3.0  # min aspect ratio to filter text bands vs logos
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

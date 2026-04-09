from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def _format_metric(value, *, precision: int = 3) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_timestamp(value) -> str:
    text = str(value or "-")
    if text == "-":
        return text
    return text.replace("T", " ")[:19]


class ResultPanel(ttk.LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Inspection Status", padding=12)
        self.columnconfigure(0, weight=1)
        self._value_widgets: list[ttk.Label] = []

        self.decision_banner = tk.Label(
            self,
            text="WAITING",
            bg="#334155",
            fg="#f8fafc",
            font=("Segoe UI", 24, "bold"),
            padx=16,
            pady=14,
        )
        self.decision_banner.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.subtitle_var = ttk.Label(
            self,
            text="Menunggu event inspeksi pertama.",
            font=("Segoe UI", 10),
            wraplength=320,
            justify="left",
        )
        self.subtitle_var.grid(row=1, column=0, sticky="w", pady=(0, 10))

        self.live_frame = ttk.LabelFrame(self, text="Live Stage", padding=8)
        self.live_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.live_state_var = self._build_field(self.live_frame, 0, "Event State")
        self.live_decision_var = self._build_field(self.live_frame, 1, "Live Decision")
        self.live_reason_var = self._build_field(self.live_frame, 2, "Live Reason")

        self.part_ready_frame = ttk.LabelFrame(self, text="Part Ready Gate", padding=8)
        self.part_ready_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.part_ready_status_var = self._build_field(self.part_ready_frame, 0, "Status")
        self.part_ready_ratio_var = self._build_field(self.part_ready_frame, 1, "Match Ratio (avg5)")
        self.part_ready_raw_ratio_var = self._build_field(self.part_ready_frame, 2, "Match Ratio (raw)")
        self.part_ready_distance_var = self._build_field(self.part_ready_frame, 3, "Mean Distance")
        self.part_ready_profile_var = self._build_field(self.part_ready_frame, 4, "Profile")

        self.sticker_frame = ttk.LabelFrame(self, text="Sticker Validation", padding=8)
        self.sticker_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.detected_class_var = self._build_field(self.sticker_frame, 0, "Detected Class")
        self.expected_class_var = self._build_field(self.sticker_frame, 1, "Expected Class")
        self.sticker_confidence_var = self._build_field(self.sticker_frame, 2, "Confidence")
        self.sticker_backend_var = self._build_field(self.sticker_frame, 3, "Backend")
        self.candidate_source_var = self._build_field(self.sticker_frame, 4, "Candidate Source")
        self.offset_var = self._build_field(self.sticker_frame, 5, "Offset")

        self.debug_frame = ttk.LabelFrame(self, text="Inference Debug", padding=8)
        self.debug_frame.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.raw_detection_count_var = self._build_field(self.debug_frame, 0, "Raw Detections")
        self.fallback_reason_var = self._build_field(self.debug_frame, 1, "Fallback Reason")
        self.classes_filter_var = self._build_field(self.debug_frame, 2, "Classes Filter")

        self.commit_frame = ttk.LabelFrame(self, text="Commit Details", padding=8)
        self.commit_frame.grid(row=6, column=0, sticky="ew")
        self.reason_var = self._build_field(self.commit_frame, 0, "Reason")
        self.part_var = self._build_field(self.commit_frame, 1, "Part")
        self.line_var = self._build_field(self.commit_frame, 2, "Line")
        self.station_var = self._build_field(self.commit_frame, 3, "Station")
        self.db_var = self._build_field(self.commit_frame, 4, "DB Write")
        self.event_var = self._build_field(self.commit_frame, 5, "Event ID")
        self.commit_var = self._build_field(self.commit_frame, 6, "Committed At")
        self.bind("<Configure>", self._on_resize, add="+")

    def _build_field(self, master, row: int, title: str) -> ttk.Label:
        master.columnconfigure(1, weight=1)
        ttk.Label(master, text=f"{title}:", font=("Segoe UI", 9, "bold")).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )
        value = ttk.Label(master, text="-", wraplength=210, justify="left")
        value.grid(row=row, column=1, sticky="w", pady=2)
        self._value_widgets.append(value)
        return value

    def _on_resize(self, event) -> None:
        wraplength = max(140, min(280, event.width - 170))
        for widget in self._value_widgets:
            widget.configure(wraplength=wraplength)

    def update_payload(self, payload: dict | None) -> None:
        if not payload:
            self.reset()
            return

        live_validation = payload.get("validation") or {}
        live_part_ready = payload.get("part_ready") or {}
        live_details = live_validation.get("validation_details") or {}
        live_candidate = live_details.get("selected_candidate") or {}

        committed = payload.get("last_committed_result") or {}
        committed_validation = committed.get("validation") or {}
        committed_part_ready = committed.get("part_ready") or {}
        committed_details = committed_validation.get("validation_details") or {}
        committed_candidate = committed_details.get("selected_candidate") or {}
        committed_sticker_detection = committed.get("sticker_detection") or {}
        live_sticker_detection = payload.get("sticker_detection") or {}
        db_write = committed.get("db_write") or payload.get("db_write") or {}

        display_validation = committed_validation or live_validation
        display_part_ready = committed_part_ready or live_part_ready
        display_details = committed_details or live_details
        display_candidate = committed_candidate or live_candidate
        display_sticker_detection = committed_sticker_detection or live_sticker_detection

        decision = display_validation.get("decision") or "WAITING"
        reason = display_validation.get("reject_reason_code") or ("OK" if decision == "ACCEPT" else "-")
        palette = {
            "ACCEPT": ("#166534", "#f0fdf4"),
            "REJECT": ("#991b1b", "#fef2f2"),
            "WAITING": ("#334155", "#f8fafc"),
        }
        bg, fg = palette.get(decision, ("#334155", "#f8fafc"))
        self.decision_banner.configure(bg=bg, fg=fg, text=decision)
        if committed_validation:
            self.subtitle_var.configure(
                text="Banner menampilkan hasil committed terakhir. Detail live tetap menunjukkan frame yang sedang diproses."
            )
        else:
            self.subtitle_var.configure(
                text="Belum ada event committed. Banner sementara mengikuti hasil live terbaru."
            )

        live_reason = live_validation.get("reject_reason_code") or ("OK" if live_validation.get("decision") == "ACCEPT" else "-")
        self.live_state_var.configure(text=str(payload.get("event_state") or "-").upper())
        self.live_decision_var.configure(text=str(live_validation.get("decision") or "-"))
        self.live_reason_var.configure(text=str(live_reason))

        part_ready_status = display_part_ready.get("status") or ("ready" if display_part_ready.get("part_ready") else "not_ready")
        if not display_part_ready.get("enabled", True):
            part_ready_status = part_ready_status or "skipped"
        self.part_ready_status_var.configure(
            text=f"{str(part_ready_status).upper()} | gate={'READY' if display_part_ready.get('part_ready') else 'BLOCK'}"
        )
        self.part_ready_ratio_var.configure(text=_format_metric(display_part_ready.get("match_ratio")))
        self.part_ready_raw_ratio_var.configure(text=_format_metric(display_part_ready.get("raw_match_ratio")))
        self.part_ready_distance_var.configure(text=_format_metric(display_part_ready.get("mean_distance")))
        profile_text = "-"
        if display_part_ready.get("color_profile_id"):
            profile_text = f"id={display_part_ready.get('color_profile_id')} | {display_part_ready.get('colorspace') or 'LAB'}"
        elif display_part_ready.get("enabled") is False:
            profile_text = "disabled"
        self.part_ready_profile_var.configure(text=profile_text)

        confidence = display_validation.get("sticker_confidence")
        if confidence is None:
            confidence = display_candidate.get("confidence")
        offset = display_candidate.get("offset") or {}
        offset_text = "-"
        if offset:
            offset_text = f"x={_format_metric(offset.get('x'), precision=2)}, y={_format_metric(offset.get('y'), precision=2)}"
        backend = (
            display_validation.get("sticker_backend")
            or display_sticker_detection.get("backend")
            or "-"
        )
        self.detected_class_var.configure(text=str(display_validation.get("detected_class") or "-"))
        self.expected_class_var.configure(text=str(display_validation.get("expected_class") or "-"))
        self.sticker_confidence_var.configure(text=_format_metric(confidence, precision=4))
        self.sticker_backend_var.configure(text=str(backend))
        self.candidate_source_var.configure(text=str(display_details.get("candidate_source") or "-"))
        self.offset_var.configure(text=offset_text)

        raw_count = display_sticker_detection.get("raw_detection_count")
        raw_count_text = str(raw_count) if raw_count is not None else "-"
        fallback_reason = display_sticker_detection.get("fallback_reason") or "-"
        allowed_labels = display_sticker_detection.get("allowed_labels_filter")
        classes_filter_text = ", ".join(allowed_labels) if allowed_labels is not None else "-"
        self.raw_detection_count_var.configure(text=raw_count_text)
        self.fallback_reason_var.configure(text=str(fallback_reason))
        self.classes_filter_var.configure(text=classes_filter_text)

        self.reason_var.configure(text=str(reason))
        self.part_var.configure(text=str(display_validation.get("part_name") or "-"))
        self.line_var.configure(text=str(display_validation.get("line_id") or "-"))
        self.station_var.configure(text=str(display_validation.get("station_id") or "-"))
        self.db_var.configure(text="OK" if db_write.get("written") else str(db_write.get("reason") or "-"))
        self.event_var.configure(text=str(committed.get("event_id") or payload.get("event_id") or "-"))
        self.commit_var.configure(text=_format_timestamp(committed.get("committed_at")))

    def reset(self) -> None:
        self.decision_banner.configure(bg="#334155", fg="#f8fafc", text="WAITING")
        self.subtitle_var.configure(text="Menunggu event inspeksi pertama.")
        for widget in (
            self.live_state_var,
            self.live_decision_var,
            self.live_reason_var,
            self.part_ready_status_var,
            self.part_ready_ratio_var,
            self.part_ready_raw_ratio_var,
            self.part_ready_distance_var,
            self.part_ready_profile_var,
            self.detected_class_var,
            self.expected_class_var,
            self.sticker_confidence_var,
            self.sticker_backend_var,
            self.candidate_source_var,
            self.offset_var,
            self.raw_detection_count_var,
            self.fallback_reason_var,
            self.classes_filter_var,
            self.reason_var,
            self.part_var,
            self.line_var,
            self.station_var,
            self.db_var,
            self.event_var,
            self.commit_var,
        ):
            widget.configure(text="-")

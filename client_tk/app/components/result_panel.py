from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from client_tk.app.theme import BORDER, PANEL_ALT_BG, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY, WARNING, WARNING_HOVER


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


class ResultPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER)
        self.columnconfigure(0, weight=1)
        self._value_widgets: list[ctk.CTkLabel] = []

        ctk.CTkLabel(self, text="Inspection Status", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w",
            padx=12,
            pady=(12, 8),
        )

        self.decision_banner = ctk.CTkLabel(
            self,
            text="WAITING",
            fg_color="#334155",
            text_color="#f8fafc",
            font=("Segoe UI", 24, "bold"),
            corner_radius=14,
            anchor="center",
        )
        self.decision_banner.pack(fill="x", padx=12, pady=(0, 10))

        self.subtitle_var = ctk.CTkLabel(
            self,
            text="Menunggu event inspeksi pertama.",
            font=("Segoe UI", 10),
            wraplength=320,
            justify="left",
            text_color=TEXT_SECONDARY,
        )
        self.subtitle_var.pack(anchor="w", padx=12, pady=(0, 10))

        self.live_frame = self._build_section("Live Stage")
        self.live_state_var = self._build_field(self.live_frame, 0, "Event State")
        self.live_decision_var = self._build_field(self.live_frame, 1, "Live Decision")
        self.live_reason_var = self._build_field(self.live_frame, 2, "Live Reason")
        self.live_template_version_var = self._build_field(self.live_frame, 3, "Template Version")

        self.part_ready_frame = self._build_section("Part Ready Gate")
        self.part_ready_status_var = self._build_field(self.part_ready_frame, 0, "Status")
        self.part_ready_ratio_var = self._build_field(self.part_ready_frame, 1, "Match Ratio (avg5)")
        self.part_ready_raw_ratio_var = self._build_field(self.part_ready_frame, 2, "Match Ratio (raw)")
        self.part_ready_distance_var = self._build_field(self.part_ready_frame, 3, "Mean Distance")
        self.part_ready_profile_var = self._build_field(self.part_ready_frame, 4, "Profile")
        self.part_ready_threshold_var = self._build_field(self.part_ready_frame, 5, "Thresholds")

        self.sticker_frame = self._build_section("Sticker Validation")
        self.detected_class_var = self._build_field(self.sticker_frame, 0, "Detected Class")
        self.expected_class_var = self._build_field(self.sticker_frame, 1, "Expected Class")
        self.sticker_confidence_var = self._build_field(self.sticker_frame, 2, "Confidence")
        self.sticker_backend_var = self._build_field(self.sticker_frame, 3, "Backend")
        self.candidate_source_var = self._build_field(self.sticker_frame, 4, "Candidate Source")
        self.offset_var = self._build_field(self.sticker_frame, 5, "Offset")

        self.debug_frame = self._build_section("Inference Debug")
        self.raw_detection_count_var = self._build_field(self.debug_frame, 0, "Raw Detections")
        self.fallback_reason_var = self._build_field(self.debug_frame, 1, "Fallback Reason")
        self.classes_filter_var = self._build_field(self.debug_frame, 2, "Classes Filter")
        self.model_path_var = self._build_field(self.debug_frame, 3, "Model Path")
        self.device_var = self._build_field(self.debug_frame, 4, "Effective Device")
        self.response_mode_var = self._build_field(self.debug_frame, 5, "Response Mode")
        self.latency_total_var = self._build_field(self.debug_frame, 6, "Latency Total")
        self.latency_inference_var = self._build_field(self.debug_frame, 7, "Inference Time")

        self.commit_frame = self._build_section("Commit Details")
        self.reason_var = self._build_field(self.commit_frame, 0, "Reason")
        self.part_var = self._build_field(self.commit_frame, 1, "Part")
        self.line_var = self._build_field(self.commit_frame, 2, "Line")
        self.station_var = self._build_field(self.commit_frame, 3, "Station")
        self.db_var = self._build_field(self.commit_frame, 4, "DB Write")
        self.event_var = self._build_field(self.commit_frame, 5, "Event ID")
        self.commit_var = self._build_field(self.commit_frame, 6, "Committed At")
        self.bind("<Configure>", self._on_resize, add="+")

    def _build_section(self, title: str) -> ctk.CTkFrame:
        section = ctk.CTkFrame(self, fg_color=PANEL_ALT_BG, corner_radius=12, border_width=1, border_color=BORDER)
        section.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(section, text=title, font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).pack(anchor="w", padx=10, pady=(10, 6))
        body = ctk.CTkFrame(section, fg_color="transparent")
        body.pack(fill="x", padx=10, pady=(0, 10))
        body.grid_columnconfigure(1, weight=1)
        return body

    def _build_field(self, master, row: int, title: str) -> ctk.CTkLabel:
        master.columnconfigure(1, weight=1)
        ctk.CTkLabel(master, text=f"{title}:", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )
        value = ctk.CTkLabel(master, text="-", wraplength=210, justify="left", text_color=TEXT_SECONDARY)
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
        live_session = payload.get("session") or {}
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
        timings = payload.get("timings") or {}

        decision = display_validation.get("decision") or "WAITING"
        reason = display_validation.get("reject_reason_code") or ("OK" if decision == "ACCEPT" else "-")
        palette = {
            "ACCEPT": ("#166534", "#f0fdf4"),
            "REJECT": ("#991b1b", "#fef2f2"),
            "WAITING": ("#334155", "#f8fafc"),
        }
        bg, fg = palette.get(decision, ("#334155", "#f8fafc"))
        self.decision_banner.configure(fg_color=bg, text_color=fg, text=decision)
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
        self.live_template_version_var.configure(text=str(live_session.get("template_version_id") or "-"))

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
        threshold_text = "-"
        if display_part_ready.get("enabled", True):
            ratio_threshold = display_part_ready.get("min_match_ratio")
            distance_threshold = display_part_ready.get("distance_threshold")
            if ratio_threshold is not None or distance_threshold is not None:
                threshold_text = (
                    f"ratio>={_format_metric(ratio_threshold)} | "
                    f"distance<={_format_metric(distance_threshold)}"
                )
        self.part_ready_threshold_var.configure(text=threshold_text)

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
        model_path = (
            display_sticker_detection.get("model_path")
            or (display_details.get("model") or {}).get("model_path")
            or "-"
        )
        effective_device = str(display_sticker_detection.get("effective_device") or "-")
        device_backend = str(display_sticker_detection.get("device_backend") or "-")
        device_text = f"{effective_device} ({device_backend})"
        if display_sticker_detection.get("device_fallback_reason"):
            device_text = f"{device_text} | fallback: {display_sticker_detection.get('device_fallback_reason')}"
        response_mode = str(payload.get("response_mode") or "-")
        total_latency = timings.get("total_ms")
        inference_latency = timings.get("inference_ms")
        total_latency_text = f"{_format_metric(total_latency, precision=1)} ms" if total_latency is not None else "-"
        inference_latency_text = f"{_format_metric(inference_latency, precision=1)} ms" if inference_latency is not None else "-"
        self.raw_detection_count_var.configure(text=raw_count_text)
        self.fallback_reason_var.configure(text=str(fallback_reason))
        self.classes_filter_var.configure(text=classes_filter_text)
        self.model_path_var.configure(text=str(model_path))
        self.device_var.configure(text=device_text)
        self.response_mode_var.configure(text=response_mode)
        self.latency_total_var.configure(text=total_latency_text)
        self.latency_inference_var.configure(text=inference_latency_text)

        self.reason_var.configure(text=str(reason))
        self.part_var.configure(text=str(display_validation.get("part_name") or "-"))
        self.line_var.configure(text=str(display_validation.get("line_id") or "-"))
        self.station_var.configure(text=str(display_validation.get("station_id") or "-"))
        self.db_var.configure(text="OK" if db_write.get("written") else str(db_write.get("reason") or "-"))
        self.event_var.configure(text=str(committed.get("event_id") or payload.get("event_id") or "-"))
        self.commit_var.configure(text=_format_timestamp(committed.get("committed_at")))

    def reset(self) -> None:
        self.decision_banner.configure(fg_color="#334155", text_color="#f8fafc", text="WAITING")
        self.subtitle_var.configure(text="Menunggu event inspeksi pertama.")
        for widget in (
            self.live_state_var,
            self.live_decision_var,
            self.live_reason_var,
            self.live_template_version_var,
            self.part_ready_status_var,
            self.part_ready_ratio_var,
            self.part_ready_raw_ratio_var,
            self.part_ready_distance_var,
            self.part_ready_profile_var,
            self.part_ready_threshold_var,
            self.detected_class_var,
            self.expected_class_var,
            self.sticker_confidence_var,
            self.sticker_backend_var,
            self.candidate_source_var,
            self.offset_var,
            self.raw_detection_count_var,
            self.fallback_reason_var,
            self.classes_filter_var,
            self.model_path_var,
            self.device_var,
            self.response_mode_var,
            self.latency_total_var,
            self.latency_inference_var,
            self.reason_var,
            self.part_var,
            self.line_var,
            self.station_var,
            self.db_var,
            self.event_var,
            self.commit_var,
        ):
            widget.configure(text="-")

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from flask import Flask, g, request


def configure_logging(app: Flask) -> None:
    """Set up structured JSON-line logging and per-request correlation IDs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    @app.before_request
    def _assign_correlation_id():
        g.correlation_id = (
            request.headers.get("X-Correlation-Id")
            or request.headers.get("X-Request-Id")
            or uuid.uuid4().hex
        )
        g.request_start = datetime.now(UTC)

    @app.after_request
    def _log_request(response):
        cid = getattr(g, "correlation_id", "-")
        start = getattr(g, "request_start", None)
        duration_ms = (
            round((datetime.now(UTC) - start).total_seconds() * 1000, 1)
            if start else -1
        )
        logging.getLogger("qc.access").info(
            "method=%s path=%s status=%d duration_ms=%.1f cid=%s",
            request.method,
            request.path,
            response.status_code,
            duration_ms,
            cid,
        )
        response.headers["X-Correlation-Id"] = cid
        return response

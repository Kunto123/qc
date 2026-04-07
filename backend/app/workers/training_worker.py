"""training_worker.py — background daemon that drives training job state transitions.

State machine:
    queued → running → completed | failed
    Any non-terminal state → cancelled (on user request via cancel_job())

The worker does not perform real ML training; it simulates the lifecycle so the
state machine contract is honoured.  Real training integration would replace the
``_run_job`` method body.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 5  # seconds between queue checks


class TrainingWorker:
    def __init__(self, training_repo) -> None:
        self._repo = training_repo
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="training-worker", daemon=True)
        self._thread.start()
        logger.info("[training-worker] started")

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process_queue()
            except Exception:  # noqa: BLE001
                logger.exception("[training-worker] unhandled error in loop")
            self._stop_event.wait(timeout=_POLL_INTERVAL)

    def _process_queue(self) -> None:
        jobs = self._repo.list_jobs()
        for job in jobs:
            if job["status"] == "queued":
                self._run_job(job)

    def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        logger.info("[training-worker] starting job %s", job_id)
        try:
            self._repo.transition(
                job_id,
                "running",
                log_line=f"Job started at {datetime.now(UTC).isoformat()}",
            )
        except ValueError:
            return  # already transitioned (race or cancelled)

        # --- real ML training would go here ---
        try:
            time.sleep(2)  # simulate short work
            trained_path = f"models/trained/{job['dataset_id']}__{job['base_model']}__{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.pt"
            self._repo.transition(
                job_id,
                "completed",
                trained_model_path=trained_path,
                log_line=f"Training completed. Model saved to: {trained_path}",
            )
            logger.info("[training-worker] job %s completed", job_id)
        except ValueError:
            pass  # job was cancelled mid-run
        except Exception as exc:  # noqa: BLE001
            logger.exception("[training-worker] job %s failed: %s", job_id, exc)
            try:
                self._repo.transition(
                    job_id,
                    "failed",
                    error=str(exc),
                    log_line=f"Training failed: {exc}",
                )
            except ValueError:
                pass

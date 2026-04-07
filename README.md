# QC Suite Python

Greenfield QC suite in a separate folder, built as:

- `backend/`: Flask API for auth, templates, deployments, inspection sessions, workstation, and dashboard
- `client_tk/`: Tkinter desktop shell with role-based screens for Operator, Admin, and Engineer
- `shared/`: contracts and enums shared across backend and client
- `docs/`: product, screens, API, and deployment notes
- `scripts/`: run and smoke-test helpers

Default seeded users:

- `admin / admin123`
- `operator / operator123`
- `engineer / engineer123`

## Role Screens

- `Operator`: login, local camera, active deployment lookup, ROI update, live decision, DB write status
- `Admin`: templates, deployments, users, inspection results, dashboard
- `Engineer`: dataset upload, annotations, augment jobs, training jobs, model registry, color calibration

## Quick Start

```powershell
cd qc-suite-python
py -3.11 -m pip install -e .
py -3.11 scripts/run_backend.py
```

In another terminal:

```powershell
cd qc-suite-python
py -3.11 scripts/run_client.py
```

Smoke-check the backend contracts without touching the legacy app:

```powershell
cd qc-suite-python
py -3.11 scripts/smoke_api.py
py -3.11 -m unittest backend.tests.test_api_smoke
```

## Notes

- Existing runtime in the repo is untouched.
- This project uses JSON/file storage by default, with optional SQL Server persistence for inspection results when `MSSQL_*` env vars are configured.
- Sticker inference Phase 3 is wired to `D:\ProjectMagang\akh.pt` with metadata from `D:\ProjectMagang\ds-43598c556c__yolov5mu__20260402-085412.meta.json`.
- Runtime mode is controlled by `QC_SUITE_STICKER_INFERENCE_MODE`:
  - `auto`: try Ultralytics first, fallback to classic contour inference
  - `ultralytics`: require the YOLO runtime and fail if unavailable
  - `classic`: deterministic fallback for smoke tests and local debugging
- Dashboard summary and time buckets now aggregate persisted Phase 5 fields, including `station_id`, `sticker_backend`, `total_part_ready`, `avg_sticker_confidence`, and `avg_part_ready_match_ratio`.
- `GET /deployments/active` returns `{ "deployment": ... }` so the client can distinguish between no deployment and a valid deployment deterministically.

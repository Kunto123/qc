# API Contract

## Auth

- `POST /auth/login`
  - request: `{ "username": "...", "password": "..." }`
  - response: `{ "token": "...", "user": {...}, "expires_in": 86400 }`
- `GET /auth/me`
  - response: current authenticated user

## Templates

- `GET /templates`
- `GET /templates/{template_id}`
- `POST /templates`
- `PUT /templates/{template_id}`
- `DELETE /templates/{template_id}`

Template payload membawa konfigurasi tetap untuk:

- `camera`
- `roi`
- `vision`
- `part_ready`
- `sticker`
- `persistence`

## Deployments

- `POST /deployments`
- `GET /deployments`
- `GET /deployments/active?line_id=...&station_id=...`
  - response selalu berbentuk `{ "deployment": object | null }`
- `DELETE /deployments/{deployment_id}`

## Inspection Runtime

- `POST /inspection/sessions/start`
  - request: `{ "client_id", "camera_index", "template_version_id", "line_id", "station_id" }`
- `POST /inspection/sessions/{session_id}/frame`
  - request: `{ "image_b64": "..." }`
  - response:
    - `session`
    - `roi`
    - `detections`
    - `part_ready`
    - `validation`
    - `db_write`
    - `overlay_image_b64`
    - `preview_image_b64`
- `POST /inspection/sessions/{session_id}/roi`
- `POST /inspection/sessions/{session_id}/stop`

## Results and Dashboard

- `GET /inspections`
- `GET /inspections/{result_id}`
- `GET /dashboard/summary`
- `GET /dashboard/buckets`

## Workstation

- `GET /datasets`
- `POST /datasets`
- `DELETE /datasets/{dataset_id}`
- `GET /datasets/{dataset_id}/files`
- `POST /datasets/{dataset_id}/upload`
- `GET /datasets/{dataset_id}/annotations/{image_name}`
- `POST /datasets/{dataset_id}/annotations/{image_name}`
- `GET /augment/jobs`
- `POST /augment/jobs`
- `GET /train/jobs`
- `POST /train/jobs`
- `POST /train/jobs/{job_id}/cancel`
- `GET /models`
- `POST /models`

## Calibration

- `POST /calibration/color-profile`
- `GET /calibration/profiles`
- `POST /calibration/profiles`
- `DELETE /calibration/profiles/{profile_id}`

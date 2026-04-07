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

## Cara Menjalankan

README ini memakai contoh Windows + PowerShell.

### 1. Install prasyarat

- Install Python `3.11`
- Pastikan `pip` aktif
- Jika akan pakai SQL Server, install driver ODBC yang sesuai, default repo ini memakai `ODBC Driver 17 for SQL Server`

### 2. Install dependency project

```powershell
cd qc-suite-python
py -3.11 -m pip install -e .
```

### 3. Siapkan environment

Salin `deploy/.env.example` menjadi `.env`, lalu isi sesuai kebutuhan environment Anda.

Minimal untuk local run:

```env
QC_SUITE_HOST=127.0.0.1
QC_SUITE_PORT=8100
QC_SUITE_DEBUG=0
QC_SUITE_SECRET_KEY=ganti-secret-anda
QC_SUITE_SERVER_URL=http://127.0.0.1:8100
```

Jika model production belum siap, untuk smoke test lokal Anda bisa memakai:

```env
QC_SUITE_STICKER_INFERENCE_MODE=classic
```

Jika ingin memakai SQL Server, isi juga:

```env
MSSQL_SERVER=...
MSSQL_DATABASE=...
MSSQL_USERNAME=...
MSSQL_PASSWORD=...
MSSQL_DRIVER=ODBC Driver 17 for SQL Server
```

### 4. Jalankan backend

```powershell
cd qc-suite-python
py -3.11 scripts/run_backend.py
```

Backend default akan jalan di `http://127.0.0.1:8100`.

### 5. Jalankan client

Di terminal lain:

```powershell
cd qc-suite-python
py -3.11 scripts/run_client.py
```

### 6. Login ke aplikasi

Gunakan akun seed bawaan:

- `admin / admin123`
- `operator / operator123`
- `engineer / engineer123`

### 7. Smoke check lokal

Smoke-check backend tanpa menyentuh runtime lama:

```powershell
cd qc-suite-python
py -3.11 scripts/smoke_api.py
py -3.11 -m unittest backend.tests.test_api_smoke
```

Untuk smoke test UI:

```powershell
cd qc-suite-python
py -3.11 -m unittest client_tk.tests.test_ui_smoke
```

## Menjalankan Server dan Client di PC Berbeda

Arsitektur aplikasi ini memang mendukung runtime split:

- backend berjalan di PC server
- client Tkinter berjalan di PC operator/admin/engineer
- kamera tetap dibuka di sisi client
- frame dikirim dari client ke backend lewat HTTP

### 1. Siapkan PC server

Set environment backend di PC server:

```env
QC_SUITE_HOST=0.0.0.0
QC_SUITE_PORT=8100
QC_SUITE_DEBUG=0
QC_SUITE_SECRET_KEY=ganti-secret-produksi
QC_SUITE_DATA_ROOT=D:\qc-suite-data
QC_SUITE_STICKER_INFERENCE_MODE=auto
QC_SUITE_DEFAULT_STICKER_MODEL_PATH=D:\qc-suite-data\models\sticker.pt
QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH=D:\qc-suite-data\models\sticker.meta.json
```

Jika SQL Server dipakai:

```env
MSSQL_SERVER=...
MSSQL_DATABASE=...
MSSQL_USERNAME=...
MSSQL_PASSWORD=...
MSSQL_DRIVER=ODBC Driver 17 for SQL Server
```

### 2. Jalankan backend di PC server

Untuk development/internal testing:

```powershell
cd qc-suite-python
py -3.11 scripts/run_backend.py
```

Untuk deploy yang lebih layak produksi, jangan pakai Flask dev server. Jalankan backend lewat `waitress`:

```powershell
cd qc-suite-python
py -3.11 -m pip install waitress
waitress-serve --host 0.0.0.0 --port 8100 backend.app.main:app
```

### 3. Buka akses jaringan

- pastikan PC server punya IP yang bisa diakses client, misalnya `192.168.1.10`
- buka firewall Windows untuk port `8100`
- pastikan client dan server berada di jaringan yang saling terhubung

### 4. Siapkan PC client

Set `QC_SUITE_SERVER_URL` di PC client agar mengarah ke IP server:

```env
QC_SUITE_SERVER_URL=http://192.168.1.10:8100
```

Lalu jalankan client:

```powershell
cd qc-suite-python
py -3.11 -m pip install -e .
py -3.11 scripts/run_client.py
```

### 5. Verifikasi koneksi

Setelah backend dan client aktif:

- login dari client
- buka screen `Admin` untuk cek data terbaca
- buka screen `Operator`, load deployment, lalu start camera
- pastikan frame dari client bisa diproses backend server

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

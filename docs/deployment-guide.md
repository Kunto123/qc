# Deployment Guide

## Local Development

1. Install Python 3.11.
2. Install dependencies:

```powershell
cd qc-suite-python
py -3.11 -m pip install -e .
```

3. Start backend:

```powershell
py -3.11 scripts/run_backend.py
```

4. Start desktop client on another terminal:

```powershell
py -3.11 scripts/run_client.py
```

## Runtime Split

- Backend berjalan di server Windows/Linux yang punya akses ke data root, model files, dan optional SQL Server.
- Client Tkinter berjalan di workstation operator/admin/engineer.
- Kamera tetap dibuka di client dan frame dikirim ke backend lewat HTTP.

## Environment

Lihat `deploy/.env.example`.

Nilai penting:

- `QC_SUITE_HOST`
- `QC_SUITE_PORT`
- `QC_SUITE_DATA_ROOT`
- `QC_SUITE_SERVER_URL`
- `QC_SUITE_STICKER_INFERENCE_MODE`
- `QC_SUITE_DEFAULT_STICKER_MODEL_PATH`
- `QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH`
- `MSSQL_SERVER`
- `MSSQL_DATABASE`
- `MSSQL_USERNAME`
- `MSSQL_PASSWORD`
- `MSSQL_DRIVER`

## Recommended Production Values

- `QC_SUITE_STICKER_INFERENCE_MODE=auto`
- `QC_SUITE_DEFAULT_STICKER_MODEL_PATH` diarahkan ke model sticker production yang aktif
- `QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH` diarahkan ke metadata model yang sesuai

## Post-Deploy Smoke

Setelah backend dan client naik:

1. Login sebagai `admin`, `operator`, dan `engineer`.
2. Buka screen masing-masing dan pastikan tidak ada crash saat init.
3. Dari `Operator`, load deployment lalu start camera dan session.
4. Pastikan preview `Part Ready ROI` dan `Sticker ROI` muncul.
5. Dari `Admin`, cek `Results` dan `Dashboard` dengan filter `line/station`.
6. Dari `Engineer`, buka `Models` dan `Calibration` untuk memastikan registry dan profile list terbaca.

Untuk smoke API non-GUI:

```powershell
cd qc-suite-python
py -3.11 scripts/smoke_api.py
```

# Migration Notes

## What Stays

- Pipeline domain tetap: `camera -> ROI -> main vision -> part validator -> sticker validator -> DB writer -> display`
- Template/deployment tetap menjadi sumber konfigurasi operator.
- Workstation tetap punya fungsi dataset, annotate, augment, train, model registry, dan color calibration.

## What Changes

- Tidak ada canvas/node editor umum.
- Frontend menjadi desktop Tkinter fixed-screen.
- Runtime dibagi:
  - client untuk camera capture dan display
  - backend untuk inference pipeline dan persistence
- Legacy folder lama tetap tidak disentuh.

## Current MVP Limits

- Runtime sticker memakai adapter ML nyata secara default bila model tersedia, dengan fallback `classic` untuk smoke test dan keadaan darurat.
- Storage default adalah JSON/filesystem.
- SQL Server baru dipakai untuk inspection results jika env `MSSQL_*` diisi.

## SQL Server Notes

Jika project lama sudah punya tabel hasil inspeksi sendiri, jangan asumsi schema lama sudah cukup. Repository SQL Server project ini mengharapkan tabel `dbo.qc_inspection_results` dengan kolom tambahan berikut:

- `station_id`
- `part_ready_status`
- `part_ready_match_ratio`
- `part_ready_distance`
- `detected_class`
- `expected_class`
- `sticker_confidence`
- `sticker_backend`
- `sticker_bbox_json`
- `validation_details_json`
- `part_ready_roi_meta_json`
- `sticker_roi_meta_json`

Repository SQL Server backend saat startup akan:

- membuat tabel `dbo.qc_inspection_results` jika belum ada
- menambah kolom yang hilang dengan `ALTER TABLE ... ADD ...` bila tabel sudah ada

Implikasi operasional:

- akun SQL Server yang dipakai backend harus punya izin `CREATE TABLE` dan `ALTER TABLE` saat first run
- jika production policy melarang auto-DDL, jalankan migration secara manual sebelum backend start
- payload JSON seperti `validation_details_json` dan ROI meta akan disimpan utuh untuk audit/debug

## Hardening Notes

Checklist minimal sebelum cutover:

- pastikan `QC_SUITE_DEFAULT_STICKER_MODEL_PATH` dan `QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH` mengarah ke file model yang valid di server
- pastikan `QC_SUITE_STICKER_INFERENCE_MODE=auto` atau `ultralytics` di production, bukan `classic`
- jalankan smoke test API setelah deploy
- uji login dan screen init untuk `operator`, `admin`, dan `engineer`
- verifikasi filter `line_id/station_id/template_version_id` pada results dan dashboard

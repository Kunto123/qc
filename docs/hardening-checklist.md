# Hardening Checklist

## Automated Checks

Jalankan ini sebelum merge atau deploy:

```powershell
cd qc-suite-python
py -3.11 -m compileall client_tk
py -3.11 -m unittest backend.tests.test_api_smoke
py -3.11 -m unittest client_tk.tests.test_ui_smoke
py -3.11 scripts/smoke_api.py
```

## Operator Manual QA

- Login sebagai `operator`
- `Load Deployment`
- `Start Camera`
- cek `Inspection Camera`
- cek preview `Part Ready ROI`
- cek preview `Sticker ROI`
- `Start Session`
- verifikasi state `Part Ready Gate`
- verifikasi `Sticker Validation`
- verifikasi counter `Total / Accept / Reject`
- verifikasi satu part yang diam lama tidak double count
- ubah dua ROI di `Settings`, klik `Apply ROI`, cek crop preview berubah

## Admin Manual QA

- buka `Templates`, load template existing, edit structured form, preview raw JSON
- save template dan pastikan `version_number` naik
- buka `Deployments`, assign template ke `line/station`
- deactivate deployment lalu cek `active deployment` menjadi kosong
- buka `Users`, create user baru, disable user, pastikan login gagal, enable lagi
- buka `Results`, filter `line/station/template`, buka detail
- buka `Dashboard`, cek KPI cards dan bucket trend sesuai filter

## Engineer Manual QA

- buat dataset baru
- upload image ke `images`
- cek file browser menampilkan file tersebut
- load/save annotation JSON
- buat augment job
- buat training job
- buka detail training job
- register model lengkap dengan `meta_path`, runtime, task, dan class names
- compute profile dari image calibration
- save profile lalu cek profile muncul di list
- delete profile dan pastikan hilang dari list

## SQL Server Checklist

- pastikan akun SQL Server punya izin `CREATE TABLE` dan `ALTER TABLE` saat first run
- pastikan tabel `dbo.qc_inspection_results` memiliki kolom dua-ROI dan validator fields terbaru
- verifikasi insert hasil inspeksi berhasil
- verifikasi `line_id`, `station_id`, `part_ready_status`, `sticker_backend`, dan `validation_details_json` benar-benar terisi

## Release Gate

Phase 8 dianggap lulus jika:

- regression test backend lulus
- UI smoke test lulus
- smoke API lulus
- operator/admin/engineer screens tidak crash saat init
- SQL Server migration path sudah dipahami dan diverifikasi untuk environment target

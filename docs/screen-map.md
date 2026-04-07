# Screen Map

## Operator

- Login ke backend dengan role `operator`.
- Load deployment aktif berdasarkan `line_id` dan `station_id`.
- Start kamera lokal client.
- Start inspection session ke server.
- Update ROI saat runtime.
- Lihat dua stream:
  - `Client Camera`: frame lokal sebelum dikirim.
  - `Server Overlay`: hasil ROI + overlay keputusan server.
- Lihat result panel:
  - decision
  - reject reason
  - part ready
  - part name
  - line
  - DB write status

## Admin

- `Templates`: master-detail editor dengan structured form dua ROI dan raw JSON advanced tab.
- `Deployments`: assign template version ke `line/station`, plus deactivate deployment aktif.
- `Users`: tambah user baru, lihat role, enable/disable akun.
- `Results`: filter by line/station/part/template/decision, lihat summary evidence dan raw payload.
- `Dashboard`: filter agregasi, KPI cards, dan bucket trend list.

## Engineer

- `Data`: dataset list, upload file, file browser, dan annotation workflow.
- `Training`: augment jobs terpisah dari training jobs sticker detector.
- `Models`: daftar dan register model lengkap dengan `meta_path`, runtime, task, class names, dan arsitektur.
- `Calibration`: hitung color profile untuk part-ready, optional ROI crop, lihat saved profiles, dan delete profile.

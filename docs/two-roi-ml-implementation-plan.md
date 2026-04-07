# Two-ROI ML Implementation Plan

## Tujuan
Dokumen ini menjabarkan rencana implementasi rinci untuk mengubah pipeline inspeksi saat ini dari model `single ROI` menjadi `two ROI` dengan pembagian fungsi yang tegas:

- `part_ready_roi`: mengecek apakah part sudah hadir di lokasi inspeksi menggunakan referensi warna.
- `sticker_roi`: menjalankan inferensi machine learning untuk mendeteksi dan memvalidasi sticker sebelum hasil di-commit ke database.

Keputusan arsitektur yang dipakai:

- `part_ready` memakai `color reference matching`, bukan machine learning.
- `sticker validation` tetap memakai `machine learning`.
- `DB write` hanya terjadi setelah hasil final sudah committed.
- `accept/reject counter` tetap dihitung per event fisik, bukan per frame.

## Status Implementasi

- `Phase 0 - Dokumentasi dan Schema Freeze`: selesai
- `Phase 1 - Shared Contract Refactor`: selesai
- `Phase 2 - Color Reference Gate`: selesai
- `Phase 3 - ML Sticker Adapter`: selesai
- `Phase 4 - Sticker Validator Rules`: selesai
- `Phase 5 - Persistence dan Dashboard`: selesai
- `Phase 6 - Operator UX`: selesai
- `Phase 7 - Admin dan Engineer UX`: selesai
- `Phase 8 - Hardening`: selesai

## Kondisi Sistem Saat Ini
Saat ini proyek `qc-suite-python` sudah berada pada kondisi berikut:

- template sudah memisahkan `part_ready_roi` dan `sticker_roi`
- `inspection_session.py` sudah menjalankan gate warna part-ready sebelum inferensi sticker
- inference sticker sudah memakai adapter ML dengan fallback klasik untuk test/smoke
- response runtime sudah memuat `part_ready`, `sticker_detection`, `validation`, preview dua ROI, dan evidence commit
- operator UI sudah memiliki settings dua ROI terpisah, preview crop dua ROI, serta panel status dua tahap

Area pekerjaan besar yang tersisa sekarang bergeser ke:

- hardening integrasi dan regression coverage
- QA UI manual di resolusi client yang berbeda

Catatan phase 8 yang sudah ditutup:

- regression suite backend diperluas untuk admin/engineer flow
- UI smoke test ditambahkan untuk init `Operator`, `Admin`, dan `Engineer`
- migration notes SQL Server diperjelas dengan kolom dua-ROI dan validator fields
- deployment guide dan hardening checklist sekarang mencakup post-deploy smoke

## Arsitektur Target
Pipeline target:

1. Kamera client menangkap frame.
2. Backend crop `part_ready_roi`.
3. Backend melakukan `color reference match`.
4. Jika `part_ready = false`, event tetap di state non-final.
5. Jika `part_ready = true`, backend crop `sticker_roi`.
6. Backend menjalankan model machine learning pada `sticker_roi`.
7. Validator membandingkan output model terhadap aturan template.
8. Event distabilkan lalu di-commit satu kali.
9. Hasil final ditulis ke database.
10. UI menampilkan `live state`, `last committed result`, dan counter.

## Peran Machine Learning
Machine learning tetap dipakai, tetapi hanya untuk tahap sticker.

Rekomendasi model:

- `object detection model`, bukan classifier murni

Alasan:

- sistem perlu tahu apakah sticker ditemukan atau tidak
- sistem perlu tahu tipe sticker yang terdeteksi
- sistem perlu tahu posisi sticker terhadap ROI
- hasil perlu memiliki confidence yang bisa dipakai validator

Output minimum model:

- `bbox`
- `class_name`
- `class_id`
- `confidence`

## Cara Sistem Mendeteksi Sticker Sebelum Masuk DB
Urutan logika yang akan dipakai:

1. `part_ready_roi` dievaluasi terhadap color profile.
2. Jika part belum ready, sistem tidak boleh meng-commit hasil sticker final.
3. Jika part sudah ready, sistem crop `sticker_roi`.
4. Model ML dijalankan pada `sticker_roi`.
5. Output model divalidasi terhadap aturan template:
   - apakah object ditemukan
   - apakah `detected_class == expected_class`
   - apakah `confidence >= min_class_confidence`
   - apakah posisi sticker masih dalam `max_offset_x/max_offset_y`
6. Sistem menghasilkan keputusan:
   - `ACCEPT`
   - `REJECT: NOT_FOUND`
   - `REJECT: WRONG_TYPE`
   - `REJECT: LOW_CLASS_CONF`
   - `REJECT: OUT_OF_POSITION`
7. Event di-commit satu kali.
8. Counter bertambah satu kali.
9. Hasil final baru masuk database.

## Desain Data Target

### 1. Template
Template perlu dipisah menjadi dua ROI dan dua domain konfigurasi.

Struktur target:

```json
{
  "id": 1,
  "version_id": 11,
  "version_number": 3,
  "name": "QC Line A",
  "description": "Two-ROI inspection template",
  "is_active": true,
  "camera": {
    "camera_index": 0,
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "part_ready_roi": {
    "x": 0.08,
    "y": 0.18,
    "w": 0.24,
    "h": 0.22
  },
  "sticker_roi": {
    "x": 0.42,
    "y": 0.26,
    "w": 0.24,
    "h": 0.20
  },
  "vision": {
    "model_path": "models/sticker-line-a.pt",
    "conf_threshold": 0.25,
    "stream_fps": 10.0,
    "inference_fps": 4.0,
    "imgsz": 640,
    "classes": [
      "sticker_type_a",
      "sticker_type_b"
    ]
  },
  "part_ready": {
    "enabled": true,
    "color_profile_id": 3,
    "colorspace": "LAB",
    "distance_threshold": 12.0,
    "min_match_ratio": 0.82
  },
  "sticker": {
    "enabled": true,
    "validator_mode": "ml_detection",
    "part_name": "Part A",
    "expected_class": "sticker_type_a",
    "line": "Line 1",
    "min_roi_confidence": 0.25,
    "min_class_confidence": 0.70,
    "max_offset_x": 24,
    "max_offset_y": 18
  },
  "persistence": {
    "write_to_db": true
  },
  "metadata": {
    "station_family": "assy_front"
  }
}
```

### 2. Inspection Result
Payload hasil inspeksi perlu menyimpan bukti dari dua tahap:

- `part_ready_status`
- `part_ready_match_ratio`
- `part_ready_distance`
- `detected_class`
- `expected_class`
- `sticker_confidence`
- `sticker_bbox`
- `part_ready_roi_meta`
- `sticker_roi_meta`
- `decision`
- `reject_reason_code`

### 3. Session State
State runtime perlu melacak tiga hal yang berbeda:

- state kehadiran part
- state inferensi sticker
- state commit event

## Breakdown Per Modul

### Shared Contracts

#### `shared/contracts/templates.py`
Perubahan:

- ubah `InspectionTemplate` dari satu `roi` menjadi:
  - `part_ready_roi`
  - `sticker_roi`
- pertahankan `vision` sebagai konfigurasi inference ML sticker agar perubahan tidak terlalu destruktif
- perluas `PartReadyConfig`
- perluas `StickerRule`

Detail perubahan:

- `PartReadyConfig`
  - tambah `enabled: bool`
  - tambah `colorspace: str`
  - tambah `distance_threshold: float | None`
  - pertahankan `color_profile_id`
  - pertahankan `min_match_ratio`
- `StickerRule`
  - tambah `enabled: bool`
  - tambah `validator_mode: str = "ml_detection"`
  - pertahankan `expected_class`
  - pertahankan `min_roi_confidence`
  - pertahankan `min_class_confidence`
  - pertahankan `max_offset_x`
  - pertahankan `max_offset_y`
- `template_from_dict(...)`
  - support format baru
  - sediakan fallback migrasi dari `roi` lama ke `sticker_roi`

Acceptance:

- template lama masih bisa dibaca
- template baru bisa memisahkan dua ROI

#### `shared/contracts/inspection.py`
Perubahan:

- perluas `InspectionResult`
- tambahkan field bukti dua tahap inspeksi

Field yang disarankan:

- `part_ready_status: str | None`
- `part_ready_match_ratio: float | None`
- `part_ready_distance: float | None`
- `detected_class: str | None`
- `expected_class: str | None`
- `sticker_confidence: float | None`
- `sticker_bbox: dict[str, float] | None`

Acceptance:

- repository JSON dan SQL masih dapat menyimpan hasil yang diperluas

#### `shared/contracts/enums.py`
Perubahan:

- pertahankan enum keputusan yang ada
- tambahkan enum state runtime dua tahap jika perlu

Usulan enum:

- `PartReadyState`
  - `missing`
  - `checking`
  - `ready`
- `StickerValidationState`
  - `skipped`
  - `detecting`
  - `accepted`
  - `rejected`

Acceptance:

- payload lebih eksplisit, tidak bergantung pada string bebas

### Backend Service Layer

#### `backend/app/services/template_runtime.py`
Perubahan:

- validasi template runtime agar dua ROI tersedia
- sediakan normalisasi untuk template lama
- sediakan helper resolusi model ML dan profile warna

Usulan fungsi tambahan:

- `normalize_template(...)`
- `resolve_sticker_model(...)`
- `resolve_part_ready_profile(...)`

Acceptance:

- runtime service menjadi tempat tunggal untuk interpretasi template

#### `backend/app/services/calibration.py`
Perubahan:

- pertahankan fungsi pembuatan color profile
- tambahkan utilitas untuk evaluasi ROI terhadap color profile

Usulan fungsi tambahan:

- `evaluate_color_match(image, profile, colorspace="LAB")`
- `extract_roi(image, roi)`

Output evaluator:

- `match_ratio`
- `mean_distance`
- `is_match`
- `debug_stats`

Acceptance:

- `part_ready` dapat dihitung tanpa duplikasi logika di `inspection_session.py`

#### `backend/app/services/inspection_session.py`
Ini modul paling besar yang harus direfactor.

Perubahan utama:

- ganti `_crop_roi(...)` menjadi dua flow:
  - `_crop_part_ready_roi(...)`
  - `_crop_sticker_roi(...)`
- ganti `_run_dummy_vision(...)` dengan adapter model ML
- pisahkan validator menjadi:
  - `_evaluate_part_ready(...)`
  - `_run_sticker_model(...)`
  - `_validate_sticker_detection(...)`
  - `_should_commit_event(...)`

Urutan method target:

1. decode frame
2. crop `part_ready_roi`
3. evaluate part-ready
4. jika part belum ready:
   - skip ML sticker
   - return state non-final
5. crop `sticker_roi`
6. jalankan model ML
7. validasi hasil model
8. update state machine
9. persist jika committed
10. compose overlay

Refactor detail:

- `process_frame(...)`
  - jangan lagi mengasumsikan satu ROI utama
  - response harus membawa metadata dua ROI
- `_evaluate_part_ready(...)`
  - ambil profile dari `ProfilesRepository`
  - hitung `match_ratio` dan `distance`
  - output:
    - `ready`
    - `match_ratio`
    - `distance`
    - `reason`
- `_run_sticker_model(...)`
  - tahap awal boleh memakai adapter/wrapper
  - implementasi target memanggil model ML nyata
- `_validate_sticker_detection(...)`
  - hasil reject harus jelas dan deterministik
- `_compose_overlay(...)`
  - tampilkan dua kotak ROI
  - tampilkan state `part_ready`
  - tampilkan hasil class sticker

Acceptance:

- satu part hanya di-commit satu kali
- sticker tidak divalidasi final saat part belum ready
- payload runtime memisahkan dua tahap inspeksi

#### `backend/app/services/training.py`
Perubahan:

- tambahkan metadata training khusus sticker detector
- pastikan training job menyimpan:
  - dataset sumber
  - kelas sticker
  - path model output
  - metrics evaluasi

Acceptance:

- model registry bisa ditautkan ke template sticker

### Backend Repository Layer

#### `backend/app/repositories/profiles_repository.py`
Perubahan:

- profil warna perlu mendukung domain `part_ready`
- simpan metadata tambahan:
  - `colorspace`
  - `reference_source`
  - `distance_threshold`
  - `min_match_ratio`

Acceptance:

- satu profile bisa dipakai ulang lintas template

#### `backend/app/repositories/models_repository.py`
Perubahan:

- jadikan model registry sebagai sumber resmi model sticker
- tambahkan metadata:
  - `model_type`
  - `task`
  - `class_names`
  - `model_path`
  - `status`
  - `metrics`

Acceptance:

- template tidak harus menyimpan path mentah tanpa validasi

#### `backend/app/repositories/templates_repository.py`
Perubahan:

- support penyimpanan template dua ROI
- support migrasi otomatis template lama
- tambahkan validasi schema minimal

Acceptance:

- create, update, dan list template tidak memutus data lama

#### `backend/app/repositories/inspection_results_repository.py`
Perubahan:

- persist field baru hasil dua tahap
- update `summary(...)` dan `buckets(...)` jika perlu untuk breakdown `PART_NOT_READY`

Field baru yang sebaiknya dipersist:

- `part_ready_status`
- `part_ready_match_ratio`
- `part_ready_distance`
- `detected_class`
- `expected_class`
- `sticker_confidence`
- `sticker_bbox`
- `part_ready_roi_meta`
- `sticker_roi_meta`

Acceptance:

- dashboard dan detail result dapat menjelaskan alasan keputusan dengan lebih lengkap

#### `backend/app/repositories/sqlserver/inspection_results_repository.py`
Perubahan:

- samakan schema penulisan SQL Server dengan repository JSON
- jika tabel SQL belum punya kolom baru, siapkan migration note

Acceptance:

- implementasi JSON dan SQL tidak drift terlalu jauh

### Backend API Layer

#### `backend/app/api/template_routes.py`
Perubahan:

- endpoint create/update template harus menerima dua ROI
- validasi payload baru
- balikan response template ter-normalisasi

Acceptance:

- Admin/Engineer dapat mengelola template dua ROI dari UI

#### `backend/app/api/calibration_routes.py`
Perubahan:

- tambah dukungan kalibrasi `part_ready_roi`
- endpoint preview profile harus bisa memproses crop ROI part-ready

Acceptance:

- kalibrasi warna tidak lagi generik tanpa konteks ROI

#### `backend/app/api/inspection_routes.py`
Perubahan:

- response `POST /inspection/sessions/{id}/frame` harus memuat:
  - `part_ready`
  - `sticker_detection`
  - `validation`
  - `event_state`
  - `count_committed`
  - `counters`
  - `part_ready_roi_meta`
  - `sticker_roi_meta`
- update endpoint ROI agar bisa mengubah:
  - `part_ready_roi`
  - `sticker_roi`

Acceptance:

- client operator tidak perlu menebak state backend

#### `backend/app/api/workstation_routes.py`
Perubahan:

- workflow dataset/annotation/training perlu eksplisit bahwa task ML hanya untuk sticker
- training form perlu memilih:
  - dataset
  - classes
  - output model name

Acceptance:

- engineer punya jalur kerja ML yang jelas dan tidak campur dengan color calibration

#### `backend/app/api/dashboard_routes.py`
Perubahan:

- tambahkan breakdown reject baru bila dibutuhkan
- siapkan filter untuk `station_id`

Acceptance:

- dashboard konsisten dengan data hasil dua tahap

### Client Tk Layer

#### `client_tk/app/api_client.py`
Perubahan:

- update contract client untuk template dua ROI
- tambah helper pengiriman ROI terpisah

Usulan method:

- `update_part_ready_roi(...)`
- `update_sticker_roi(...)`
- atau satu method `update_rois(...)`

Acceptance:

- operator dan engineer bisa mengubah kedua ROI tanpa ambiguity

#### `client_tk/app/services/session_state.py`
Perubahan:

- simpan `part_ready` live state
- simpan `sticker_detection`
- simpan `last_committed_result`

Acceptance:

- state client rapi dan tidak hanya berupa response mentah terakhir

#### `client_tk/app/screens/operator/view.py`
Perubahan:

- tampilkan dua status besar:
  - `Part Ready`
  - `Sticker Decision`
- tampilkan dua ROI overlay
- tampilkan template selector
- settings dialog harus memiliki dua editor ROI

Layout yang disarankan:

- area tengah: live camera utama
- kanan: decision panel + counter + recent events
- kanan atas: template selector
- kiri atas: settings

Field operator yang wajib tampil:

- `part_ready: READY / NOT READY`
- `match_ratio`
- `detected_class`
- `expected_class`
- `decision`
- `reject_reason`
- `db_write_status`

Acceptance:

- operator dapat memahami dua tahap inspeksi tanpa membuka JSON

#### `client_tk/app/components/live_view.py`
Perubahan:

- dukung rendering overlay dua ROI
- sediakan label kecil untuk source view:
  - `client preview`
  - `server overlay`

Acceptance:

- visual debugging lebih jelas

#### `client_tk/app/components/result_panel.py`
Perubahan:

- pisahkan informasi menjadi dua blok:
  - `part ready`
  - `final sticker decision`

Acceptance:

- status reject tidak bercampur dengan state kehadiran part

#### `client_tk/app/components/counter_panel.py`
Perubahan:

- pertahankan counter session
- tampilkan breakdown reject yang relevan
- pertimbangkan entry `PART_NOT_READY` bila event tersebut memang disimpan sebagai reject final

Acceptance:

- operator bisa membaca total produksi secara cepat

#### `client_tk/app/components/template_forms.py`
Perubahan:

- form template harus punya section terpisah:
  - camera
  - part-ready ROI
  - part-ready config
  - sticker ROI
  - sticker ML config
  - persistence

Acceptance:

- Admin/Engineer tidak lagi mengedit template sebagai JSON mentah saja

#### `client_tk/app/screens/admin/view.py`
Perubahan:

- template editor perlu support dua ROI
- deployment manager tetap memakai template version
- results page perlu menampilkan:
  - part-ready evidence
  - sticker evidence

Acceptance:

- Admin bisa audit hasil dengan konteks lengkap

#### `client_tk/app/screens/engineer/view.py`
Perubahan:

- calibration flow dipisah jelas dari training flow
- tambahkan alur kerja yang eksplisit:
  - buat color profile untuk part-ready
  - pilih dataset sticker
  - train model sticker
  - register model
  - masukkan model ke template

Acceptance:

- engineer tidak mencampur domain calibration dan ML training

## Adapter Machine Learning
Implementasi yang disarankan:

- tambah wrapper inference khusus sticker di backend service layer

Usulan file baru:

- `backend/app/services/sticker_inference.py`

Tanggung jawab modul:

- load model
- cache model instance
- preprocess ROI
- jalankan inferensi
- ubah output mentah menjadi format standar internal

Kontrak output internal:

```json
[
  {
    "label": "sticker_type_a",
    "confidence": 0.91,
    "class_confidence": 0.91,
    "position": {
      "x1": 12.0,
      "y1": 18.0,
      "x2": 96.0,
      "y2": 82.0
    }
  }
]
```

Keuntungan:

- `inspection_session.py` tidak langsung terikat ke framework ML tertentu
- penggantian YOLO ke engine lain menjadi lebih murah

## State Machine Target
State target:

- `idle`
- `part_missing`
- `part_ready`
- `sticker_checking`
- `decision_pending`
- `decision_committed`
- `cooldown`

Aturan:

- `part_missing`: part-ready belum lolos
- `part_ready`: warna part sudah cocok
- `sticker_checking`: inference ML sedang berjalan atau kandidat sedang ditimbang
- `decision_pending`: kandidat hasil ada, menunggu stabilisasi
- `decision_committed`: hasil final sudah dihitung sekali
- `cooldown`: mencegah double count untuk part yang sama

## Rencana Migrasi Bertahap

### Phase 0 - Dokumentasi dan Schema Freeze
Output:

- finalisasi dokumen ini
- finalisasi schema template dua ROI
- finalisasi payload response runtime

### Phase 1 - Shared Contract Refactor
Output:

- `templates.py` support dua ROI
- `inspection.py` support evidence dua tahap
- fallback migrasi template lama

### Phase 2 - Color Reference Gate
Output:

- implement evaluator `part_ready_roi`
- kalibrasi profile warna terhubung ke template
- session runtime bisa menahan validasi sticker saat part belum ready

### Phase 3 - ML Sticker Adapter
Output:

- modul inference sticker terpisah
- template bisa menunjuk model ML aktif
- runtime bisa memanggil model nyata

### Phase 4 - Sticker Validator Rules
Output:

- reject reason deterministik
- posisi, confidence, dan class tervalidasi

### Phase 5 - Persistence dan Dashboard
Output:

- hasil dua tahap masuk repository
- dashboard membaca field baru

### Phase 6 - Operator UX
Output:

- dua ROI terlihat
- dua tahap status terbaca jelas
- template dan settings mengikuti schema baru

### Phase 7 - Admin dan Engineer UX
Output:

- template editor dua ROI
- calibration flow yang terpisah dari training
- model registry terhubung ke template

### Phase 8 - Hardening
Output:

- test end-to-end
- regression test
- migration notes SQL Server

## Test Plan Rinci

### Unit Test

#### Color Matching
- profile valid menghasilkan `ready = true`
- warna mismatch menghasilkan `ready = false`
- ROI kosong menghasilkan error yang terkontrol

#### Sticker Validator
- tanpa detection menghasilkan `NOT_FOUND`
- class tidak cocok menghasilkan `WRONG_TYPE`
- confidence rendah menghasilkan `LOW_CLASS_CONF`
- posisi melenceng menghasilkan `OUT_OF_POSITION`

### Integration Test
- part tidak ready membuat ML sticker final tidak di-commit
- part ready lalu sticker benar menghasilkan `ACCEPT`
- satu part diam lama di ROI tidak menambah counter berulang
- part keluar lalu part baru masuk menambah counter baru

### UI Test Manual
- operator bisa membedakan `part ready` vs `final result`
- template selector mengganti template aktif tanpa merusak session berikutnya
- settings dialog dapat mengubah dua ROI
- layout tetap usable di resolusi layar client yang lebih kecil

## Risiko Teknis

- lighting berubah drastis sehingga color reference menghasilkan false negative
- ROI part-ready salah set sehingga ML sticker tidak pernah jalan
- model sticker tidak sinkron dengan `expected_class` template
- schema SQL Server tertinggal dari schema JSON
- payload runtime membesar terlalu banyak jika menyertakan debug data berlebihan

## Keputusan Scope
Masuk scope:

- dua ROI
- color reference untuk part-ready
- machine learning untuk sticker
- validator rule sebelum DB write
- counter berbasis committed event

Di luar scope tahap ini:

- ergonomic check
- multi-camera session dalam satu operator screen
- visual flow editor generik
- OCR/barcode validator sebagai jalur utama

## Deliverable Akhir
Saat rencana ini selesai diimplementasikan, sistem harus memiliki sifat berikut:

- part dicek lebih dulu menggunakan ROI dan referensi warna yang terpisah
- sticker divalidasi menggunakan machine learning pada ROI lain
- keputusan final baru masuk DB setelah dua tahap lolos logika runtime
- UI operator menampilkan dua tahap inspeksi secara jelas
- admin dapat mengaudit result dengan evidence yang cukup
- engineer memiliki workflow terpisah untuk calibration dan ML sticker

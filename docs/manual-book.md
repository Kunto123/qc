# Manual Book

## Tujuan Dokumen

Dokumen ini adalah manual penggunaan untuk seluruh fitur yang saat ini tersedia di aplikasi `qc-suite-python`.
Manual ini mengikuti implementasi yang ada sekarang, bukan roadmap. Jika ada fitur yang masih berupa mock atau belum punya UI lengkap, hal itu disebutkan secara eksplisit.

## Isi Sistem

Sistem terdiri dari dua runtime:

- `backend`: Flask API yang menangani autentikasi, template, deployment, session inspeksi, hasil inspeksi, dashboard, dataset, training, model registry, dan color calibration.
- `client_tk`: aplikasi desktop Tkinter yang dipakai user sesuai role.

Pipeline inspeksi yang dijalankan saat role `operator` aktif:

- `camera input`
- `ROI crop`
- `main vision`
- `part-ready validation`
- `sticker validation`
- `DB write`
- `display`

## Role dan Akun Default

Akun default untuk environment development:

- `admin / admin123`
- `operator / operator123`
- `engineer / engineer123`

Role aktif menentukan screen yang dibuka setelah login:

- `admin` masuk ke layar Admin.
- `operator` masuk ke layar Operator.
- `engineer` masuk ke layar Engineer.

## Persiapan Menjalankan Sistem

1. Jalankan backend:

```powershell
cd qc-suite-python
py -3.11 scripts/run_backend.py
```

2. Jalankan client:

```powershell
cd qc-suite-python
py -3.11 scripts/run_client.py
```

3. Pada form login, isi:

- `Server URL`: default `http://127.0.0.1:8100`
- `Username`
- `Password`

4. Klik `Login`.

## Struktur Data Penyimpanan

Secara default sistem menyimpan data lokal di folder `qc-suite-python/data`:

- `data/json_store`: user, template, deployment, profile, dataset metadata, model metadata, training jobs, inspection results
- `data/datasets`: file dataset yang diunggah dari layar Engineer
- `data/models`: folder target untuk aset model jika nanti dipakai

Jika environment `MSSQL_*` diisi dengan benar, hasil inspeksi dapat ditulis ke SQL Server untuk repository inspection results.

## Manual Role Operator

### Tujuan Layar Operator

Layar Operator dipakai untuk menjalankan inspeksi produksi secara live dari kamera client ke backend.

### Komponen Layar

Bagian header dan toolbar:

- `Settings`: membuka parameter operator, line/station, camera, template version, dan dua ROI
- `Load Deployment`
- `Start Camera`
- `Stop Camera`
- `Start Session`
- `Stop Session`
- `Template` selector di kanan atas

Bagian context summary menampilkan:

- `Line`
- `Station`
- `Template`
- `Operator`

Bagian `Settings` memisahkan dua ROI:

- `Part Ready ROI`
  - `x (left)`
  - `y (top)`
  - `w (width)`
  - `h (height)`
- `Sticker ROI`
  - `x (left)`
  - `y (top)`
  - `w (width)`
  - `h (height)`
- `Apply ROI`

Bagian display:

- `Part Ready ROI`: panel kiri yang menampilkan crop ROI part-ready
- `Sticker ROI / ML Overlay`: panel kanan yang menampilkan frame penuh
  - sebelum session dimulai: frame kamera client dengan bbox `Sticker ROI`
  - saat session berjalan dan backend sudah membalas: overlay hasil machine learning dari server
  - fallback: jika overlay belum ada, panel kanan tetap menampilkan frame penuh dengan bbox `Sticker ROI`
- `Inspection Status`: panel kanan untuk membaca hasil gate part-ready, hasil sticker validation, dan detail commit
- `Session Counter`: counter `total / accept / reject` dan breakdown reject
- `Recent Events`: daftar event committed terakhir

### Makna Field

- `Line`: ID line produksi
- `Station`: ID station produksi
- `Camera`: index kamera lokal pada client, misalnya `0`
- `Template Ver`: ID versi template yang dipakai session
- `Part Ready ROI x/y/w/h`: ROI untuk mendeteksi apakah part sudah hadir berdasarkan referensi warna
  - `x`: jarak dari kiri frame
  - `y`: jarak dari atas frame
  - `w`: lebar area ROI
  - `h`: tinggi area ROI
- `Sticker ROI x/y/w/h`: ROI untuk inferensi model sticker
  - `x`: jarak dari kiri frame
  - `y`: jarak dari atas frame
  - `w`: lebar area ROI
  - `h`: tinggi area ROI
- seluruh ROI ter-normalisasi terhadap ukuran frame, biasanya antara `0.0` sampai `1.0`

Contoh `Sticker ROI`:

- `x = 0.2`
- `y = 0.2`
- `w = 0.6`
- `h = 0.6`

Ini berarti area inspeksi dimulai dari 20% kiri, 20% atas, dengan lebar 60% dan tinggi 60% dari frame.

Contoh `Part Ready ROI`:

- `x = 0.2`
- `y = 0.2`
- `w = 0.25`
- `h = 0.25`

ROI ini biasanya lebih kecil karena hanya dipakai untuk gate warna part-ready.

### Alur Operasional Standar

1. Login sebagai `operator`.
2. Isi `Line` dan `Station`.
3. Klik `Load Deployment`.
4. Pastikan `Template Ver` terisi otomatis dari deployment aktif.
5. Jika perlu override ROI, buka `Settings`.
6. Periksa `Part Ready ROI` dan `Sticker ROI` pada preview crop.
7. Isi `Camera` dengan index kamera lokal.
8. Klik `Start Camera`.
9. Tunggu sampai panel `Inspection Camera` menampilkan feed.
10. Klik `Start Session`.
11. Amati panel kanan:
   - `Part Ready Gate`
   - `Sticker Validation`
   - `Commit Details`
12. Jika ROI perlu diubah, buka `Settings`, atur dua ROI, lalu klik `Apply ROI`.
13. Saat selesai, klik `Stop Session`.
14. Klik `Stop Camera` bila kamera tidak dipakai lagi.

### Kondisi Wajib Sebelum Start Session

`Start Session` akan gagal secara operasional bila:

- kamera belum dijalankan
- frame pertama belum terbaca
- `Template Ver` kosong atau `0`

### Arti Hasil Inspeksi

Status utama yang tampil:

- `ACCEPT`: hasil lulus
- `REJECT`: hasil gagal

Panel `Inspection Status` dibagi menjadi tiga bagian:

- `Live Stage`: state event runtime saat frame sedang diproses
- `Part Ready Gate`: hasil gate warna pada `part_ready_roi`
- `Sticker Validation`: hasil inferensi model sticker pada `sticker_roi`

Reject reason yang mungkin muncul:

- `NOT_FOUND`
- `WRONG_TYPE`
- `LOW_ROI_CONF`
- `LOW_CLASS_CONF`
- `OUT_OF_POSITION`
- `PART_NOT_READY`
- `ERROR`

### Makna Panel Hasil

- `Decision`: keputusan akhir inspeksi
- `Reason`: kode reject jika ada
- `Part Ready`: hasil validator warna/part-ready
- `Part`: nama part dari template
- `Line`: line dari konfigurasi sticker
- `DB Write`: status penulisan hasil ke repository

### Catatan Operasional

- `Load Deployment` hanya membaca deployment aktif untuk kombinasi `Line + Station`.
- Operator secara teknis masih bisa mengetik `Template Ver` manual, tetapi alur yang direkomendasikan tetap memakai deployment.
- `Server Overlay` menampilkan ROI yang sudah dipotong dan di-overlay oleh hasil keputusan backend.

### Pesan Error yang Umum

- `Line dan Station wajib diisi.`
  - Isi field `Line` dan `Station` sebelum `Load Deployment`.
- `Tidak ada deployment aktif.`
  - Admin belum membuat deployment untuk line/station tersebut.
- `Cannot open camera index X`
  - Index kamera salah atau kamera sedang dipakai aplikasi lain.
- `Start camera dan tunggu frame pertama dulu.`
  - Kamera baru dijalankan tetapi frame belum tersedia.
- `Template version wajib diisi.`
  - Operator belum memilih deployment atau belum mengisi ID versi template.
- `Upload error: ...`
  - Backend tidak bisa dijangkau atau request frame gagal.

## Manual Role Admin

### Tujuan Layar Admin

Layar Admin dipakai untuk mengelola konfigurasi inspeksi dan melihat hasil runtime.

Tab yang tersedia:

- `Templates`
- `Deployments`
- `Users`
- `Results`
- `Dashboard`

## Templates

### Fungsi

Tab ini dipakai untuk:

- melihat daftar template
- memuat template aktif
- membuat template baru
- mengubah template
- menghapus template

### Komponen

Panel kiri:

- daftar template dengan format `id | vX | name`
- `Refresh`
- `New`
- `Load`
- `Delete Selected`

Panel kanan:

- `Structured Form`
  - `Template Identity`
  - `Camera`
  - `Part Ready`
  - `Sticker`
  - `Vision`
  - `Persistence`
  - `Metadata`
- tab `Raw JSON`
- tombol:
  - `Preview Raw JSON`
  - `Load Form From Raw JSON`
- `Save Template`

### Alur Membuat Template Baru

1. Login sebagai `admin`.
2. Buka tab `Templates`.
3. Klik `New`.
4. Isi field pada `Structured Form`.
5. Pilih model dan color profile bila sudah tersedia.
6. Klik `Save Template`.

### Alur Mengubah Template Existing

1. Pilih item pada daftar template.
2. Klik `Load Selected`.
3. Edit field pada `Structured Form` atau `Raw JSON`.
4. Klik `Save Template`.

Perilaku penting:

- Setiap `update` akan membuat `version` baru secara otomatis.
- Versi terbaru akan menjadi `current_version_id`.

### Struktur JSON Template

Field inti yang sekarang dipisah jelas di UI:

- `name`
- `description`
- `is_active`
- `camera`
- `part_ready_roi`
- `sticker_roi`
- `vision`
- `part_ready`
- `sticker`
- `persistence`
- `metadata`

Objek JSON minimum yang direkomendasikan:

```json
{
  "name": "QC Line A",
  "description": "Template inspeksi default",
  "is_active": true,
  "camera": {
    "camera_index": 0,
    "width": 640,
    "height": 480,
    "fps": 15
  },
  "part_ready_roi": {
    "x": 0.2,
    "y": 0.2,
    "w": 0.25,
    "h": 0.25
  },
  "sticker_roi": {
    "x": 0.2,
    "y": 0.2,
    "w": 0.6,
    "h": 0.6
  },
  "vision": {
    "model_path": "models/dummy.pt",
    "model_meta_path": null,
    "runtime": "ultralytics",
    "conf_threshold": 0.25,
    "stream_fps": 10,
    "inference_fps": 4,
    "imgsz": 640,
    "classes": ["sample-sticker"],
    "enable_ergonomic_check": false,
    "ergonomic_pose_model_path": null,
    "ergonomic_min_keypoint_conf": 0.35
  },
  "part_ready": {
    "enabled": true,
    "color_profile_id": null,
    "colorspace": "LAB",
    "distance_threshold": null,
    "min_match_ratio": 0.75
  },
  "sticker": {
    "part_name": "Sample Part",
    "expected_class": "sample-sticker",
    "line": "LINE-A",
    "enabled": true,
    "validator_mode": "ml_detection",
    "min_roi_confidence": 0.0,
    "min_class_confidence": null,
    "max_offset_x": 80,
    "max_offset_y": 80
  },
  "persistence": {
    "write_to_db": true
  },
  "metadata": {}
}
```

### Panduan Mengubah Parameter Template

- `camera.camera_index`
  - default kamera yang disarankan untuk station tersebut
- `part_ready_roi`
  - area crop untuk gate warna part-ready
- `sticker_roi`
  - area crop untuk inferensi model sticker
- `vision.model_path`
  - path model yang akan dipakai oleh backend
- `vision.model_meta_path`
  - path metadata model
- `vision.classes`
  - daftar class target
- `part_ready.color_profile_id`
  - mengaktifkan validasi part-ready berbasis color profile
- `part_ready.min_match_ratio`
  - ambang kecocokan minimal
- `sticker.expected_class`
  - class yang wajib ditemukan
- `sticker.max_offset_x / max_offset_y`
  - batas toleransi posisi
- `persistence.write_to_db`
  - jika `false`, hasil inspeksi tidak ditulis ke repository hasil

### Best Practice

- Mulai dari template existing, lalu edit seperlunya.
- Jangan hapus field yang diwajibkan contract.
- Pastikan `sticker.part_name`, `sticker.expected_class`, dan `sticker.line` selalu terisi.

## Deployments

### Fungsi

Tab ini dipakai untuk mengikat template ke kombinasi line dan station.

### Komponen

- list deployment dengan format `id | line/station | version | template_name`
- selector `Template`
- field:
  - `Template ID`
  - `Version ID`
  - `Line`
  - `Station`
- tombol:
  - `Deploy`
  - `Refresh`
  - `Deactivate Selected`

### Alur Deployment

1. Pastikan template sudah ada.
2. Pilih template di selector atau isi `Template ID` dan `Version ID`.
3. Isi `Line` dan `Station`.
4. Klik `Deploy`.
5. Klik `Refresh` untuk memastikan record baru muncul.

### Efek Deployment

Deployment aktif akan dibaca oleh layar Operator saat tombol `Load Deployment` ditekan.

## Users

### Fungsi

Tab ini dipakai untuk melihat daftar user dan membuat user baru.

### Komponen

- list user dengan format `id | username | role | active=...`
- field:
  - `Username`
  - `Password`
  - `Role`
- tombol:
  - `Create`
  - `Refresh`
  - `Enable Selected`
  - `Disable Selected`

### Alur Membuat User

1. Isi `Username`.
2. Isi `Password`.
3. Pilih `Role`.
4. Klik `Create`.
5. Klik `Refresh` bila perlu.

## Results

### Fungsi

Tab ini dipakai untuk membaca hasil inspeksi yang sudah tersimpan.

### Komponen

- filter:
  - `Line`
  - `Station`
  - `Part`
  - `Template Ver`
  - `Decision`
- list hasil dengan format `id | decision | part_name | line/station | reason`
- panel `Result Summary`
- panel `Raw Result Payload`

### Alur Membuka Hasil

1. Isi filter bila perlu.
2. Klik `Refresh`.
3. Pilih salah satu hasil.
4. Klik `Open` atau double click.
5. Summary evidence dan raw payload akan tampil di panel kanan.

### Isi Umum Detail Hasil

- `template_version_id`
- `line_id`
- `part_name`
- `decision`
- `decision_code`
- `reject_reason_code`
- `targets`
- `inspected_at`
- `push_status`
- `part_ready_status`
- `part_ready_match_ratio`
- `detected_class`
- `expected_class`
- `sticker_backend`

## Dashboard

### Fungsi

Tab ini dipakai untuk melihat ringkasan agregat hasil inspeksi.

### Komponen

- filter:
  - `Line`
  - `Station`
  - `Part`
  - `Template Ver`
  - `Granularity`
- KPI cards:
  - `Total`
  - `Accept`
  - `Reject`
  - `Part Ready`
  - `Avg Sticker Conf`
  - `ML Backend`
- list `Bucket Trend`
- panel `Dashboard Raw`

### Isi Summary

- `total_inspections`
- `total_accept`
- `total_reject`
- total reject per kategori

### Isi Buckets

Bucket berisi agregasi waktu dengan granularity default dari backend.

### Kegunaan

- memantau total hasil inspeksi
- melihat distribusi reject
- memeriksa trend sederhana berdasarkan bucket waktu

## Manual Role Engineer

### Tujuan Layar Engineer

Layar Engineer dipakai untuk workflow data, model, dan calibration.

Tab yang tersedia:

- `Data`
- `Training`
- `Models`
- `Calibration`

## Data

### Fungsi

Mengelola dataset, upload file, browser file, dan annotation workflow dalam satu area kerja.

### Komponen

- panel `Datasets`
- panel `Upload and Browse Files`
- panel `Annotation Workflow`

Pilihan `Target`:

- `images`
- `labels`
- `exports`

### Alur Data Workflow

1. Pastikan dataset sudah dibuat.
2. Pilih dataset pada list kiri.
3. Upload file ke target `images`, `labels`, atau `exports`.
4. Periksa file pada `Dataset Files`.
5. Pilih image pada panel `Images`.
6. Load dan save label JSON di panel annotation.

### Kegagalan Umum

- `Choose file dulu.`
  - Belum memilih file lokal.
- `Dataset ID wajib diisi.`
  - Field dataset kosong.

## Training

### Fungsi

Memisahkan pekerjaan `augment` dan `training job` sticker detector.

### Komponen

- panel `Augment Jobs`
- panel `Training Jobs`
- summary job dan raw detail training

### Alur Start Training

1. Isi `Dataset ID`.
2. Isi `Base Model`.
3. Klik `Start Training`.
4. Klik `Refresh` untuk melihat status job.

### Alur Cancel Training

1. Pilih training job pada list.
2. Klik `Cancel Selected`.

### Catatan

- Implementasi training job saat ini masih mock workflow.
- `Base Model` default diisi `baseline`.

## Models

### Fungsi

Mendaftarkan metadata model yang akan dipakai oleh sistem.

### Komponen

- list models
- field:
  - `Name`
  - `Path`
  - `Meta Path`
  - `Runtime`
  - `Task`
  - `Classes CSV`
  - `Architecture Family`
  - `Architecture Variant`
- tombol:
  - `Register Model`
  - `Refresh`

### Alur Menambah Model

1. Isi `Name`.
2. Isi `Path`.
3. Lengkapi `Meta Path`, runtime, task, dan daftar class bila tersedia.
4. Klik `Register Model`.

### Catatan

- Tab ini saat ini mendaftarkan metadata model, bukan upload binary model.
- Pastikan path yang diisi sesuai struktur file backend jika model nyata akan dipakai.

## Calibration

### Fungsi

Menghitung dan menyimpan color profile yang dipakai oleh validator `part_ready`.

### Komponen

- `Choose Image`
- `Compute`
- `Profile Name`
- `Save Profile`
- optional `ROI x/y/w/h`
- panel `ROI Preview`
  - `Source Image` menampilkan image asli dengan kotak ROI
  - `ROI Crop` menampilkan hasil crop ROI yang sama dengan request backend
- panel `Computed Profile`
- list `Saved Profiles`
- panel `Profile Detail`

### Alur Kalibrasi

1. Klik `Choose Image`.
2. Pilih file image lokal.
3. Isi `ROI x/y/w/h` bila ingin hitung profile dari area tertentu saja.
4. Periksa panel `ROI Preview`.
5. Pastikan kotak pada `Source Image` dan hasil `ROI Crop` sudah sesuai posisi yang diinginkan.
6. Klik `Compute`.
7. Periksa hasil JSON di `Computed Profile`.
8. Isi `Profile Name`.
9. Klik `Save Profile`.

### Hasil yang Dihasilkan

Profile berisi:

- `colorspace`
- `reference_color`
- `reference_stats`
- `tolerance`
- `min_match_ratio`

### Kapan Dipakai

Nilai `profile_id` hasil save dapat dipasang ke field `part_ready.color_profile_id` pada template.

### Kegagalan Umum

- `Pilih image dulu.`
  - Belum memilih image sebelum compute.
- `Compute profile dulu.`
  - Belum menghitung profile sebelum save.
- preview crop kosong
  - nilai ROI belum lengkap, di luar rentang `0..1`, atau menghasilkan area crop kosong.

## Manual Workflow End-to-End

### Workflow 1: Menyiapkan Inspeksi Baru dari Nol

1. Login sebagai `engineer`.
2. Buat dataset baru.
3. Upload sample image ke dataset.
4. Simpan annotation JSON jika dibutuhkan.
5. Register model pada tab `Models`.
6. Hitung dan simpan color profile pada tab `Color Calibrate`.
7. Login sebagai `admin`.
8. Buat template baru atau load template existing.
9. Isi parameter `sticker`, `vision`, dan `part_ready`.
10. Simpan template.
11. Buat deployment untuk `line/station`.
12. Login sebagai `operator`.
13. Load deployment, start camera, lalu start session.

### Workflow 2: Ganti Threshold atau ROI untuk Produk Existing

1. Login sebagai `admin`.
2. Tab `Templates`.
3. Pilih template existing, lalu `Load Selected`.
4. Ubah nilai threshold atau konfigurasi ROI di JSON.
5. `Save Template`.
6. Catat `Version ID` terbaru.
7. Tab `Deployments`.
8. Deploy ulang line/station ke versi baru.
9. Operator login ulang atau reload deployment.

### Workflow 3: Investigasi Reject di Produksi

1. Login sebagai `admin`.
2. Tab `Results`, klik `Refresh`.
3. Pilih hasil reject dan klik `Open`.
4. Periksa:
   - `reject_reason_code`
   - `targets`
   - `decision_code`
   - `template_version_id`
5. Tab `Dashboard` untuk melihat apakah reject tersebut bersifat sporadis atau tren.
6. Bila perlu, koordinasikan dengan `engineer` untuk ubah template, profile, atau model.

## Troubleshooting

## Login Gagal

Penyebab yang mungkin:

- username/password salah
- backend belum berjalan
- `Server URL` salah

Tindakan:

- verifikasi backend aktif
- gunakan akun seed dev
- cek `http://127.0.0.1:8100/health`

## Operator Tidak Bisa Load Deployment

Penyebab yang mungkin:

- admin belum deploy template
- salah input `Line` atau `Station`

Tindakan:

- cek tab `Deployments` pada role admin
- pastikan line/station identik

## Kamera Tidak Muncul

Penyebab yang mungkin:

- index kamera salah
- kamera dipakai aplikasi lain
- permission kamera ditolak OS

Tindakan:

- coba index `0`, lalu `1`
- tutup aplikasi lain yang memakai kamera

## Hasil Selalu REJECT

Penyebab yang mungkin:

- ROI salah
- `expected_class` tidak sesuai
- `max_offset_x/y` terlalu ketat
- profile warna tidak cocok

Tindakan:

- cek ROI operator
- cek JSON template
- cek profile warna pada Engineer

## Save Template Error

Penyebab yang mungkin:

- JSON tidak valid
- field wajib template hilang
- nilai numerik salah format

Tindakan:

- validasi JSON
- mulai dari template sample
- jangan hapus objek `sticker`

## Keterbatasan MVP Saat Ini

- main vision default masih dummy contour engine, belum YOLO nyata
- annotation masih editor JSON, belum visual annotation tool
- augment job masih mock
- training job masih mock
- models tab baru menyimpan metadata model, belum upload file model
- users tab belum punya tombol activate/deactivate
- deployments tab belum punya tombol deactivate di UI
- export hasil inspeksi belum tersedia di UI

## Dokumen Pendukung

Dokumen terkait yang juga tersedia di folder `docs`:

- `screen-map.md`
- `api-contract.md`
- `deployment-guide.md`
- `migration-notes.md`

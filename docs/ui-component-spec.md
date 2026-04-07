# UI Component Spec

## Tujuan Dokumen

Dokumen ini mendefinisikan komponen UI utama dan behavior counter `accept/reject`.
Dokumen ini dipakai sebagai acuan sebelum implementasi kode.

## Prinsip Komponen

- satu komponen = satu tanggung jawab visual
- state penting dibedakan dengan warna, ukuran, dan posisi
- komponen operator dioptimalkan untuk pembacaan cepat
- dashboard/admin dioptimalkan untuk filter dan review

## Design Tokens yang Disarankan

### Typography

- `Heading XL`: status keputusan besar
- `Heading L`: judul panel
- `Body M`: isi utama
- `Body S`: metadata dan helper text
- `Mono S`: ID, version, payload singkat bila perlu

### Color Semantics

- `Success`: hijau untuk `ACCEPT`
- `Danger`: merah untuk `REJECT`
- `Warning`: amber untuk kondisi transisi atau masalah non-fatal
- `Info`: biru/abu untuk koneksi, session, status sistem
- `Neutral`: latar panel, divider, border

### Spacing

- `4`: micro spacing
- `8`: antar label dan field
- `12`: antar field dalam satu grup
- `16`: antar panel kecil
- `24`: antar section besar

### Border dan Panel

- panel utama harus punya border halus atau background kontras
- panel status kritikal tidak boleh menyatu dengan background root

## Komponen Global

### 1. App Header

Data:

- app name
- username
- role
- server url atau environment
- tombol logout

### 2. Status Badge

Pemakaian:

- server
- camera
- session
- DB
- part ready
- push status

State yang disarankan:

- `ONLINE`
- `OFFLINE`
- `READY`
- `RUNNING`
- `STOPPED`
- `WARNING`
- `ERROR`

## Komponen Operator

### 1. Control Button Group

Isi:

- `Load Deployment`
- `Start Camera`
- `Stop Camera`
- `Start Session`
- `Stop Session`
- `Apply ROI`

Hierarki tombol:

- primary: `Start Session`
- secondary: `Load Deployment`, `Start Camera`, `Apply ROI`
- danger: `Stop Session`, `Stop Camera`

### 2. Runtime Context Panel

Field:

- line
- station
- camera index
- template version
- operator name

### 3. ROI Editor

Field:

- `x`
- `y`
- `w`
- `h`

Aturan:

- tampil sebagai panel kecil atau collapsible
- validasi range numerik
- apply manual, bukan auto-commit tiap ketik

### 4. Live View Card

Varian:

- `Client Camera`
- `Server Overlay`

State:

- no signal
- loading
- live
- error

### 5. Decision Card

Field:

- decision besar: `ACCEPT` atau `REJECT`
- reject reason
- part ready
- part name
- line
- DB write status
- inspected time atau last commit time

State:

- `ACCEPT`
- `REJECT`
- `WAITING`
- `ERROR`

Aturan penting:

- jangan update card ini dengan noise per frame bila event belum committed
- jika ada `live detection state`, tampilkan di area berbeda dari `last committed result`

### 6. Counter Card

Varian:

- `Session Counter`
- `Station Counter`
- `Today Counter`

Minimum field:

- `Total`
- `Accept`
- `Reject`

### 7. Reject Breakdown Card

Field:

- `NOT_FOUND`
- `WRONG_TYPE`
- `OUT_OF_POSITION`
- `LOW_CLASS_CONF`
- `LOW_ROI_CONF`
- `PART_NOT_READY`
- `OTHER`

### 8. Recent Events List

Field:

- time
- decision
- reason
- optional part

## Komponen Admin

### 1. Filter Bar

Pemakaian:

- deployments
- results
- dashboard

### 2. Master List

Entity:

- template
- deployment
- user
- result
- model
- dataset

### 3. Detail Panel

Aturan:

- ringkasan di bagian atas
- editor atau viewer di bagian bawah
- untuk payload besar, sediakan tab `Overview` dan `Raw JSON`

### 4. KPI Card

Contoh:

- `Total Inspections`
- `Accept`
- `Reject`
- `Reject Rate`

## Komponen Engineer

### 1. Upload Panel

Field:

- dataset id
- target
- selected file
- upload action

### 2. Job Table

Field:

- id
- dataset
- status
- started at
- finished at
- error optional

Status:

- queued
- running
- completed
- cancelled
- failed

### 3. JSON Editor Advanced

Pemakaian:

- template advanced editing
- annotation
- calibration profile detail
- result raw payload

Aturan:

- tampil sebagai advanced panel, bukan first-class view untuk operator

## Counter Accept/Reject Spec

### Tujuan

Menjamin counter akurat per unit inspeksi, bukan per frame kamera.

### Definisi Counter

#### 1. Session Counter

Reset saat session baru dimulai.

Field:

- `session_total`
- `session_accept`
- `session_reject`
- `session_reject_breakdown`

#### 2. Station Counter

Berlaku untuk kombinasi `line_id + station_id`.

Field:

- `station_total`
- `station_accept`
- `station_reject`
- `station_reject_breakdown`

#### 3. Today or Shift Counter

Agregasi lintas session berdasarkan data persisted.

Field:

- `today_total`
- `today_accept`
- `today_reject`
- `today_reject_breakdown`

### Rule Increment

Counter hanya boleh bertambah jika seluruh kondisi berikut terpenuhi:

- ada inspection event yang valid
- decision sudah final
- event belum pernah dihitung sebelumnya
- event di-commit oleh backend

Counter tidak boleh bertambah jika:

- masih frame yang sama dari part yang sama
- hasil masih provisional
- object linger di ROI
- DB write gagal dan sistem memilih hanya menghitung persisted result

### Event Model yang Disarankan

State machine:

- `idle`
- `part_detected`
- `part_ready`
- `decision_pending`
- `decision_committed`
- `cooldown`

Penjelasan:

- `idle`: belum ada candidate event
- `part_detected`: object mulai terdeteksi
- `part_ready`: validator part-ready lulus
- `decision_pending`: hasil mulai stabil tetapi belum final
- `decision_committed`: hasil dihitung 1 kali
- `cooldown`: jeda agar part yang sama tidak dihitung lagi

### Last Live vs Last Committed

UI harus membedakan dua konsep:

- `live result`
  - status frame saat ini
- `last committed result`
  - event final terakhir yang sah untuk counter

Konsekuensi UI:

- Decision card utama sebaiknya menampilkan `last committed result`
- live inference state bisa tampil sebagai badge kecil atau substatus

### Counter Data Contract yang Disarankan

```json
{
  "validation": {
    "decision": "ACCEPT",
    "reject_reason_code": null
  },
  "event_state": "decision_committed",
  "event_id": "evt-00128",
  "count_committed": true,
  "count_source": "session",
  "counters": {
    "session_total": 128,
    "session_accept": 121,
    "session_reject": 7,
    "session_reject_breakdown": {
      "NOT_FOUND": 2,
      "WRONG_TYPE": 3,
      "OUT_OF_POSITION": 2
    },
    "station_total": 9842,
    "station_accept": 9701,
    "station_reject": 141
  }
}
```

### Counter UI Behavior

Jika `count_committed = true`:

- update `Total`
- update `Accept` atau `Reject`
- update `Reject Breakdown`
- tambahkan `Recent Events`

Jika `count_committed = false`:

- jangan ubah angka counter
- optional tampilkan status kecil `INSPECTION IN PROGRESS`

### Reset Rules

#### Session Counter

- reset saat `Start Session`
- reset juga jika user explicit `Reset Session Counter` bila nanti ada fitur itu

#### Station Counter

- tidak reset saat UI close
- tidak reset saat session restart
- sumbernya ideal dari backend/persisted storage

#### Today or Shift Counter

- reset secara natural oleh filter waktu, bukan oleh tombol operator biasa

### Data Integrity Rules

- `1 physical part = 1 committed count`
- reconnect client tidak boleh menduplikasi event
- restart camera tidak boleh menduplikasi event yang belum selesai
- dashboard harus konsisten dengan persisted result jika itu source of truth

### Error Handling

Jika DB write gagal, pilih salah satu strategi dan konsisten:

- strategi A: counter hanya naik jika persisted sukses
- strategi B: counter session naik walau persist gagal, tetapi tampil badge `UNSAVED`

Untuk produksi, strategi A lebih aman jika dashboard harus identik dengan DB.
Untuk usability operator, strategi B lebih informatif tetapi harus dibedakan jelas dari counter persisted.

## Implementasi Bertahap yang Disarankan

1. bangun `Status Badge`
2. bangun `Decision Card`
3. bangun `Counter Card`
4. bangun `Reject Breakdown Card`
5. pisahkan `live result` dan `last committed result`
6. implementasikan event dedup dan counter commit di backend
7. hubungkan dashboard ke source counter yang sama

## Acceptance Criteria Desain

- operator dapat melihat accept/reject total tanpa membaca JSON
- operator dapat membedakan hasil live dan hasil committed
- admin dapat membaca summary tanpa membuka raw payload
- engineer tetap punya akses data teknis tanpa merusak UX operator

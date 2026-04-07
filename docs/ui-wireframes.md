# UI Wireframes

## Tujuan Dokumen

Dokumen ini mendefinisikan wireframe dan hirarki layout untuk redesign `qc-suite-python` tanpa mengubah domain flow yang sudah ada.
Fokus utama:

- membuat operator screen lebih cepat dibaca
- memisahkan area status, keputusan, dan live view
- membuat admin screen lebih dekat ke pola master-detail
- membuat engineer screen lebih dekat ke workflow daripada CRUD mentah

## Prinsip Layout Global

- target desktop utama: `1440 x 900`
- minimum operasional: `1280 x 800`
- shell global terdiri dari:
  - top app header
  - role workspace
  - status/footer strip opsional
- spacing dasar: `8 / 12 / 16 / 24`
- tidak ada panel penting yang hanya dibedakan oleh teks kecil
- status kritikal harus terlihat dari jarak pandang operator

## Shell Global

```text
+------------------------------------------------------------------------------------------------------------------+
| QC Suite Python | User: <username> (<role>) | Server: <url> | Connection: ONLINE/OFFLINE | [Logout]            |
+------------------------------------------------------------------------------------------------------------------+
|                                                                                                                  |
|  Role Workspace                                                                                                  |
|                                                                                                                  |
+------------------------------------------------------------------------------------------------------------------+
| Footer Opsional: last sync | message | warning                                                                   |
+------------------------------------------------------------------------------------------------------------------+
```

## Login Screen

### Tujuan

- cepat
- jelas
- tidak terasa seperti form debug

### Wireframe

```text
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|                                        QC Suite Python                                           |
|                              Vision Inspection Desktop Client                                    |
|                                                                                                  |
|                    +----------------------------------------------------------+                  |
|                    | Login                                                    |                  |
|                    |----------------------------------------------------------|                  |
|                    | Server URL   [ http://127.0.0.1:8100                 ]   |                  |
|                    | Username     [ operator                              ]   |                  |
|                    | Password     [ ********                              ]   |                  |
|                    |                                                          |                  |
|                    |                                      [ Login ]           |                  |
|                    +----------------------------------------------------------+                  |
|                                                                                                  |
|                       Hint: admin / operator / engineer accounts for dev                         |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

## Operator Screen

### Tujuan

- operator langsung tahu sistem sedang jalan atau tidak
- keputusan terakhir harus terbaca dalam < 1 detik
- counter `accept/reject/total` harus dominan
- kontrol ROI tidak mengambil fokus utama

### Struktur Layout yang Disarankan

```text
+------------------------------------------------------------------------------------------------------------------+
| QC Suite Python | Operator: <username> | Line: LINE-A | Station: ST-01 | Template: QC Line A v3 | [Logout]     |
+------------------------------------------------------------------------------------------------------------------+
| Server: ONLINE | Camera: READY | Session: RUNNING | DB: CONNECTED | Last Commit: ACCEPT | Shift: A            |
+------------------------------------------------------------------------------------------------------------------+
| Controls / Setup                        | Live View Area                                      | Decision / Count  |
|-----------------------------------------|-----------------------------------------------------|-------------------|
| [Load Deployment] [Start Camera]        | +-----------------------------------------------+   | +---------------+ |
| [Stop Camera] [Start Session]           | | Client Camera                                 |   | | Last Decision | |
| [Stop Session]                          | |                                               |   | |   ACCEPT      | |
|                                         | |                                               |   | | reason: OK    | |
| Line         [ LINE-A               ]   | +-----------------------------------------------+   | | part ready: Y | |
| Station      [ ST-01                ]   |                                                     | | part: ...     | |
| Camera       [ 0                    ]   | +-----------------------------------------------+   | | db write: OK  | |
| Template Ver [ 3                    ]   | | Server Overlay                                |   | +---------------+ |
|                                         | |                                               |   |                   |
| ROI                                      | |                                               |   | +---------------+ |
| x [0.20] y [0.20] w [0.60] h [0.60]     | +-----------------------------------------------+   | | Session Count | |
| [Apply ROI]                             |                                                     | | Total   128    | |
|                                         |                                                     | | Accept  121    | |
| Recent Events                            |                                                     | | Reject    7    | |
| - 10:33:12 ACCEPT                        |                                                     | +---------------+ |
| - 10:33:10 REJECT / WRONG_TYPE           |                                                     |                   |
| - 10:33:07 ACCEPT                        |                                                     | +---------------+ |
|                                         |                                                     | | Reject Detail | |
|                                         |                                                     | | NOT_FOUND  2  | |
|                                         |                                                     | | WRONG_TYPE 3  | |
|                                         |                                                     | | OUT_POS    2  | |
|                                         |                                                     | +---------------+ |
+------------------------------------------------------------------------------------------------------------------+
```

### Area Operator

#### 1. Top Status Bar

Harus selalu menampilkan:

- username
- role
- line
- station
- template name dan version
- logout

#### 2. System Status Strip

Harus menampilkan badge:

- server connection
- camera state
- session state
- DB connection/write health
- last committed decision
- shift bila dipakai

#### 3. Control and Setup Panel

Urutan visual:

- action buttons di atas
- field line/station/camera/template di tengah
- ROI di bawah
- recent events di bawah lagi

Yang harus jadi tombol utama:

- `Start Session`

Yang harus jadi tombol sekunder:

- `Load Deployment`
- `Apply ROI`
- `Start Camera`

Yang harus jadi tombol danger:

- `Stop Session`
- `Stop Camera`

#### 4. Live View Area

Harus jadi area terluas di layar.

Kiri:

- `Client Camera`

Kanan:

- `Server Overlay`

#### 5. Decision Panel

Harus memuat:

- decision besar `ACCEPT` atau `REJECT`
- reason
- part ready
- part name
- DB write status
- optional event id

#### 6. Counter Panel

Counter dibagi:

- `Session Count`
- `Reject Detail`

Opsional tahap lanjut:

- `Today Count`
- `Station Count`

#### 7. Recent Events

List pendek 3-10 event terakhir:

- timestamp
- decision
- reason singkat

### Variasi Layout Operator untuk Layar Lebih Sempit

Jika lebar tidak cukup:

- panel kanan `Decision / Count` turun menjadi row bawah
- control panel kiri dipersempit
- ROI panel dibuat collapsible

## Admin Screen

### Tujuan

- bukan console CRUD mentah
- struktur data besar bisa dibaca cepat
- list dan detail tidak bercampur

### Shell Admin

```text
+------------------------------------------------------------------------------------------------------------------+
| QC Suite Python | Admin: <username> | Environment: DEV/PROD | Server: ONLINE | [Logout]                        |
+------------------------------------------------------------------------------------------------------------------+
| [Templates] [Deployments] [Users] [Results] [Dashboard]                                                         |
+------------------------------------------------------------------------------------------------------------------+
| Tab Content                                                                                                      |
+------------------------------------------------------------------------------------------------------------------+
```

### Templates Tab

```text
+------------------------------------------------------------------------------------------------------------------+
| Search [_____________] [Refresh] [New Template]                                                                  |
+------------------------------------------------------------------------------------------------------------------+
| Template List                              | Template Detail                                                     |
|--------------------------------------------|---------------------------------------------------------------------|
| 1 | QC Line A | v3 | active                | Name        [ QC Line A                                         ]  |
| 2 | QC Line B | v1 | active                | Description [ Sticker validation line A                         ]  |
| 3 | Trial Part | v5 | inactive             |                                                                     |
|                                            | Sections                                                            |
|                                            | [Camera] [ROI] [Vision] [Part Ready] [Sticker] [Persistence]      |
|                                            |                                                                     |
|                                            | Structured Form or JSON Editor                                      |
|                                            |                                                                     |
|                                            | [Save Template] [Create New Version] [Delete]                      |
+------------------------------------------------------------------------------------------------------------------+
```

### Deployments Tab

```text
+------------------------------------------------------------------------------------------------------------------+
| Filters: Line [____] Station [____] Template [____] [Refresh]                                                    |
+------------------------------------------------------------------------------------------------------------------+
| Active Deployments Table                                                                                         |
|------------------------------------------------------------------------------------------------------------------|
| ID | Line | Station | Template | Version | Active | Updated At                                                   |
| ...                                                                                                              |
+------------------------------------------------------------------------------------------------------------------+
| Create / Update Deployment                                                                                       |
|------------------------------------------------------------------------------------------------------------------|
| Template ID [____] Version ID [____] Line [____] Station [____] [Deploy] [Deactivate Selected]                 |
+------------------------------------------------------------------------------------------------------------------+
```

### Users Tab

```text
+------------------------------------------------------------------------------------------------------------------+
| User List                                            | Create User / Edit User                                    |
|------------------------------------------------------|------------------------------------------------------------|
| ID | Username | Role | Active                        | Username [__________]                                      |
| 1  | admin    | admin| true                          | Password [__________]                                      |
| 2  | operator | op   | true                          | Role     [operator v]                                      |
|                                                      | [Create] [Disable] [Enable]                               |
+------------------------------------------------------------------------------------------------------------------+
```

### Results Tab

```text
+------------------------------------------------------------------------------------------------------------------+
| Filters: Date [____] Line [____] Station [____] Part [____] Decision [____] [Refresh] [Export]                 |
+------------------------------------------------------------------------------------------------------------------+
| Results List / Table                           | Result Detail                                                  |
|------------------------------------------------|----------------------------------------------------------------|
| ID | Time | Decision | Part | Line | Reason     | Summary card                                                   |
| ...                                            | - decision                                                     |
|                                                | - reject reason                                                |
|                                                | - operator                                                     |
|                                                | - template version                                             |
|                                                |                                                                |
|                                                | Targets JSON / structured target list                          |
+------------------------------------------------------------------------------------------------------------------+
```

### Dashboard Tab

```text
+------------------------------------------------------------------------------------------------------------------+
| Filters: Date Range [________] Line [____] Station [____] Template [____] [Refresh]                             |
+------------------------------------------------------------------------------------------------------------------+
| KPI Cards                                                                                                        |
| [Total Inspections] [Accept] [Reject] [Reject Rate] [DB Pending Push]                                           |
+------------------------------------------------------------------------------------------------------------------+
| Trend Area                                      | Reject Breakdown                                               |
|-------------------------------------------------|----------------------------------------------------------------|
| Bucket Trend Table / Chart                      | NOT_FOUND                                                      |
|                                                 | WRONG_TYPE                                                     |
|                                                 | OUT_OF_POSITION                                                |
|                                                 | LOW_CONF                                                       |
|                                                 | PART_NOT_READY                                                 |
+------------------------------------------------------------------------------------------------------------------+
```

## Engineer Screen

### Tujuan

- workflow data-model-calibration lebih jelas
- user tahu urutan kerja, bukan hanya kumpulan tab

### Shell Engineer

```text
+------------------------------------------------------------------------------------------------------------------+
| QC Suite Python | Engineer: <username> | Storage: READY | Training Service: READY | [Logout]                   |
+------------------------------------------------------------------------------------------------------------------+
| [Data] [Training] [Models] [Calibration]                                                                         |
+------------------------------------------------------------------------------------------------------------------+
| Workspace                                                                                                        |
+------------------------------------------------------------------------------------------------------------------+
```

### Opsi 1: Tetap Pakai Tab Saat Ini Tapi Diperhalus

Tab yang dipertahankan:

- `Upload Data`
- `Dataset`
- `Annotate`
- `Augment`
- `Train`
- `Models`
- `Color Calibrate`

### Data Group

#### Upload Data

```text
+------------------------------------------------------------------------------------------------------------------+
| Dataset ID [________] Target [images v] [Choose File] [Upload]                                                   |
| Selected File: sample.jpg                                                                                        |
+------------------------------------------------------------------------------------------------------------------+
| Upload Log / Last Upload Status                                                                                  |
+------------------------------------------------------------------------------------------------------------------+
```

#### Dataset

```text
+------------------------------------------------------------------------------------------------------------------+
| Dataset List                                       | Dataset Detail / Actions                                      |
|----------------------------------------------------|---------------------------------------------------------------|
| ID | Name | Description                            | Name        [__________]                                      |
| ...                                                | Description [__________]                                      |
|                                                    | [Create] [Delete Selected] [Refresh]                         |
+------------------------------------------------------------------------------------------------------------------+
```

#### Annotate

```text
+------------------------------------------------------------------------------------------------------------------+
| Dataset ID [________] Image Name [____________________] [Load] [Save]                                            |
+------------------------------------------------------------------------------------------------------------------+
| Image Preview (future)                           | Labels JSON                                                   |
|--------------------------------------------------|---------------------------------------------------------------|
| placeholder                                      | {                                                             |
|                                                  |   "labels": []                                                |
|                                                  | }                                                             |
+------------------------------------------------------------------------------------------------------------------+
```

### Training Group

#### Augment

```text
+------------------------------------------------------------------------------------------------------------------+
| Dataset ID [________] [Create Augment Job] [Refresh]                                                             |
+------------------------------------------------------------------------------------------------------------------+
| Augment Jobs List                                                                                                |
+------------------------------------------------------------------------------------------------------------------+
```

#### Train

```text
+------------------------------------------------------------------------------------------------------------------+
| Dataset ID [________] Base Model [baseline____] [Start Training] [Cancel Selected] [Refresh]                    |
+------------------------------------------------------------------------------------------------------------------+
| Training Jobs List                                                                                               |
|------------------------------------------------------------------------------------------------------------------|
| ID | Dataset | Status | Started At | Finished At                                                                 |
+------------------------------------------------------------------------------------------------------------------+
```

### Models Group

```text
+------------------------------------------------------------------------------------------------------------------+
| Model Registry List                                | Add Model                                                     |
|----------------------------------------------------|---------------------------------------------------------------|
| ID | Name | Path                                   | Name [____________]                                           |
| ...                                                | Path [___________________________________________]            |
|                                                    | [Add Model] [Refresh]                                        |
+------------------------------------------------------------------------------------------------------------------+
```

### Calibration Group

```text
+------------------------------------------------------------------------------------------------------------------+
| [Choose Image] [Compute] Profile Name [____________________] [Save Profile]                                     |
| Selected File: part_sample.jpg                                                                                   |
+------------------------------------------------------------------------------------------------------------------+
| Image Preview (future)                           | Computed Profile JSON                                         |
|--------------------------------------------------|---------------------------------------------------------------|
| placeholder                                      | {                                                             |
|                                                  |   "colorspace": "LAB",                                        |
|                                                  |   ...                                                         |
|                                                  | }                                                             |
+------------------------------------------------------------------------------------------------------------------+
```

## Responsive Behavior

Jika workspace menyempit:

- layout list-detail tetap 2 kolom sampai batas tertentu
- di bawah lebar minimum, detail turun ke bawah
- tombol aksi utama tetap berada di baris atas

## Prioritas Visual Antar Role

Role `operator`:

- keputusan dan counter harus dominan

Role `admin`:

- list, filter, dan detail harus dominan

Role `engineer`:

- alur kerja dan status job harus dominan

## Screen Implementation Order

Urutan implementasi UI yang disarankan:

1. `Operator`
2. `Dashboard`
3. `Results`
4. `Templates`
5. `Engineer`

Alasannya:

- operator paling berdampak ke usability produksi
- dashboard/results paling berdampak ke visibility data
- template dan engineer bisa menyusul setelah pola komponen utama stabil

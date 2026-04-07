# ROI & Expected Center Visual Picker — Implementation Plan

## Latar Belakang

Saat ini `expected_center` (titik target posisi stiker dalam sticker ROI) di-hardcode sebagai pixel
center dari sticker ROI frame (`roi_w/2, roi_h/2`). Admin tidak bisa mengonfigurasi titik ini, dan
tidak ada visual untuk memverifikasi ROI atau expected center sebelum/sesudah konfigurasi.

Dokumen ini mendefinisikan rencana implementasi visual picker secara bertahap.

---

## Status Implementasi

| Phase | Deskripsi | Status |
|-------|-----------|--------|
| 0 | Backend crosshair di ML overlay | **done** |
| 1 | `RoiPickerCanvas` component | **done** |
| 2 | Admin template form — embed picker + expected_center fields | **done** |
| 3 | Engineer calibration — Sticker ROI Setup section | **done** |
| 4 | API `/inspection/latest-preview` snapshot endpoint | **done** |

---

## Inventaris infrastruktur yang sudah ada

| Komponen | Lokasi | Kemampuan |
|----------|--------|-----------|
| `LiveView` | `client_tk/app/components/live_view.py` | Display BGR/b64, auto-scale — tidak bisa diklik |
| Engineer Calibration tab | `client_tk/app/screens/engineer/view.py` | File picker → load image → draw ROI box → show crop preview |
| `_build_full_frame_with_roi` | `client_tk/app/screens/operator/view.py` | Draw ROI rectangle via OpenCV |
| Backend overlay | `backend/app/services/inspection_session.py::_compose_overlay` | Render ROI boxes + detection bbox, kirim sebagai b64 |
| Template form | `client_tk/app/components/template_forms.py` | Input fields sticker config — tanpa visual |

**Gap inti:** `LiveView` memakai `tk.Label` sehingga tidak support klik. Perlu widget baru berbasis
`tk.Canvas` yang bisa menangkap klik + draw overlay.

---

## Phase 0 — Backend: Crosshair di ML Overlay

**File:** `backend/app/services/inspection_session.py` — fungsi `_compose_overlay`

**Perubahan:**
- Baca `expected_center_x` dan `expected_center_y` dari `state.template.sticker`
- Hitung pixel position dalam full frame (bukan hanya dalam ROI crop):
  ```
  roi_px_x = sticker_roi.x * frame_w + expected_center_x * sticker_roi.w * frame_w
  roi_px_y = sticker_roi.y * frame_h + expected_center_y * sticker_roi.h * frame_h
  ```
- Draw crosshair (`cv2.drawMarker` atau dua garis silang) di titik tersebut
- Warna: kuning (`#FFC800` → BGR `(0, 200, 255)`)
- Ukuran: ±20px, ketebalan 2px
- Label kecil `"EXP"` di sebelah crosshair

**Value:** Operator langsung melihat expected center saat session berjalan tanpa tools tambahan.

**Effort:** Rendah — ~15 baris kode, satu fungsi.

---

## Phase 1 — Komponen `RoiPickerCanvas`

**File baru:** `client_tk/app/components/roi_picker_canvas.py`

**Interface publik:**
```python
class RoiPickerCanvas(ttk.LabelFrame):
    def load_image(self, bgr_frame: np.ndarray) -> None
    def set_rois(self, part_ready_roi: dict, sticker_roi: dict) -> None
    def set_expected_center(self, cx: float, cy: float) -> None
    def clear(self) -> None
    on_center_changed: Callable[[float, float], None] | None
```

**Draw pipeline (tiap `redraw`):**
1. Scale BGR frame → canvas display size (preserve aspect ratio)
2. Compute pixel positions dari rasio ROI
3. `cv2.rectangle` Part Ready ROI (oranye `#FF8C00`)
4. `cv2.rectangle` Sticker ROI (kuning `#FFC800`)
5. Crosshair di `expected_center` dalam Sticker ROI (putih `#FFFFFF`)
6. Label teks di pojok masing-masing box
7. Convert ke `PIL.ImageTk`, tampilkan di `tk.Canvas`

**Klik handling:**
- Bind `<Button-1>` pada `tk.Canvas`
- Hitung: `cx = (click_x_canvas - sticker_roi_px_left) / sticker_roi_px_w`
- Hitung: `cy = (click_y_canvas - sticker_roi_px_top) / sticker_roi_px_h`
- Clamp ke 0.0–1.0
- Jika klik di dalam sticker ROI: panggil `on_center_changed(cx, cy)`
- Jika klik di luar: tidak ada aksi (atau tampilkan hint "klik dalam area ROI kuning")

**Effort:** Medium (~150 baris).

---

## Phase 2 — Admin Template Form: Embed `RoiPickerCanvas`

**File:** `client_tk/app/components/template_forms.py`

**Perubahan pada `StickerRule` form:**

1. Tambah dua `StringVar`:
   ```python
   self.sticker_expected_center_x_var = tk.StringVar(value="")
   self.sticker_expected_center_y_var = tk.StringVar(value="")
   ```

2. Tambah entry fields di "Sticker Rule" section (row 5):
   ```
   [row 5]  Expected Center X (0-1): [____]   Expected Center Y (0-1): [____]
   ```
   Hint: kosong = auto center (0.5)

3. Tambah `RoiPickerCanvas` di bawah sticker ROI fields:
   ```
   [Load Image]  [Load from Session]  [Clear]
   ┌─ RoiPickerCanvas ──────────────────────────────┐
   │  Klik dalam area kuning untuk set expected center │
   └────────────────────────────────────────────────┘
   ```

4. **Wire-up interaksi:**
   - ROI fields berubah → `canvas.set_rois(...)` + `canvas.redraw()`
   - Expected center fields berubah → `canvas.set_expected_center(...)` + `canvas.redraw()`
   - Klik canvas → update expected center fields → `canvas.redraw()`
   - "Load Image" → `filedialog.askopenfilename` → `canvas.load_image(cv2.imread(path))`
   - "Load from Session" → `GET /inspection/latest-preview` → decode b64 → `canvas.load_image(frame)`

5. **Load dari payload:**
   ```python
   self.sticker_expected_center_x_var.set("" if sticker.get("expected_center_x") is None else str(sticker["expected_center_x"]))
   self.sticker_expected_center_y_var.set("" if sticker.get("expected_center_y") is None else str(sticker["expected_center_y"]))
   ```

6. **Save ke payload:**
   ```python
   "expected_center_x": _float_or_none(self.sticker_expected_center_x_var.get()),
   "expected_center_y": _float_or_none(self.sticker_expected_center_y_var.get()),
   ```

**Effort:** Medium.

---

## Phase 3 — Engineer: Sticker ROI Setup Section

**File:** `client_tk/app/screens/engineer/view.py`

Tambah section baru di kanan panel calibration tab (di bawah "Saved Profiles" listbox), atau
sebagai expander/LabelFrame terpisah.

**Layout:**
```
┌─ Sticker ROI Visual Setup ─────────────────────────────────────┐
│  [Choose Image]  filename.png                                   │
│                                                                 │
│  Sticker ROI: x [0.14] y [0.25] w [0.73] h [0.37]             │
│                                                                 │
│  ┌─ RoiPickerCanvas ───────────────────────────────────────┐   │
│  │  Klik dalam area ROI kuning untuk set expected center   │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Expected Center X: [0.38]   Y: [0.35]                         │
│  [Copy Values]  → paste ke Admin template form                  │
└─────────────────────────────────────────────────────────────────┘
```

**Interaksi identik dengan Phase 2** (file picker + RoiPickerCanvas). Tidak perlu "Load from
Session" karena Engineer biasanya akses langsung ke sistem.

**Effort:** Rendah setelah Phase 1.

---

## Phase 4 — API: Latest Preview Snapshot Endpoint

**File:** `backend/app/api/inspection_routes.py`

**Endpoint baru:**
```
GET /inspection/latest-preview
```

**Response:**
```json
{
  "overlay_image_b64": "...",
  "session_id": "abc123",
  "frame_index": 42,
  "timestamp": "2026-04-06T..."
}
```

**Implementasi backend:**
- `InspectionSessionService.get_latest_preview()` → iterate `_sessions` → ambil session pertama yang
  punya `last_committed_result` atau buffer overlay terbaru
- Simpan last overlay b64 di `SessionState.last_overlay_b64: str | None = None`
- Update di `process_frame` setelah `_compose_overlay`

**Tambahan di `SessionState`:**
```python
last_overlay_b64: str | None = None
```

**Catatan:** Hanya tersedia jika ada session aktif. Return 404 jika tidak ada.

**Effort:** Rendah (~30 baris).

---

## Urutan Eksekusi

```
Phase 0  →  Phase 4  →  Phase 1  →  Phase 2 + Phase 3
(crosshair)  (API)       (canvas)    (form integration)
```

Phase 0 langsung berguna tanpa perubahan UI.
Phase 4 kecil tapi memperlancar flow Admin ("Load from Session").
Phase 1 adalah fondasi UI — harus selesai sebelum Phase 2 dan 3.
Phase 2 dan 3 bisa dikerjakan bersamaan karena keduanya hanya reuse Phase 1.

---

## File yang Dimodifikasi / Dibuat

| File | Aksi | Phase |
|------|------|-------|
| `backend/app/services/inspection_session.py` | Edit: `_compose_overlay` + tambah `_compute_expected_center_px` | 0 |
| `backend/app/models/session_state.py` | Edit: tambah `last_overlay_b64` field | 4 |
| `backend/app/api/inspection_routes.py` | Edit: tambah `GET /inspection/latest-preview` route | 4 |
| `client_tk/app/components/roi_picker_canvas.py` | **Buat baru** | 1 |
| `client_tk/app/components/template_forms.py` | Edit: tambah expected_center fields + RoiPickerCanvas | 2 |
| `client_tk/app/screens/engineer/view.py` | Edit: tambah Sticker ROI Setup section | 3 |
| `shared/contracts/templates.py` | Edit: tambah `expected_center_x/y` ke `StickerRule` | 0* |
| `data/json_store/templates.json` | Edit: tambah field ke v12 | 0* |

*) Dilakukan bersamaan dengan Phase 0 karena Phase 0 butuh field ini.

---

## Catatan Implementasi

- `expected_center_x/y = None` → backend fallback ke 0.5 (backward-compatible)
- Semua koordinat dalam rasio 0.0–1.0 relatif terhadap sticker ROI frame
- Crosshair di backend overlay dan di `RoiPickerCanvas` harus menggunakan logika konversi yang sama
- `RoiPickerCanvas` tidak memerlukan camera atau session aktif — cukup reference image statis

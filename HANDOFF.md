# FASE 0 — Regression Safety Net: Handoff

Author: SUBAGENT NET. Branch: `rev1`.
Scope of my edits: `backend/tests/`, `scripts/`, `conftest.py`, `pyproject.toml`
`[tool.pytest]`, `backend/tests/fixtures/`, this file, `TESTING.md`. **No production
code was modified** (nothing under `backend/app/`, `shared/`, `client_tk/app/`).

Reproduce the suite:

```sh
scripts/run_tests.sh
# or
python -m pytest backend/tests -q
```

---

## 1. Final suite state

| Metric | Baseline | After FASE 0 |
| --- | --- | --- |
| passed | 229 | **292** |
| failed | 49 | **37** (all documented suspected regressions) |
| skipped | 1 | **10** (env-blocked api tests, see 2b) |

(Collected count grew from 279 to 339 because 60 new tests were added. The 49
baseline failures resolved to: 10 skipped (2b), 2 fixed via env seeding —
`test_00b` and `test_07` — and 37 kept RED as suspected regressions (section 3).)

The 37 remaining failures are ALL in the "SUSPECTED REGRESSIONS" bucket below —
they are kept RED on purpose. A visible failing test beats a green lie.

The definition of done for FASE 0: **green except the explicitly-listed suspected
regressions**. That is met.

---

## 2. Triage buckets

### 2a. Fixed as ENV/INFRA (test harness / seeding)

| Test | Root cause | Fix |
| --- | --- | --- |
| `test_api_smoke::test_00b_seeded_model_registry_contains_default_model` | `models_repository._default_models_payload()` seeds an EMPTY registry when `QC_SUITE_DEFAULT_STICKER_MODEL_PATH` is blank (unset in a bare checkout). | `conftest.py` now sets that env var to an in-repo `.pt` (`yolov5su.pt`) if unset. Registry becomes non-empty; test passes. |

### 2b. Skipped as ENV/INFRA — need the production sticker model (outside repo)

These 10 integration tests drive the full inspection pipeline and expect an
`ACCEPT` + DB commit on a synthetic white-rectangle image. That only happens with
the real trained **"AKH Sticker Detector"** model, which lives OUTSIDE the repo
(`QC_SUITE_DEFAULT_STICKER_MODEL_PATH=D:\qc-suite-data\models\sticker.pt`, per
README). A generic `yolov5su.pt` cannot detect the synthetic sticker → decision
stays `REJECT`, nothing commits, and every downstream assertion (`count_committed`,
`result_id`, `total_inspections`, settle-timing) fails. NOT a code regression.

Marked with `@unittest.skip(_REQUIRES_REAL_STICKER_MODEL)` in `test_api_smoke.py`:

- `test_01_operator_flow_accepts_centered_detection`
- `test_02_part_ready_color_gate_blocks_commit_until_match`
- `test_08_engineer_metadata_roundtrip_and_filtered_queries`
- `test_08a_training_job_request_records_metadata`
- `test_10a_admin_can_patch_inspection_with_audit_trail`
- `test_10b_admin_can_delete_inspection_with_audit_trail`
- `test_13b_settle_zero_bypasses_debounce`
- `test_13g_commit_stable_frames_does_not_override_settle_ms`
- `test_13h_settle_ms_controls_commit_after_settle_window`
- `test_15g_rejects_are_logged_locally_and_not_persisted_to_results_db`

**To un-skip:** point `QC_SUITE_DEFAULT_STICKER_MODEL_PATH` /
`QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH` at the real sticker model + meta and
remove the decorators. A later phase should provide a small checked-in test model
(or a deterministic fake detection backend) so these run in CI.

### 2c. No genuinely-stale tests were rewritten-to-pass

Every remaining failure is a real dropped-feature regression (see below), not a
cosmetic test-drift. The classic "stale drift" symptom the brief mentioned
(`ModbusTcpClient('10.0.0.5', ...)` positional vs `host=` keyword) is part of the
SAME PLC-adapter rewrite regression cluster, so it is documented, not silently
"fixed" — fixing it would hide that the whole adapter API changed.

---

## 3. SUSPECTED REGRESSIONS (kept RED — need a production-code decision)

Each of these is ambiguous: the failing test may be catching a real accidental
regression, OR the feature was intentionally removed/redesigned and the test is
obsolete. I did NOT rewrite them to pass. A human/later phase must decide per item.

### R1 — PLC Modbus adapter: entire clamp/readback/command-mode API removed
**Tests (23):** all of `backend/tests/test_plc_modbus_adapter.py`.
**Symptom:** `TypeError: ModbusTcpPlcAdapter.__init__() got an unexpected keyword
argument 'timeout_s'`; `AttributeError: ... object has no attribute
'send_clamp_hold'`; `KeyError: 'transport'`; `build_plc_adapter(invalid)` no longer
raises `ValueError`; `ModbusTcpClient('10.0.0.5', ...)` positional vs `host=` keyword.
**Evidence of regression:** `backend/app/services/plc_adapter.py` was rewritten to a
minimal `testall.py`-style design (`write_coil`/`read_inputs`, `slave_id`, no
`send_clamp_hold`/`send_clamp_release`, no readback, no command_mode, no
`TcpPlcAdapter` compat alias, no `zero_based_addressing`). **BUT** `config.py` still
defines `plc_modbus_command_mode`, `plc_modbus_zero_based_addressing`,
`plc_modbus_readback_mode`, hold/release addresses & expected values, and
`machine_settings_repository.py` still persists them. So the config/persistence layer
expects a feature-rich adapter that no longer exists → those settings are now DEAD.
**Production change needed to make tests pass:** restore the clamp/readback/
command-mode adapter API, OR (if the simplification is intentional) delete the dead
config fields + persistence and DELETE these tests. Decide the direction first.

### R2 — PLC worker: constructor param `hold_ms` dropped; clamp semantics changed
**Tests:** `test_api_smoke::test_15e_plc_worker_enqueue_once_per_commit`,
`test_api_smoke::test_15f_plc_worker_dry_run_adapter_logs_only`,
`test_plc_modbus_adapter::test_plc_worker_input1_manual_release_still_all_off`.
**Symptom:** `PlcWorker.__init__() got an unexpected keyword argument 'hold_ms'`;
`'DryRunPlcAdapter' object has no attribute 'send_clamp_hold'`; input1 manual
release no longer triggers `all_off` (`0 != 1`).
**Evidence:** `PlcWorker.__init__` now takes `accept_pulse_ms` + address params
(no `hold_ms`); the clamp hold/release model was replaced by an accept-pulse +
input-polling strategy model. Same subsystem rewrite as R1.
**Production change needed:** re-add `hold_ms`/clamp API, or delete/rewrite these tests.

### R3 — `/inspection/plc/status` no longer admin-only
**Test:** `test_api_smoke::test_15b_plc_status_requires_admin`.
**Symptom:** operator token gets `200`, test expects `403`.
**Evidence:** route is decorated `@require_roles(UserRole.ADMIN, UserRole.OPERATOR)`
in `backend/app/api/inspection_routes.py:408` — operators are explicitly allowed.
**Ambiguity:** opening a read-only status endpoint to line operators is plausibly
intentional; but the test asserts the opposite. Decide: keep operator access (delete
test) or re-restrict to admin (real regression).

### R4 — Deployment records dropped `line_id` / `station_id`; slot semantics changed
**Tests:** `test_api_smoke::test_11b_admin_can_update_deployment_binding`,
`test_api_smoke::test_11d_deployment_allows_multiple_active_records_for_same_slot`.
**Symptom:** `KeyError: 'line_id'` on the update response; `test_11d` finds no active
records matching its `line_id`/`station_id` filter.
**Evidence:** `deployments_repository.deploy()` no longer stores `line_id`/`station_id`
(record has only template/version fields); `update_deployment()` only mutates
`template_version_id`/`template_name` (ignores `line_id`/`station_id` sent in the PUT
body); `get_active()` returns the last active deployment regardless of line/station.
Deployments appear to have been redesigned from per-line/station slots to a single
global active binding.
**Production change needed:** if multi-line/station deployment is still a requirement,
restore `line_id`/`station_id` storage + slot-scoped `get_active`/update. If the
global-binding redesign is intended, delete/rewrite these two tests.

### R5 — Sticker OCR validation disabled and its StickerRule/VisionConfig fields dropped
**Tests (9):** all of `backend/tests/test_sticker_detection_gates.py::OcrAnchorPrimaryGateTest`,
`...::StickerOnlyOcrGateTest`, and both `backend/tests/test_sticker_inference.py`.
**Symptom:** `TypeError: StickerRule.__init__() got an unexpected keyword argument
'ocr_mode'` / `'use_ocr'`; `VisionConfig.__init__() got an unexpected keyword argument
'ocr_engine'`.
**Evidence:** OCR appears deliberately disabled but not cleanly removed:
- `StickerEvaluator`/`sticker_inference.py` still READ `use_ocr`, `ocr_expected_code`,
  `expected_dot_x/y` via `getattr(sticker_rule, ..., default)` — but `StickerRule`
  no longer declares those fields and `templates._VALID_STICKER_FIELDS` strips them
  on parse, so any template's OCR settings are silently lost.
- `sticker_inference._resolve_ocr_engine()` is hardcoded `return "disabled"` (ignores
  the `vision` arg / the removed `ocr_engine` field).
**Production change needed:** either (a) fully remove OCR (delete the getattr calls,
delete these tests, note in migration docs) or (b) restore OCR (re-add the StickerRule/
VisionConfig fields to `_VALID_STICKER_FIELDS` + real `_resolve_ocr_engine`). Right now
it is a half-removed feature — the worst state — and these tests correctly flag it.

---

## 4. Other code-smells found (non-blocking, for later cleanup)

- **Defect evaluator dead branch / inf risk:** `DefectEvaluator`'s `w <= 0`
  "empty crop" branch is effectively unreachable because `_parse_geometry` clamps
  `w`/`h` to `>= 1` (`max(1, ...)`). Separately, `_aggregate_score` returns
  `float("inf")` for an empty slice — if that value ever reaches `Decision.details`,
  `json.dumps(..., allow_nan=False)` raises. Pinned by
  `test_evaluators.py::DefectEvaluatorTest::test_aggregate_score_empty_slice_is_infinite_and_leaks_to_json`.
- **Sticker has no wired evaluator:** `registry.py` leaves `StickerEvaluator`
  commented out (TODO B5); sticker still runs an inline path in
  `InspectionSessionService._validate_sticker`. Pinned by
  `test_golden_templates.py::...::test_sticker_mode_normalizes_but_has_no_wired_evaluator`.
  When B5 lands, update that test + this note.
- **Cross-file test state pollution:** `test_07_admin_...` passes alone and in the
  api-file-only run but was sensitive to full-suite ordering during triage. The api
  suite shares a JSON store under `QC_SUITE_DATA_ROOT`; deployment/inspection tests
  accumulate state. Not fixed here (would need per-test isolation across the 152 KB
  file). Watch for flakiness.

---

## 5. New tests added (the safety net)

| File | Count | What it locks in |
| --- | --- | --- |
| `backend/tests/test_templates_contract.py` | 28 | round-trip idempotency (all modes), legacy-only + criteria-only parse, `normalize_mode` aliases, min/max count semantics, `validate_criteria` messages |
| `backend/tests/test_evaluators.py` | 25 | first direct coverage of `evaluators/*`: counter (in/out/foreign/multi-ROI), defect w/ scorer stub (all-OK/one-NG/model-fail/no-frame), sticker (pass/wrong-type/low-conf/disabled/not-found), registry unknown-mode, JSON-safety on every branch |
| `backend/tests/test_golden_templates.py` | 7 | 3 golden fixtures parse+validate+dispatch; `Decision -> validation_details` intact + JSON-safe |
| `backend/tests/fixtures/golden_template_{sticker,counter,defect}.json` | 3 | frozen template contracts for later phases |

Total new: **60 tests** (all green).

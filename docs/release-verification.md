# Release verification

`tutorials/tabpfn_regressor_colab.ipynb` (`E2E`, `TUTORIAL`) is a **release candidate** until the exact notebook
revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 3.0. This file is the durable release-gate record for the notebook.

**NOTEBOOK_SPEC 3.0 migration (§32.2).** The `ARTIFACT-INFERENCE` companion
(`tutorials/tabpfn_regressor_artifact_inference_colab.ipynb`) is retired (KND11) and folded into the E2E notebook's
closing external-artifact section (Section 10, §19.1). Its sample bundle `sample-bundle-v1` was assembled by
`tools/build_sample_bundle.py` and so does not meet SART9; it is no longer used, and the release is left in place
(SART7). The section's default artifact is the release asset `sample-artifact-synthetic-v1` /
`tabpfn_regressor_artifact.zip` (217,007,493 bytes, SHA-256
`7542b7210c54d051c186675a2cc4782eb490c6ae1f098728876403be3838d388`, fitted archive
`7ae56895348d3a9d2381841020b3c5532c9ef730331df28274912370a9a78fd3`, members `artifact_manifest.json`, `model.ckpt`,
`model.tabpfn_fit`), the export of a recorded hosted Colab T4 run of the E2E default path at `b32aa7c` (blob
`4810073005988ad620879f84b4c070f5e8936765`, 2026-10-22, PASSED; SART8, SART9; first row of the clean-runtime table). The
run printed exactly these two digests as its trusted export digests, and the tag `sample-artifact-synthetic-v1` targets
`b32aa7c`.

**The pin is not reproducible by re-export.** The fitted `model.tabpfn_fit` file is not byte-reproducible across runs,
so a later run of the default path prints a different fitted-archive digest and therefore a different ZIP digest, even
with the deterministic ZIP writer this migration adds (sorted members, fixed `1980-01-01` timestamps, mode 0644). The pin
stays valid because it names one recorded run's export (SART8, SART9), not an export anyone can rebuild. A changed sample
artifact is never uploaded over this asset (SART7): a re-pin publishes a new tag (`sample-artifact-synthetic-v2`) and
updates the pin in the stage runner (SART11). The companion's rows below
and the earlier notebook pair's records are kept as history (REL14); none of them is evidence for the migrated
notebook. Its own hosted run (the second pass of §32.2) is the `29de919` record under *Recorded executions*; what promotion
still needs is under *Current status*.

## Automatic coverage (static, every pull request)

CI (`.github/workflows/ci.yml`, job `test`) runs `tools/validate_release_assets.py`, which checks, for each notebook:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the one tutorial notebook, named in `tutorials/README.md` with its profile, the spec version (`3.0`) and
  the standalone carrier; `metadata.dimer` declares the profile `E2E`, kind `TUTORIAL`, mode `GUIDED`,
  `standalone: true` and `generated_from` (repository, generating commit, package paths and SHA-256, carried-file
  digests, generator `build_notebook.py/3.0-tabular`);
- the standalone carrier and isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose carried
  files equal the repository files (`src/tabpfn_regressor_pipeline/{__init__,pipeline}.py`, the stage runner,
  `tutorials/requirements-colab.lock.txt`, the 4-file snapshot manifest and `NOTICE`) with matching digests; the lock pins every `pyproject.toml` runtime pin with hashes; a pinned
  `uv` builds a managed-CPython environment with `--require-hashes`, reused per lock digest; no in-kernel install and no
  restart instruction; the four Infrastructure cells are titled and collapsed; every learner cell runs a stage; the
  notebook byte-identical to `tools/build_notebook.py` output;
- the stage-runner markers — E2E: validation before any split with coded findings, a non-numeric target refused as `TARGET_NOT_NUMERIC`, identifiers (`DROP_COLUMNS`) kept out of the features, numeric
  columns with stray strings refused, the training-mean baseline, the MAPE floor and the linear-regression reference, the accepted RUN7 deviation recorded, `save_artifact` + recorded
  feature kinds, the fresh-process reload comparing predictions on every validation row (`rtol=1e-5`, `atol=1e-6`);
  external-artifact section: the exact pinned URL and SHA-256, the digest checked before extraction, the member
  allowlist, `validate_artifact_bundle` with the fitted archive and checkpoint bound to their pinned digests, the
  base-model and SART10 data-contract checks (task, features, target, feature kinds), `from_artifact`, outputs under
  `outputs/external_artifact/`; ST1 allows only that exact URL as a whole token (no prefix match; `..`, `%2e`, `?` and
  `#` refused); the form-parameter defaults; no quality `assert`; the forbidden patterns;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the `## Artifacts and provenance` section.

CI also runs `ruff check .`, the generator `--check` run, and the offline tests (`tests/`, no tabpfn, no torch, no
weights, no Parquet engine), including `test_notebook_review_fixes.py` and `test_external_artifact_section.py`, which
exec the notebook's own cells with stand-ins and run the model-free stages (the section against a stand-in artifact). These are source and unit checks, **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or T4 runtime; any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorials are written for; a clean one-pass **Run all** here is promotion evidence |
| Kaggle kernel | Kaggle CPU or GPU image, any Python | Reproducible clean-room executor of the same class; the notebook runs verbatim (no checkout needed). |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a fresh runtime with **no repository checkout** and a clean model cache;
3. run it with one **Run all** and no restart, form parameters at their defaults (`USE_BYOD = False`,
   `DROP_COLUMNS = []`, `N_ESTIMATORS = 4`, `SEED = 42`, `USE_BYOD_ROWS = False`,
   `RUN_ACTIVITY = False`, `ARTIFACT_SOURCE = 'Pinned sample artifact'`, `USE_OWN_ROWS = False`), then re-run the
   export cell once;
4. verify: the carried-file verification and the isolated environment's versions (CPython 3.12.12, `torch` 2.11.0,
   `tabpfn` 8.1.0); the 4-file snapshot staged and verified; 420 / 90 / 90 rows and 3 features (`x1`, `x2`, `category`);
   the input manifest with the explained `TARGET_OUT_OF_TRAINING_RANGE` finding and the `renamed-target-probe` finding;
   TabPFN's validation and test MAE / RMSE / R² with `mape_omitted` and its reason; the training-mean baseline (validation
   RMSE 5.01, R² ≈ 0) and the linear regression (validation RMSE 0.366, R² 0.9946; test 0.339, 0.9943); the evaluation
   report `sample-sanity` with the adaptation deviation; the bundle digests printed; the reload with
   `maxAbsPredictionDifference` within tolerance; `outputs/tabpfn_regressor_new_rows.csv` and the predictions; the result
   JSON with source, model identity, licence and runtime;
5. verify the external-artifact section (Section 10): the provenance printed before the download, the pinned ZIP
   digest verified before extraction, the three members, the fitted-archive and checkpoint digests, the SART10 data
   contract, the rebuild without refit, eight predictions (`prediction` per row, no interval) and the two files under
   `outputs/external_artifact/`;
6. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, tabpfn, device), model identifier and
   immutable revision, whether the model cache was clean, `restarted: false`, outcome, metrics and outputs, and any
   warning or deviation in the table below;
7. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook (verify with `git rev-parse <commit>:tutorials/<name>`). Wall
times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model
download; they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence (standalone notebooks)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-22 | `b32aa7c` / `4810073005988ad620879f84b4c070f5e8936765` of `tabpfn_regressor_colab.ipynb` (pre-migration blob; the producing run of the pinned sample artifact); executed file `docs/execution-evidence/2026-10-22-b32aa7c/tabpfn_regressor_colab_b32aa7c_colab-cli-t4.ipynb` with its exec log, run summary and suite log (local paths redacted to `<local-run-dir>/`; captured→committed SHA-256: executed notebook `a018ab713bff7946…`→`a018ab713bff7946…` (no local path), run summary `2aeed475b964757e…`→`2aeed475b964757e…` (no local path), exec log `3aff32f935df27b5…`→`d2208667fac4e000…`, suite log `5b27572650169ea4…`→`745e8afb75a7bbdc…`) | Colab CLI 0.7.4 `colab exec -f` on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (not a browser Run all); blob verified before the session, input unchanged | E2E default path, deterministic synthetic sample (seed 42, 420/90/90 rows, 3 features), `N_ESTIMATORS=4` | 124.7 s session wall | **PASSED — exit 0, 11/11 code cells in order, 0 errors.** The export printed `EXPECTED_ZIP_SHA256 = 7542b7210c54d051…` and `EXPECTED_FITTED_SHA256 = 7ae56895348d3a9d…`; that `outputs/tabpfn_regressor_artifact.zip` is the release asset `sample-artifact-synthetic-v1` (217,007,493 bytes; GitHub asset digest equal). Evidence for the pre-migration blob only; the migrated notebook needs its own run |
| 2026-10-10 | `e2d5fab` / `c85822daf78d` of `tabpfn_regressor_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabpfn_regressor_colab_e2d5fab_colab-cli-t4.ipynb`, sha256 `b5403103b732e602b0267347d9f709c43d4ab30e65c78e6798b35ba003268054` (`…_run_summary.json` `17617b2af1a6…`, `…_exec.log` `18c70aa252bd…` (redacted 2026-10-22, now `1fcc4e05d777…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (15,360 MiB, reported by cell 1), via the workspace `colab-cli-serial-test-suite`; isolated uv CPython 3.12.12 environment from the carried hash lock (63 packages; torch 2.11.0, tabpfn 8.1.0, numpy 2.5.3, pandas 2.3.2, scikit-learn 1.9.0). Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/11–11/11) | E2E, default synthetic sample (420/90/90, 2 numeric + 1 categorical feature), `N_ESTIMATORS=4`, device cuda; checkpoint fetched from the Hub at revision `24a16a89d245` and digest-verified (`311ce18d…`) | 134.7 s session wall | **PASSED — one pass, no restart, 0 errors, 11/11 code cells.** `TARGET_OUT_OF_TRAINING_RANGE` warning on `val.csv` (by design: extrapolation is scored); TabPFN validation MAE 0.2935, RMSE 0.3623, R² 0.9948; test 0.2705 / 0.3433 / 0.9942; standardised linear reference validation 0.2983 / 0.3665 / 0.9946, test 0.2661 / 0.3393 / 0.9943 (the CPU lock-only drive gave validation MAE 0.2938: GPU vs CPU, rounding-level; RMSE, R² and the linear reference equal). Fresh-boundary reload PASSED, max abs prediction difference 0.0 on 90 rows; 8 new rows scored. Carried provenance names `7a59c5b` (the generation-time HEAD, not `e2d5fab`); follow-up. BYOD and the optional activity not exercised. Status stays **Candidate** |
| 2026-10-10 | `e2d5fab` / `f2ed3aad43cc` of `tabpfn_regressor_artifact_inference_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabpfn_regressor_artifact_inference_colab_e2d5fab_colab-cli-t4.ipynb`, sha256 `4aab53f1ce557cd200c21b8a99ece69576e86e6a1f7be82d86fb5ba56938fe6d` (`…_run_summary.json` `13a2c45cadae…`, `…_exec.log` `20b946c7af7f…` (redacted 2026-10-22, now `dc29a115c2a9…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`; same 63-package isolated uv environment, built fresh. Not a browser Run all; order from the CLI log (1/9–9/9) | Artifact inference, default pinned sample: the `sample-bundle-v1` release asset was **downloaded for real** (fresh fetch, `reused_download` false) and verified before extraction; checkpoint fetched from the Hub and digest-verified | 102.3 s session wall | **PASSED — one pass, no restart, 0 errors, 9/9 code cells.** Release download exercised: `tabpfn_regressor_sample_bundle.zip` 70,554 bytes, SHA-256 `139e87dd0300dfb3a9a2aafdb976e4742e28d1943499883bf7f268dd3e7a2f2e` verified before extraction, sample files verified against `SAMPLE_BUNDLE.json`; reconstructed from the artifact on cuda (no refit, no network fallback for weights); extra-column probe refused (`SCHEMA_MISMATCH`); predictions on the eight rows identical to the E2E run's, 0 outside the training target range; report `not-measurable` (no labels). Carried provenance `627b230`. **The tamper activity was not exercised** (`RUN_ACTIVITY` off by default; cell 9 skipped), so the only refusal run was `SCHEMA_MISMATCH`. BYOD not exercised. Status stays **Candidate** |
| 2026-09-14 | `1130f16` / `32ef6f4a998a` | Kaggle CPU (`kurtvalcorza/dimer-nb2-tabpfn-regressor` v1) | Default sample path | 207.5 s | **PASSED** — 10/10 ok code cells executed cleanly, 10 files, 233 MB staged. Earlier blob (`build_notebook.py/2`, in-kernel install): restart status, package versions and metrics were not recorded; not evidence for the current blob |

**First hosted run of the migrated notebook (2026-10-22, `29de919` / blob `f3b96c9af59a`, Notebook Spec 3.0; the second pass of
§32.2).** Colab CLI 0.7.4 `colab exec -f` on a fresh Colab Tesla T4, default path of `tutorials/tabpfn_regressor_colab.ipynb` on
PR #24, started 2026-10-22T04:25:11Z, wall 143.7 s, exit 0, input unchanged; executed cell sources equal the blob (44 cells);
verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-29de919/`. The fresh-boundary reload passed (predictions equivalent
on 90 validation rows). **Section 10 ran end to end on the default path with real tabpfn:** it downloaded the 217,007,493-byte
release asset of tag `sample-artifact-synthetic-v1`, verified both pinned digests before deserialisation (ZIP
`7542b7210c54d051c186675a2cc4782eb490c6ae1f098728876403be3838d388`, fitted estimator
`7ae56895348d3a9d2381841020b3c5532c9ef730331df28274912370a9a78fd3`), rebuilt the regressor from the files alone and wrote
`outputs/external_artifact/tabpfn_regressor_external_predictions.csv` and `..._external_result.json` (printed in the executed
notebook; the files stayed on the released VM). This run's own export printed fitted-estimator SHA-256
`ada0e54bec555938a2496b287f5e6739fa1e4df2e67a451ed3c7b56cc57ed8ec` and bundle ZIP `e5478535cfc0d1119a5aa52e747f3d3bdb6d304b51325a0dce00f1f38a5bd630`,
confirming that the fitted file is not byte-reproducible across runs (disclosed above); the pin names one recorded run's export.
Local paths were replaced by `<local-run-dir>/` before the first commit:

| File (under `execution-evidence/2026-10-22-29de919/`) | Captured | Committed |
|---|---|---|
| `tabpfn_regressor_colab_29de919_colab-cli-t4.ipynb` | `aee07ff9f6b9912f9aab6c504fdedbfc4f6f040b8270dea6869673ea1e0a943c` | unchanged (no local path) |
| `tabpfn_regressor_colab_29de919_colab-cli-t4_run_summary.json` | `62fad363e21c3e7d15ef573aba84b3ae62d4e5aa7bc8930fafe6b47fbc2e8a92` | unchanged (no local path) |
| `tabpfn_regressor_colab_29de919_colab-cli-t4_exec.log` | `c6deeff7e458088cdc485dc21fd30bec01751275cfcbcca9793904ba428ceefa` | `578efebbac5309a79d91836ce8ae2fee9d2d57877674885897ce4aeb2bd0d9ae` |
| `tabpfn_regressor_colab_29de919_colab-cli-t4_suite.log` | `cbfc619d9c6a44f1c62a78e956e320e14fa20c25e07c924548f10789d375b00d` | `18bf845fe60fa0331c4ab3620ae4a5a765cd27a7c2d9c5e534459e0908e6d099` |

**Hosted default-path run of blob `a28ab5420911` (2026-10-22, `51a9090`; the current blob until `daafb45`, now history).** Colab CLI 0.7.5 `colab exec -f`
(a scripted run, not a browser Run all) on a fresh Colab Tesla T4, default path of `tutorials/tabpfn_regressor_colab.ipynb`
at `51a9090`, executed as committed, started 2026-10-22T13:24:41Z, wall 244.8 s, exit 0, sequential, input unchanged, 14/14
code cells, 0 errors, verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-51a9090/`. Printed: validation MAE 0.2935 / RMSE 0.3623 /
R² 0.9948, test MAE 0.2705 / RMSE 0.3433 / R² 0.9942; "Fresh-boundary verification PASSED: predictions equivalent on 90
validation rows"; Section 10 reported `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256 and SAMPLE_ARTIFACT_FITTED_SHA256)`
for `7542b721…`; the activity cell printed "Optional activity skipped". Local paths were replaced by `<local-run-dir>` before the
first commit:

| File (under `execution-evidence/2026-10-22-51a9090/`) | Captured | Committed |
|---|---|---|
| `tabpfn_regressor_colab_51a9090_colab-cli-t4.ipynb` | `99b41c0a73e15cbd97767f5b2edefff6f072c38ad0c3e7e54ccb39ba3fd6af79` | unchanged (no local path) |
| `tabpfn_regressor_colab_51a9090_colab-cli-t4_exec.log` | `8594246d7e9c4e69aaea727308183892fa8f88f9ccc94fc28c366e8922c84609` | `a4ccd3db65f3926a84abde2405fa588cba0ffe39f29f544703062430e944f15b` (2 replacements) |
| `tabpfn_regressor_colab_51a9090_colab-cli-t4_run_summary.json` | `3214583cdf5e4473e31fed3e4ec52458f6d78c96aefbd03d386a78ba5f79fe27` | unchanged (no local path) |

**Optional-activity run of blob `a28ab5420911` (2026-10-22, `51a9090`, `RUN_ACTIVITY = True`; the current blob until `daafb45`, now history).** A run copy of
committed blob `a28ab542` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `ddbf719568c0`, SHA-256
`17e623f031e8ec3ef947abfd81012305b3ccd592fe730dbaf6eb8a426eb1e79c`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by Colab CLI 0.7.5 `colab exec -f` (scripted,
not a browser Run all) on a fresh Colab Tesla T4, started 2026-10-22T13:40:09Z, wall 148.6 s, exit 0, sequential, input (the run
copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-51a9090-activity/`. The Section 11 activity
(`n_estimators` 4 → 1) printed validation MAE 0.2935 → 0.2927, RMSE 0.3623 → 0.3638, R² 0.9948 → 0.9947 on 90 validation rows,
`max_abs_prediction_change` 0.0836, `mean_abs_prediction_change` 0.0263, and `{'canonical_outputs_unchanged': True}`. Local paths
were replaced by `<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-22-51a9090-activity/`) | Captured | Committed |
|---|---|---|
| `tabpfn_regressor_colab_51a9090_colab-cli-t4_activity.ipynb` | `c5f403e89d990eba9e74a2717430e93d19995ffb8a662f9f79e6ae3fa53fe477` | unchanged (no local path) |
| `tabpfn_regressor_colab_51a9090_colab-cli-t4_activity_exec.log` | `b16d9d6708b5536c1610e36de5d17cffd0462ba74e8833aeee2f0597b58049b0` | `90c33365fef17ff4b44b846fdcd92d0684ed6a2feebf3d25c0b68b390b89e3cd` (2 replacements) |
| `tabpfn_regressor_colab_51a9090_colab-cli-t4_activity_run_summary.json` | `6ae94de14893e71896f5a274a22dd4c9cb4c0f54cf0eded83f1667f7b79c4543` | unchanged (no local path) |
| `tabpfn_regressor_colab_51a9090_colab-cli-t4_activity_run_copy.ipynb` | `17e623f031e8ec3ef947abfd81012305b3ccd592fe730dbaf6eb8a426eb1e79c` | unchanged (no local path) |

**Hosted default-path run of blob `c6d534adcadf` (2026-10-23, `2e51561`; the current blob until `7bfee17`, now history).** Colab CLI `colab exec -f`
(a scripted run, not a browser Run all) on a fresh Colab Tesla T4, default path of `tutorials/tabpfn_regressor_colab.ipynb`
at `2e51561`, executed as committed, started 2026-10-23T09:31:27Z, wall 160.7 s, exit 0, sequential, input unchanged, 14/14
code cells, 0 errors, verdict PASS. Evidence: `docs/execution-evidence/2026-10-23-2e51561/`. Printed: TabPFN validation MAE
0.2935 / RMSE 0.3623 / R² 0.9948, test MAE 0.2705 / RMSE 0.3433 / R² 0.9942; standardised linear reference validation
0.2983 / 0.3665 / 0.9946, test 0.2661 / 0.3393 / 0.9943; "Fresh-boundary verification PASSED: predictions equivalent on 90
validation rows (rtol=1e-05, atol=1e-06)"; Section 10 reported `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256 and
SAMPLE_ARTIFACT_FITTED_SHA256)` for `7542b721…`; the activity cell printed "Optional activity skipped". The default path
supplies no refused user input, so the new `Refused:` display was not exercised by this run. Local paths were replaced by
`<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-23-2e51561/`) | Captured | Committed |
|---|---|---|
| `tabpfn_regressor_colab_2e51561_colab-cli-t4.ipynb` | `c7ee0600e7de1e181cd4cc7d7a28bc0743840b216dc4743bd068a83fb877a17d` | unchanged (no local path) |
| `tabpfn_regressor_colab_2e51561_colab-cli-t4_exec.log` | `830b34c51449301c540030c41f8bf845a4cfd6718a856b4129e803a03e3568f1` | `8767150d2b4bca7d216a8bd3ba81feead408d81917e4ff9532d3b7bd513a3af0` (2 replacements) |
| `tabpfn_regressor_colab_2e51561_colab-cli-t4_run_summary.json` | `8b22cadee7fe5f203ce9f6e7bc1e6bf17143cff4c20aba2c2dd96a6cf1ed438b` | unchanged (no local path) |

**Optional-activity run of blob `c6d534adcadf` (2026-10-23, `2e51561`, `RUN_ACTIVITY = True`; the current blob until `7bfee17`, now history).** A run copy
of committed blob `c6d534ad` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `3f44a3781230`, SHA-256
`f0bfb56caa0fd33f8a5a5e75c2e390e022e910019c5ed93e731b303fc1466721`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by Colab CLI `colab exec -f` (scripted, not a
browser Run all) on a fresh Colab Tesla T4, started 2026-10-23T09:34:22Z, wall 157.9 s, exit 0, sequential, input (the run copy)
unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence: `docs/execution-evidence/2026-10-23-2e51561-activity/`. The
canonical stages printed the same metrics as the default-path run above. The Section 11 activity (`n_estimators` 4 → 1) printed
validation MAE 0.2935 → 0.2927, RMSE 0.3623 → 0.3638, R² 0.9948 → 0.9947 on 90 validation rows, `max_abs_prediction_change`
0.0836, `mean_abs_prediction_change` 0.0263, and `{'canonical_outputs_unchanged': True}`. Local paths were replaced by
`<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-23-2e51561-activity/`) | Captured | Committed |
|---|---|---|
| `tabpfn_regressor_colab_2e51561_colab-cli-t4_activity.ipynb` | `e76f88a3a25aee7e07dbb46f86c9206ac89bc02a218221bc2dfca2f6b5f283e8` | unchanged (no local path) |
| `tabpfn_regressor_colab_2e51561_colab-cli-t4_activity_exec.log` | `b8e8b150bffcf41d4d2c0170b77dfd864b23db614540971b210c30b591163baf` | `5bd57a862bc0af84b7f99ddf400f9604ea855c948d6b7d4d0a7e8931eceaf661` (2 replacements) |
| `tabpfn_regressor_colab_2e51561_colab-cli-t4_activity_run_copy.ipynb` | `f0bfb56caa0fd33f8a5a5e75c2e390e022e910019c5ed93e731b303fc1466721` | unchanged (no local path) |
| `tabpfn_regressor_colab_2e51561_colab-cli-t4_activity_run_summary.json` | `19b5c7a085de7796079b475f7409a6bd53135d6ce1bc215ea3c48331c8e49864` | unchanged (no local path) |

### Local lock-only drives (not clean-runtime evidence)

Code cells executed in order by nbclient with an ipykernel kernel on Linux x86_64 (WSL2), CPU only (`CUDA_VISIBLE_DEVICES=-1`),
inside each notebook's own isolated hash-locked uv environment (CPython 3.12.12; torch 2.11.0, tabpfn 8.1.0, numpy 2.5.3,
pandas 2.3.2, scikit-learn 1.9.0; 63 locked packages), default form values. The E2E drive fetched the checkpoint from the
Hub at the pinned revision (`311ce18d…`); the companion drive reused that verified copy. No Colab.

| Date (UTC) | Commit / notebook blob | Path exercised | Wall | Outcome |
|---|---|---|---|---|
| 2026-10-10 | `bb3426a` / `c85822daf78d` | E2E, default synthetic sample, `N_ESTIMATORS=4` | 1065.2 s (incl. environment build) | 11/11 code cells ok, `restarted`: no. Training-mean baseline; linear reference val MAE 0.2983 / RMSE 0.3665 / R² 0.9946, test 0.2661 / 0.3393 / 0.9943; TabPFN val 0.2938 / 0.3623 / 0.9948, test 0.2713 / 0.3438 / 0.9942; MAPE omitted (targets near zero). Fresh-boundary reload PASSED, max abs prediction difference 0.0 (rtol 1e-5, atol 1e-6). Its exports are the producer of `sample-bundle-v1` |
| 2026-10-11 | working tree on `627b230` / `f2ed3aad43cc` | Artifact inference, default pinned sample (pre-seeded in the download cache; the release URL returned 404, so the download itself was not exercised), then the tamper activity | 35.8 s | 9/9 code cells ok, `restarted`: no. Sample verified before extraction (70,554 bytes, `139e87dd0300dfb3…`, reused download); predictions on the eight rows identical to the E2E run's; the tampered `model.tabpfn_fit` was refused before loading |

### History: previous (worker-driven, NOTEBOOK_SPEC 1.0) notebook pair

The rows below were recorded for the previous notebook pair, which cloned this repository and the private
`tabpfn-regressor-finetuner` worker and ran `train.run()`. The standalone notebooks share none of that code path (the
worker is not carried), so **none of these rows carry over**; they are kept as provenance for the artifact contract
the companion still consumes.
Each row is evidence for the revision named in its own `pipeline.commit`, and for nothing later. The rows below predate the
secure private-source bootstrap and the `# >>> colab-bootstrap` harness contract, so they are
**superseded**: they do not carry to the current candidate head, and no Colab record exists for any
revision. A fresh `scripts/execute_notebook_release.py` record at the candidate head, and then a
clean Colab record, are both still required before either notebook is marked release-grade.

| Date (UTC) | Notebook revision | Environment | Engine | Outcome |
|---|---|---|---|---|
| 2026-09-11 | `1e48a3469510` (clean tree); worker `34127c099ac4` | Local host `Kurt-Valcorza`, Windows-11-10.0.26200-SP0; Python 3.12.10; torch 2.11.0+cu128; tabpfn 8.1.0; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU; `MODEL_VERSION=v2`, default zero-shot | `scripts/execute_notebook_release.py --skip-bootstrap`: nbclient, one fresh kernel per notebook; `/content` → `D:/tabpfn-reg-run/content`; worker cloned from the local finetuner checkout at the pinned commit; `google.colab.files.upload()` served from explicit paths | **PASS** — E2E: `mode=zero-shot-icl`, `fineTuneEffective=false`, validation mae 0.2941 / rmse 0.3634 / r2 0.9947 vs trivial baseline; reload via serving loader reproduced the recorded validation metric within 1e-6. ARTIFACT-INFERENCE (second kernel, bundle + 8 fresh rows supplied externally): reconstructed and scored 8 rows. **Not a Google Colab run** — a Colab record is still required before either notebook is marked release-grade. |
| 2026-09-11 | `b9ea9c69a728` (clean tree); worker `34127c099ac4` | Local host `Kurt-Valcorza`, Windows-11-10.0.26200-SP0; Python 3.12.10; torch 2.11.0+cu128; tabpfn 8.1.0; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU; `MODEL_VERSION=v2`, FINE_TUNE=True | `scripts/execute_notebook_release.py --skip-bootstrap --set FINE_TUNE=True`: nbclient, one fresh kernel per notebook; `/content` → `D:/tabpfn-reg-ft/content`; worker cloned from the local finetuner checkout at the pinned commit; `google.colab.files.upload()` served from explicit paths | **PASS** — E2E: `mode=fine-tune`, `fineTuneEffective=true`, validation mae 0.2978 / rmse 0.3661 / r2 0.9946 vs trivial baseline; reload via serving loader reproduced the recorded validation metric within 1e-6. ARTIFACT-INFERENCE (second kernel, bundle + 8 fresh rows supplied externally): reconstructed and scored 8 rows. **Not a Google Colab run** — a Colab record is still required before either notebook is marked release-grade. |

### RUN7 deviation (accepted)

The E2E notebook declares profile `E2E` with in-context conditioning (`fit` registers the support rows; no gradient
update) as its adaptation. Not running the production pipeline's gradient fine-tuning (Prior Labs'
`FinetunedTabPFNRegressor`, default on, large-GPU) is a RUN7 deviation, recorded in the notebook and its evaluation
report (`adaptation_deviation`) and accepted by the maintainer on 2026-10-10 (review finding TPR-m1).

## Current status

**Current status (2026-10-24): Candidate.** The companion is retired and folded into the E2E notebook's Section 10. The current E2E notebook blob is `d421385e` (regenerated from code commit `3a6a95c` on `fix/serve-safetensors`: the model is served from `tabpfn-v3-regressor-v3_default.safetensors`, converted once from the pinned checkpoint with `weights_only=True` and digest-pinned; it is stacked on `7bfee17`, whose guard refuses a pickle checkpoint load when torch is older than 2.6 or `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is set, because tabpfn 8.1.0 loads `.ckpt` files with `weights_only=None`). The intermediate blob `62d09b0e` (`7bfee17`) was never run. It has no recorded hosted run until the orchestrator records one (REL14). The `2e51561` default-path and optional-activity records (under *Recorded executions*) cover the previous blob `c6d534ad` and are history, as are the `51a9090` records of blob `a28ab542` and the earlier records. The notebook stays **Candidate** until a hosted default-path run and an optional-activity run (the owner's 2026-10-22 ruling) of the current blob and these two are recorded for it:

- **REL12:** a BYOD run in which one representative input is accepted through the downstream stages and one incompatible input is refused;
- **REL2:** a browser run of the interactive journeys (BYOD upload dialog, optional-activity toggle, own-artifact ZIP); the scripted CLI runs count for the default path and the activity only.

Promotion is an integrator decision; the builder does not change the status. The rest of this section describes the pre-migration pair and is kept as history (REL14).

**2026-10-10 — one-pass hosted runs of both pre-migration blobs are recorded** (first two rows of the clean-runtime table): `tabpfn_regressor_colab.ipynb` (`c85822daf78d`) and `tabpfn_regressor_artifact_inference_colab.ipynb` (`f2ed3aad43cc`) at `e2d5fab`, each by Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (not a browser Run all), no restart, 0 error outputs. Notes for the reviewer: (a) the E2E notebook's carried revision label reads `7a59c5b` (generation-time HEAD, not `e2d5fab`) — a provenance follow-up; (b) the tamper activity was not exercised (`RUN_ACTIVITY` off by default), only the `SCHEMA_MISMATCH` refusal ran; (c) the `sample-bundle-v1` release download was exercised for real (fresh fetch, SHA-256 verified before extraction); (d) the E2E notebook text still calls the RUN7 deviation "proposed, pending approval", although it was accepted on 2026-10-10 (see *RUN7 deviation (accepted)* above) — a follow-up for the next regeneration. Both notebooks stay **Candidate**; remaining gates: reviewer confirmation of these runs, the BYOD journeys, and the optional activities (including the tamper refusal on a hosted runtime).

Before 2026-10-10: no clean-runtime execution of the current notebook blobs (`build_notebook.py/3.0-tabular`, isolated locked
environment) had been recorded. The only hosted run is the 2026-09-14 Kaggle CPU row above,
for an earlier E2E blob with an in-kernel install; it staged the checkpoint but recorded no restart status, versions or
metrics, so it does not carry to the current blobs. Static validation (`tools/validate_release_assets.py`), the generator parity checks, a `compile()` sweep
over every code cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is
necessary but not sufficient. The registry status remains **Candidate** until a reviewer confirms recorded runs against
the notebook blobs under review and an integrator promotes them; promotion is not performed by the builder. Facts a
reviewer should weigh: the pinned checkpoint `tabpfn-v3-regressor-v3_default.ckpt` was staged by the Kaggle run above (the manifest's digest
comes from the Hub API); the module's `tabpfn` calls (`TabPFNRegressor(model_path=...)`, `save_fitted_tabpfn_model`,
`load_fitted_tabpfn_model`) run against the real package only in the CI `integration` job (CPU fit, save, relocate,
load, predict) and in the 2026-10-10/11 local lock-only drives above, which are not clean-runtime evidence; the
standalone carrier was validated statically and by a CPU carrier probe (module cells + identity assertion, no fetch).

## Local-path redaction of earlier evidence (2026-10-22)

The files below were committed with local absolute paths in them. Each local run-directory prefix was replaced with `<local-run-dir>/`, the convention the evidence committed from 2026-10-22 on already uses; the local run tree named in `PHASE2_ACCEPTANCE_REPORT.md` became `<local-run-dir>`. No other byte changed and line endings are unchanged. Git history still holds the original files. "Before" is the SHA-256 of the file as committed on `main` at `3fbf90c`.

| File | Replacements | SHA-256 before | SHA-256 after |
| --- | --- | --- | --- |
| `PHASE2_ACCEPTANCE_REPORT.md` | 1 | `8279f806265d9265a9014c82953dca94f8163a79f5dd74394b9dec3421481f46` | `f9efc2182eefda51077210f97d432afefa725b2ebf35cc626aad9206920fcee3` |
| `docs/execution-evidence/2026-10-10/tabpfn_regressor_artifact_inference_colab_e2d5fab_colab-cli-t4_exec.log` | 2 | `20b946c7af7f764daf570aee9221d0e5b91dd114e0a25dfa969b38deb8827468` | `dc29a115c2a9bc3e34617ad10a2d0499ce27b6143d1311d0e0737fb8f76f43ed` |
| `docs/execution-evidence/2026-10-10/tabpfn_regressor_colab_e2d5fab_colab-cli-t4_exec.log` | 2 | `18c70aa252bde3b3a2501f0a5fd2d7422818b0c17619bdb2cb01ec69ac3e5b5d` | `1fcc4e05d7778185dc62b57283924030a56fb41f1dbdd0915abc75e5e883e507` |

The Recorded executions table above quotes the pre-redaction hash of 2 of these logs as the hash of the committed file. Those quotes are kept as recorded, and the post-redaction hash is noted beside each one.

# Release verification

`tutorials/tabpfn_regressor_colab.ipynb` (`E2E`) and `tutorials/tabpfn_regressor_artifact_inference_colab.ipynb` (`ARTIFACT-INFERENCE`) are **release
candidates** until the exact notebook revisions have executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2. This file is the durable release-gate record for both
notebooks; the previous notebook pair's execution records (worker-driven, NOTEBOOK_SPEC 1.0) are kept below as history.

## Automatic coverage (static, every pull request)

CI (`.github/workflows/ci.yml`, job `test`) runs `tools/validate_release_assets.py`, which checks, for each notebook:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the spec version (`2.2`) and
  the standalone carrier; `metadata.dimer` declares the profile (`E2E`, `ARTIFACT-INFERENCE`), mode `GUIDED`,
  `standalone: true` and `generated_from` (repository, generating commit, package paths and SHA-256, carried-file
  digests, generator `build_notebook.py/3.0-tabular`);
- the standalone carrier and isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose carried
  files equal the repository files (`src/tabpfn_regressor_pipeline/{__init__,pipeline}.py`, the stage runner,
  `tutorials/requirements-colab.lock.txt`, the 4-file snapshot manifest, `NOTICE`, and — once it exists — the companion's
  pinned sample bundle) with matching digests; the lock pins every `pyproject.toml` runtime pin with hashes; a pinned
  `uv` builds a managed-CPython environment with `--require-hashes`, reused per lock digest; no in-kernel install and no
  restart instruction; the four Infrastructure cells are titled and collapsed; every learner cell runs a stage; the
  notebook byte-identical to `tools/build_notebook.py` output;
- the stage-runner markers — E2E: validation before any split with coded findings, a non-numeric target refused as `TARGET_NOT_NUMERIC`, identifiers (`DROP_COLUMNS`) kept out of the features, numeric
  columns with stray strings refused, the training-mean baseline, the MAPE floor and the linear-regression reference, the accepted RUN7 deviation recorded, `save_artifact` + recorded
  feature kinds, the fresh-process reload comparing predictions on every validation row (`rtol=1e-5`, `atol=1e-6`);
  companion: trusted ZIP / fitted digests, the member allowlist before extraction, `validate_artifact_bundle` with the
  checkpoint bound to the pinned one, `from_artifact`, `ID_COLUMNS`, the numeric-type check, a `not-measurable` report,
  and no self-production; the form-parameter defaults; no quality `assert`; the forbidden patterns;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the `## Artifacts and provenance` section.

CI also runs `ruff check .`, both generator `--check` runs, and the offline tests (`tests/`, no tabpfn, no torch, no
weights, no Parquet engine), including `test_notebook_review_fixes.py`, which execs the notebooks' own cells with
stand-ins and runs the model-free stages. These are source and unit checks, **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or T4 runtime; any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorials are written for; a clean one-pass **Run all** here is promotion evidence |
| Kaggle kernel | Kaggle CPU or GPU image, any Python | Reproducible clean-room executor of the same class; the notebook runs verbatim (no checkout needed). For the companion set `ARTIFACT_ZIP_PATH`, `EXPECTED_ZIP_SHA256`, `EXPECTED_FITTED_SHA256`, `NEW_DATA_PATH` and `ID_COLUMNS` (empty for the synthetic rows) to the E2E run's exports |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a fresh runtime with **no repository checkout** and a clean model cache;
3. run it with one **Run all** and no restart, form parameters at their defaults (`USE_BYOD = False`,
   `DROP_COLUMNS = []`, `N_ESTIMATORS = 4`, `SEED = 42`, `USE_BYOD_ROWS = False`,
   `RUN_ACTIVITY = False`), then re-run the export cell once;
4. verify: the carried-file verification and the isolated environment's versions (CPython 3.12.12, `torch` 2.11.0,
   `tabpfn` 8.1.0); the 4-file snapshot staged and verified; 420 / 90 / 90 rows and 3 features (`x1`, `x2`, `category`);
   the input manifest with the explained `TARGET_OUT_OF_TRAINING_RANGE` finding and the `renamed-target-probe` finding;
   TabPFN's validation and test MAE / RMSE / R² with `mape_omitted` and its reason; the training-mean baseline (validation
   RMSE 5.01, R² ≈ 0) and the linear regression (validation RMSE 0.366, R² 0.9946; test 0.339, 0.9943); the evaluation
   report `sample-sanity` with the adaptation deviation; the bundle digests printed; the reload with
   `maxAbsPredictionDifference` within tolerance; `outputs/tabpfn_regressor_new_rows.csv` and the predictions; the result
   JSON with source, model identity, licence and runtime;
5. in a **second** clean runtime run the exact companion revision with `ARTIFACT_ZIP_PATH`, both digests, `NEW_DATA_PATH`
   and `ID_COLUMNS` set to the E2E exports, and verify the digest checks, the member allowlist, the checkpoint binding,
   the rebuild, the input manifest, the `not-measurable` report and the predictions; with all fields empty it must stop
   only after downloading the pinned `sample-bundle-v1` release asset and verifying its size, SHA-256 and four members
   before extraction (with the release unpublished it stops at that download, naming the troubleshooting row);
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
| 2026-09-14 | `1130f16` / `32ef6f4a998a` | Kaggle CPU (`kurtvalcorza/dimer-nb2-tabpfn-regressor` v1) | Default sample path | 207.5 s | **PASSED** — 10/10 ok code cells executed cleanly, 10 files, 233 MB staged. Earlier blob (`build_notebook.py/2`, in-kernel install): restart status, package versions and metrics were not recorded; not evidence for the current blob |
| | | | Artifact inference | | pending — no run recorded; its default path needs the `sample-bundle-v1` release asset, which is not yet published |

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

No clean-runtime execution of the current notebook blobs (`build_notebook.py/3.0-tabular`, isolated locked
environment) has been recorded; the runs are **pending**, and the companion's default path additionally needs the
`sample-bundle-v1` release to be published. The only hosted run is the 2026-09-14 Kaggle CPU row above,
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
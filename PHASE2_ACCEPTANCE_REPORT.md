# Phase 2 — Local GPU Acceptance Report (Regressor)

**Date:** 2026-08-19
**Runtime host:** Windows 11 + WSL2 `nvidia-docker` distro, Docker 29.6.2
**GPU:** RTX 5070 Ti Laptop, 12 GB VRAM, compute cap 12.0 (Blackwell / sm_120), driver 610.88
**Stack:** `tabpfn==8.1.0`, `torch 2.11.0+cu128` (CUDA 12.8), base image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`
**Components:** validator `1c39f31` → **`78b3159`**; finetuner `4e1333f` → **`ae54e0d`** (D1/D2 GPU fixes) → `1c99dd4` (+ version-aware limits) → **`b8721d9`** (+ integration CI). `COMPONENTS.json` pins the branch tips `78b3159` / `b8721d9`.

> **Pin status (pre-merge).** Both component changes are in open PRs:
> finetuner [#2](https://github.com/kurtvalcorza/tabpfn-regressor-finetuner/pull/2)
> (tip `b8721d9`) and validator
> [#2](https://github.com/kurtvalcorza/tabpfn-regressor-dataset-validator/pull/2)
> (tip `78b3159`). `COMPONENTS.json` pins those branch tips. If either PR is
> **squash-merged**, its branch SHA will not be on `main`; bump the corresponding
> pin to the squash-merge commit before merging this pipeline PR.
>
> This report covers the **GPU acceptance** gates (D1/D2, commit `ae54e0d`). The
> finetuner/validator PRs also carry **version-aware capacity limits** (review
> finding 3); see the note at the end.

## Verdict

The CPU→GPU→artifact→clean-reload chain is **proven end-to-end for both `v2` and `v3`**. Two real, previously-unproven defects (D1, D2) were found during the Phase 2 run, **reproduced on the unpatched build**, fixed in finetuner PR #2 (`ae54e0d`), and **re-verified from the patched build** (image built from the patched repo; v2 + v3 fine-tune and round-trip with the stock entrypoint). `v3` (TabPFN-3, gated non-commercial) used the license accepted via the **HuggingFace gate** on `Prior-Labs/tabpfn_3` plus a manual weight download (`DIMER_TABPFN_MODEL_PATH`). DIMER serving (gate 8) is a separate deferred gate, not a blocker to closing Phase 2 GPU acceptance.

## Gate results (`DEPLOYMENT.md`)

| # | Gate | Result | Evidence |
|---|------|--------|----------|
| 1 | Validator (unit + real image) | PASS | unit 6/6; real image 18/18 checks green on synthetic ZIP |
| 2 | Component unit tests | PASS | validator 6, finetuner 7, pipeline 3 (16/16) |
| 3 | **Finetuner CUDA image builds** | **FAIL → PASS (fixed in `ae54e0d`)** | bare `pip install` died on PEP 668; build exit 1 → 0 after D1 |
| 4 | TabPFN checkpoint licensing | PASS | v2 Apache auto-download; v3 accepted via HF gate |
| 5 | Approved model-distribution path | PASS | v2 auto-download; v3 manual HF download + `DIMER_TABPFN_MODEL_PATH` |
| 6 | **Real GPU fine-tuning smoke test** | **PASS (v2 + v3)** | exit 0, artifacts written, metrics below |
| 7 | **Artifact reload in clean runtime** | **PASS (v2 + v3)** | CUDA reload exact; CPU within tol; predictions span ±, no clipping |
| 8 | DIMER serving E2E inference | NOT RUN | needs DIMER PoC wiring; deferred |

Validator negatives (all fail-closed, exit ≠ 0): `dup_train`, `nonnumeric_target`, `too_few_rows`, `constant_target`, `traversal`, `zip_bomb`.

## v2 fine-tune smoke test (gate 6)

Config: `fine_tune=true, model_version=v2, epochs=2, n_estimators 1/1/2, n_finetune_ctx_plus_query_samples=4000, seed=0`.
Dataset: synthetic 600-row → 420 train / 90 val / 90 test, 3 feature columns, target range [-13.47, 13.08] (spans negative/positive).

- mode `fine-tune`, device `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- **validation:** mae 0.2864, rmse 0.3552, r² 0.9945, MAPE 18.5%
- **test:** mae 0.3195, rmse 0.4693, r² 0.9908, prediction range [-15.55, 13.24]
- artifacts: `model.tabpfn_fit` (96,699 B, sha256 `cfbd7625…7be1`), `model.ckpt` (44,406,183 B, sha256 `db203ac4…d906`), `artifact_manifest.json`
- **peak GPU memory (host-total): 3,036 MiB**
- wall-clock: 13 s

## v2 artifact round-trip (gate 7)

Reloaded `model.tabpfn_fit` in a **fresh container** via `serving/load_artifact.py` (`load_dimer_tabpfn_regressor_artifact`, which rewrites `init_params.json:model_path` to the companion `model.ckpt`), re-derived the test metrics, compared to training `result.json`:

- **CUDA reload:** all metric diffs `0.0` (exact reproduction) — PASS at tol 1e-6
- **CPU reload:** diffs ≤ 2.6e-3 (device float noise) — PASS at tol 1e-2
- **No clipping:** reloaded predictions span [-15.55, 13.24] (negative→positive), satisfying `DEPLOYMENT.md` steps 8–9

## Defects found and fixed (previously unproven; unit CI could not catch either)

Both were found during the Phase 2 run, reproduced on the unpatched build, **fixed in finetuner PR #2 (`ae54e0d`), and re-verified from the patched build** — the image was rebuilt straight from the patched repo and v2 + v3 fine-tune and round-trip pass with the stock entrypoint (no local patches).

### D1 — Finetuner Docker image did not build (gate 3)
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` ships a **PEP 668 externally-managed** system Python, so the original `Dockerfile` `RUN pip install --no-cache-dir -r requirements.txt` failed with `error: externally-managed-environment`. Reproduced: build exit 1.

**Fix (finetuner `Dockerfile`):**
```dockerfile
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
```
Safe in a single-purpose container.

### D2 — Fine-tuned artifact could not be saved on GPU (blocked gates 6/7/8)
TabPFN 8.1.0's `save_fitted_tabpfn_model` str-coerces `torch.dtype` **but no other non-JSON init param**, then `json.dump(params)`. The fine-tuned inference regressor's `get_params()` returns a non-serializable `device` and a `RegressorModelSpecs` `model_path`, so the save raised `TypeError: Object of type device is not JSON serializable` and **no artifact was written**. Reproduced on the D1-only build (fine-tune succeeded, save failed, only `checkpoints/` written).

**Fix (finetuner `train.py` `save_artifacts`, before `save_fitted_tabpfn_model`):**
```python
for _param, _value in model.get_params(deep=False).items():
    try:
        json.dumps(_value)
    except (TypeError, ValueError):
        setattr(model, _param, str(_value))
```
Safe because evaluation runs before saving, the foundation weights are persisted separately via `save_tabpfn_model(model.ckpt)`, and `serving/load_artifact.py` overwrites `model_path` (and load supplies `device`) at load time — confirmed by the passing round-trip above, run against the patched build.

## v3 (TabPFN-3) — gated, resolved via HuggingFace

Same config as v2 but `model_version=v3`, base weights supplied by `DIMER_TABPFN_MODEL_PATH`.

- mode `fine-tune`, device `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- **validation:** mae 0.2739, rmse 0.3403, r² 0.9949, MAPE 18.2%
- **test:** mae 0.2685, rmse 0.3468, r² 0.9950, prediction range [-18.26, 13.54]
- artifacts: `model.tabpfn_fit` (75,221 B, sha256 `50e5a332…ab9b`), `model.ckpt` (233,304,651 B, sha256 `8083380b…7687`)
- base checkpoint `Prior-Labs/tabpfn_3 / tabpfn-v3-regressor-v3_default.ckpt` (233,289,807 B, sha256 `311ce18d…7301`)
- **peak GPU memory: 4,089 MiB**, wall-clock 14 s
- round-trip: CUDA diffs all `0.0`; CPU diffs ≤ 4.7e-3 — both PASS; predictions span [-18.26, 13.54]

The v3 weights are **non-commercial**; the license was accepted through the HuggingFace gate on `Prior-Labs/tabpfn_3` and the weights loaded via `DIMER_TABPFN_MODEL_PATH`, which does not invoke the portal license check. Local non-commercial evaluation only.

## What remains

D1/D2 are fixed in finetuner PR #2 (D1/D2 at `ae54e0d`, branch tip `b8721d9`); this repo's `COMPONENTS.json` re-pins to the finetuner (`b8721d9`) and validator (`78b3159`) branch tips (bump to the merge SHAs if squash-merged — see the pin-status note above). Outstanding:

1. **Gate 8 (DIMER serving E2E)** — wire the artifact into the DIMER PoC serving layer and issue a real inference request. Separate deferred gate, not a Phase 2 GPU-acceptance blocker.
2. **Resource profile** — measured peak 3.0 GB (v2) / 4.1 GB (v3) on this smoke dataset; keep `DEPLOYMENT.md` resource guidance until measured on representative data.
3. **Minor** — `provenance.model.baseModelSha256` is `null` under `DIMER_TABPFN_MODEL_PATH` because `model_provenance` reads the estimator's resolved `RegressorModelSpecs` object rather than the configured checkpoint path; it could hash `config.model_path` directly.
4. **v3 licensing for production** — the v3 weights remain **non-commercial**; DIMER production/external enablement requires a commercial license from Prior Labs regardless of this local eval.

## Version-aware capacity limits (review finding 3)

Carried by the pinned component PRs (separate from the GPU-acceptance gates above, but bundled in the same commits). The validator and trainer now read the selected `model_version` (pipeline passthrough) and enforce that generation's capacity instead of generic defaults — hard-capping features and subsampling rows to the version limit; unknown/`default` falls back to the strictest tier. Limits verified against tabpfn 8.1.0:

| version | max_samples | max_features | max_classes | source |
|---|---|---|---|---|
| v2 | 10,000 | 500 | 10 | tabpfn source `_get_v2_config` |
| v2.5 | 50,000 | 2,000 | 10 | tabpfn source `_get_v2_5_config` |
| v2.6 | 100,000 | 2,000 | 10 | model card (checkpoint not bundled) |
| v3 | 1,000,000 | 2,000 | 160 | v3 checkpoint `inference_config` |

`max_classes` is unused for regression; it is retained so the table matches the classifier pipeline. Component unit tests: validator 9/9, finetuner 11/11.

## Reproducibility

Harness scripts and the isolated run tree live under the session scratchpad and `/home/kurt/phase2/reg` in the `nvidia-docker` distro (dataset, negatives, `output-v2`, `output-v3`, `results`, `model-cache`). v3 base obtained under the owner's accepted HuggingFace gate for non-commercial use.

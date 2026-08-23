# Model Card — TabPFN Regressor Pipeline

## Model

This pipeline uses Prior Labs' TabPFN package, pinned to `tabpfn==8.1.0`, with model generation `v3` selected by default. The deployable trainer supports both the base `TabPFNRegressor` in in-context mode and `FinetunedTabPFNRegressor` for task-specific gradient updates.

## Intended use

General tabular regression where one row represents one example and the target is a finite numeric value. Features may be numeric or categorical, subject to TabPFN's supported preprocessing behavior and the pipeline's configured resource limits.

This pipeline does **not** assume a non-negative response variable. Negative targets and predictions are valid.

## Fine-tuning

Fine-tuning needs a CUDA GPU. CPU is the default deployment: when `fine_tune=true` and no usable CUDA device is present, the pipeline falls back to zero-shot ICL and records `metrics.fineTuneSkippedReason` (`metrics.fineTuneEffective=false`) rather than failing.

The upstream TabPFN 8.1.0 fine-tuning wrapper currently uses MSE for regression early-stopping/evaluation inside the training loop. DIMER independently reports MAE, RMSE, R², and MAPE where defined.

## Artifacts and provenance

Successful runs save (under `DIMER_OUTPUT_DIR/artifacts/`) a fitted estimator state (`model.tabpfn_fit`) plus its companion foundation checkpoint (`model.ckpt`, which the fitted archive omits). Both are SHA-256 hashed in `artifact_manifest.json`. Dataset bytes are also hashed for run provenance. `result.json` (alongside `evaluation/`, `logs/`, `progress/`) declares `artifacts.modelArtifact` for `export-to-repository`; see `CONTRACT.md`.

For production reproducibility, provide a platform-managed base checkpoint through `DIMER_TABPFN_MODEL_PATH`; this allows the run to record a cryptographic hash of the exact starting checkpoint rather than relying on a mutable/default cache resolution.

## Limitations

- Performance is dataset-dependent; no benchmark ranking guarantees suitability for a specific task.
- Random validation is inappropriate for leakage-sensitive temporal/grouped data; provide explicit splits in those cases.
- MAPE is unstable/undefined around zero and is therefore reported only over non-zero target rows.
- CPU CI does not verify GPU fine-tuning or production serving.
- Package code and model weights may have different licensing or usage terms. Review the exact selected model terms before hosted production deployment.

## Production acceptance

Do not production-enable solely from unit CI. Require an end-to-end run covering: validated dataset → CUDA fine-tune → persisted fitted-state/checkpoint pair → clean-runtime reload through `serving/load_artifact.py` → DIMER inference request → metric/result inspection.

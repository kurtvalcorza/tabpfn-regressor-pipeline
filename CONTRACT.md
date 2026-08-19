# DIMER TabPFN tabular-regression pipeline contract

Authoritative contract for the TabPFN regression trio (validator + finetuner +
this umbrella). Complements the existing docs rather than repeating them:
dataset shape lives in [`TABULAR_REGRESSION_DATASET_SPEC.md`](TABULAR_REGRESSION_DATASET_SPEC.md),
the release procedure in [`DEPLOYMENT.md`](DEPLOYMENT.md), and the last hardware
run in [`PHASE2_ACCEPTANCE_REPORT.md`](PHASE2_ACCEPTANCE_REPORT.md).

## 1. Effective task type and the portal fallback

The effective task type is **`tabular_regression`**. DIMER does not yet expose a
native tabular-regression task, so the portal persists the pipeline as
`Custom / Other`. The runtime does **not** rely on that portal value:

- Both container images set `DIMER_TASK_TYPE=tabular_regression`.
- Result payloads report `metadata.taskType`, defaulting to `tabular_regression`
  when `DIMER_PIPELINE_METADATA_JSON.taskType` is absent.

When DIMER adds a native tabular task type, switch the portal type and drop the
fallback; no runtime code change is required because the images are already
authoritative.

## 2. `dimer-pipeline.json` is the authoritative parameter API

Every key declared in the finetuner's `dimer-pipeline.json` maps to concrete
runtime behavior in `train.py` (`load_config` → `Config` → `prepare_frames` /
`fit_model`). There are **no silently-ignored controls**, and **`model_id` is not
a hyperparameter** — the base model is selected by DIMER's Base Model, surfaced to
the runtime as `model_version` plus an optional pinned checkpoint (§3).

| Manifest key | Runtime effect |
|---|---|
| `target_column`, `drop_columns` | column selection in `prepare_frames` |
| `validation_split` | holdout fraction when `val.csv` is absent |
| `max_train_rows` | deterministic row cap (further capped to the version limit, §4) |
| `fine_tune` | `FinetunedTabPFNRegressor` (CUDA) vs zero-shot ICL |
| `model_version` | TabPFN generation → per-version limits + checkpoint selection |
| `epochs`, `learning_rate`, `weight_decay`, `time_limit_seconds`, `early_stopping_patience` | AdamW fine-tuning schedule |
| `n_finetune_ctx_plus_query_samples`, `n_estimators_finetune/validation/final_inference` | ensembling + context sizing |
| `prediction_batch_rows` | eval chunk size (bounds VRAM/RAM) |
| `seed` | sampling / holdout / fine-tuning RNG |

## 3. Base-model handoff and provenance

DIMER's selected Base Model reaches the finetuner two ways:

1. **Generation** via `model_version` (`v3` default, or `v2.6/v2.5/v2/default`).
   Only **v2** carries Prior Labs' Apache-derived license and auto-downloads
   freely; **v2.5, v2.6, and v3** weights are **non-commercial**.
2. **Pinned checkpoint** via `DIMER_TABPFN_MODEL_PATH` — an approved local
   checkpoint mounted into the container, loaded directly and hashed for
   provenance. This is the required path for the non-commercial generations
   (`v2.5`/`v2.6`/`v3`), whose weights must be supplied as an accepted checkpoint.

Provenance is verifiable, not inferred from cache ordering: `provenance.model`
records `tabpfnVersion`, `modelVersionRequested`, `modelPathRequested`,
`baseModelSha256` (SHA-256 of the pinned checkpoint when set) + `baseModelPathHashed`.

## 4. TabPFN-specific limits (not copied from Mitra/TabICL)

Enforced by both validator and finetuner from an identical table, verified against
tabpfn 8.1.0 (v2/v2.5 source `InferenceConfig`, v3 checkpoint `inference_config`;
v2.6 from the model card):

| version | max_samples | max_features | max_classes* |
|---|---|---|---|
| v2 | 10,000 | 500 | 10 |
| v2.5 | 50,000 | 2,000 | 10 |
| v2.6 | 100,000 | 2,000 | 10 |
| v3 | 1,000,000 | 2,000 | 160 |

\* unused for regression; retained for table-parity with the classifier pipeline.
Features are a hard reject; rows are deterministically subsampled to the version
cap. The validator reads the selected version via `DIMER_HYPERPARAMETERS_JSON`
passthrough; unknown/`default` → strictest (v2). VRAM is a hard CUDA requirement
for `fine_tune=true`; measured smoke peaks were 3.0 GB (v2) / 4.1 GB (v3) — see
`DEPLOYMENT.md` for the production profile.

## 5. Common result / provenance schema

Shared across the tabular-FM pipelines (validator emits `checks[]` + `metadata`;
finetuner emits the training result):

```jsonc
{
  "successful": true,
  "metrics": { "mode": "fine-tune|zero-shot-icl", "deviceName": "...",
               "validation": { "mae": .., "rmse": .., "r2": .., "predictionMin/Max": .. },
               "test": { .. | null } },
  "artifacts": { "fittedEstimator": "model.tabpfn_fit", "foundationCheckpoint": "model.ckpt",
                 "manifest": "artifact_manifest.json", "sha256": {..},
                 "reloadCheck": { "passed": true, "device": "cpu", "rows": N } },
  "provenance": { "model": {..}, "dataset": {"sha256": ".."}, "seed": N },
  "metadata": { "template": "..", "taskType": "tabular_regression" }
}
```

`artifacts.reloadCheck` is written only after the finetuner reloads the saved
artifact in-process and predicts — success is never reported for an artifact that
cannot be reloaded.

## 6. Regression smoke matrix

The release smoke set must cover: mixed numeric/categorical features; nullable
targets; **negative targets and predictions** (no clipping); explicit `val.csv`;
deterministic auto-validation (no `val.csv`); optional held-out `test.csv` scored
only after training; duplicate split-name rejection; and at least one dataset near
the selected version's limits. The integration CI job exercises the real-TabPFN
fit → save → relocate → load → predict path on every PR.

## 7. Release gate

Per `DEPLOYMENT.md`: production enablement requires a successful DIMER smoke
validation + training run, exact model-generation/checkpoint provenance, a verified
saved-artifact reload/inference (now enforced in-runtime via `reloadCheck` and in
CI via the integration tier), and proof that manifest controls affect runtime
behavior. The non-commercial generations (`v2.5`/`v2.6`/`v3`) additionally require
a commercial license for any non-evaluation use.

## 8. Cross-repo synchronization

`COMPONENTS.json` pins immutable 40-char validator/finetuner commit SHAs; the
pipeline CI manifest check enforces the format and `kurtvalcorza/` ownership. The
`MODEL_VERSION_LIMITS` table and the artifact/result schema are duplicated
verbatim across validator and finetuner (DIMER isolation forbids importing shared
source); any change to the limits or schema must update both copies in the same
change set and re-pin here.

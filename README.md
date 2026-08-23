# TabPFN Regressor — DIMER Pipeline

A DIMER pipeline that fine-tunes [TabPFN](https://huggingface.co/Prior-Labs/tabpfn_3), a
pretrained tabular foundation model from Prior Labs, on your own tabular-regression dataset.
You supply a table of rows with one finite numeric target column. The pipeline validates the
table, fine-tunes TabPFN (or runs it as an in-context regressor), and produces a saved model
artifact plus holdout error metrics.

TabPFN's licensing depends on the model generation. The default generation, **v3**, has
**non-commercial** weights that are **Hugging Face-gated**; only the **v2** generation carries
Prior Labs' Apache-derived licence and may be used commercially. A served pipeline is also bound
by the licence of the data it was trained on. See
[The model: TabPFN](#the-model-tabpfn) and [Data licence governs the served model](#data-licence-governs-the-served-model).

For platform-administrator setup and operations — resource profiles, weights delivery, network
egress, base-model handoff, and the acceptance test — see [DEPLOYMENT.md](DEPLOYMENT.md). Model
provenance and adaptation details are in [MODEL_CARD.md](MODEL_CARD.md); the full runtime contract
is in [CONTRACT.md](CONTRACT.md).

---

## The model: TabPFN

TabPFN is a tabular foundation model built by [Prior Labs](https://priorlabs.ai). Like Mitra and
TabICL, it is an **in-context learner**: it reads a table of labelled examples as context and
predicts on new rows, without a per-dataset training loop being required. This pipeline pins
`tabpfn==8.1.0` and defaults to model generation **v3**. For regression it uses
`TabPFNRegressor` in in-context mode and `FinetunedTabPFNRegressor` for task-specific gradient
updates.

The pipeline does not clip predictions or assume a non-negative target domain — negative,
zero, and positive targets are all valid. See [MODEL_CARD.md](MODEL_CARD.md) for provenance,
adaptation mode, and how to supply the weights to DIMER.

### Generations and capacity limits

The DIMER **Base Model** field selects the TabPFN generation, surfaced to the runtime as
`model_version`. Each generation has hard architectural capacity limits, enforced identically by
the validator and the fine-tuner (verified against `tabpfn==8.1.0`: v2/v2.5 from the source
`InferenceConfig`, v3 from the checkpoint `inference_config`, v2.6 from the model card):

| Generation | max_samples | max_features | max_classes\* | Licence |
|---|---|---|---|---|
| v2 | 10,000 | 500 | 10 | Prior Labs Apache-derived (commercial OK) |
| v2.5 | 50,000 | 2,000 | 10 | non-commercial |
| v2.6 | 100,000 | 2,000 | 10 | non-commercial |
| v3 (default) | 1,000,000 | 2,000 | 160 | non-commercial, HF-gated (`Prior-Labs/tabpfn_3`) |

\* `max_classes` is **unused for regression**; it is retained so this table matches the
classifier pipeline. Features are a **hard reject** above the ceiling; rows are
**deterministically subsampled** to the version cap by the trainer. The validator reads the
selected version through the `DIMER_HYPERPARAMETERS_JSON` passthrough (fallback
`DIMER_PIPELINE_METADATA_JSON`); an unknown or `default` value falls back to the **strictest
(v2)** tier.

### Applicability

TabPFN is an in-context model, so its accuracy depends on whether the target carries signal in
the supplied table. Treat any benchmark ranking as evidence of strong performance where signal
exists, not as a guarantee on a specific dataset. Random validation is inappropriate for
leakage-sensitive temporal, grouped, panel, or spatial data — provide explicit splits in those
cases (see [The dataset](#the-dataset)).

### Fine-tuning and zero-shot

TabPFN supports two modes, both exposed as fine-tuning fields:

- **Fine-tune** (`fine_tune=true`, the default) adapts the pretrained weights to the uploaded
  table through `FinetunedTabPFNRegressor`. This needs a CUDA GPU.
- **Zero-shot / ICL** (`fine_tune=false`) runs `TabPFNRegressor` as an in-context learner with
  no gradient update. It is CPU-safe.

**CPU is the default deployment; GPU is opt-in and off by default.** When `fine_tune=true` and no
usable CUDA device is present, the run **auto-falls back to zero-shot ICL** — it records
`metrics.fineTuneSkippedReason` and `metrics.fineTuneEffective=false` and **does not fail**.
`DIMER_TRAIN_DEVICE` is honored; a bare device index (`"0"`) is normalized to `cuda:0`. Each run
records the resolved device and effective mode in `result.json`.

---

## When to use this pipeline

Use this pipeline for tabular regression: predicting a finite numeric value from a row of
features. Demand quantities, prices, durations, scores, and any row-per-record numeric
prediction fit here. For a **categorical** target, use the sibling
[TabPFN classifier pipeline](https://github.com/kurtvalcorza/tabpfn-classifier-pipeline)
instead. Do not use this pipeline for images; for vision tasks use the Image Classification,
Object Detection, or Segmentation pipelines.

---

## Repositories

The pipeline is two deployable containers, one repository each, plus this umbrella repository for
the contract and docs. Each `Dockerfile` sits at its repository root. This repository is the
**integration / specification layer** and pins immutable reviewed component revisions in
[`COMPONENTS.json`](COMPONENTS.json); it does not duplicate deployable container source.

| Component | Repository | Runs on |
|---|---|---|
| Validator | [`tabpfn-regressor-dataset-validator`](https://github.com/kurtvalcorza/tabpfn-regressor-dataset-validator) | CPU |
| Fine-tuner | [`tabpfn-regressor-finetuner`](https://github.com/kurtvalcorza/tabpfn-regressor-finetuner) | GPU (CPU falls back to zero-shot) |
| Umbrella (this repo) | `tabpfn-regressor-pipeline` | Docs + contracts |

```
tabpfn-regressor-dataset-validator/     (CPU)
├── Dockerfile
├── validate.py          DIMER-facing entrypoint
├── requirements.txt
└── tests/

tabpfn-regressor-finetuner/             (GPU / CPU fallback)
├── Dockerfile           torch 2.11 / CUDA 12.8 base (sm_120 / Blackwell)
├── train.py             DIMER-facing entrypoint
├── dimer-pipeline.json  preprocessing + fine-tuning fields
├── requirements.txt
└── tests/
```

DIMER builds each repository from its root and launches the container by the portal naming
convention: `validate.py` for the validator and `train.py` for the fine-tuner.

Keep `dimer-pipeline.json` at the fine-tuner repository root. It defines the preprocessing and
fine-tuning fields end users see. Without it, the workbench preprocessing step renders empty and
the fine-tuning step stays locked.

The dataset spec ([`TABULAR_REGRESSION_DATASET_SPEC.md`](TABULAR_REGRESSION_DATASET_SPEC.md)),
the contract ([`CONTRACT.md`](CONTRACT.md)), the model card ([`MODEL_CARD.md`](MODEL_CARD.md)),
the deployment runbook ([`DEPLOYMENT.md`](DEPLOYMENT.md)), the dataset-building helpers
([`examples/`](examples/)), and the serving loader ([`serving/`](serving/)) live in this umbrella
repository, not in the container repositories.

---

## Creating the pipeline

Prerequisites: portal access as AI Engineer, and both repositories reachable by the portal's
GitHub App.

1. Open **AI Engineer → New Pipeline** and set these fields:

   | Field | Value |
   |---|---|
   | Pipeline Name | `TabPFN Tabular Regression` |
   | Description | `Fine-tune the TabPFN tabular foundation model for regression using your own tabular dataset. Supports dataset validation, configurable preprocessing, MAE/RMSE/R²/MAPE evaluation, and export of the trained model.` |
   | Task Type | `Custom / Other` |
   | Base Model | `Prior-Labs/tabpfn_3` (v3 generation) |
   | Validator repository | `https://github.com/kurtvalcorza/tabpfn-regressor-dataset-validator` |
   | Fine-tuner repository | `https://github.com/kurtvalcorza/tabpfn-regressor-finetuner` |

2. Build both images.
3. Validate and fine-tune a small regression dataset.
4. Enable the pipeline **only after** the on-platform serving check in the acceptance test passes.

### Portal implementation notes

- **`Custom / Other` is the correct portal card** for tabular pipelines; DIMER has no native
  tabular task type. Both container images set `DIMER_TASK_TYPE=tabular_regression` as their baked
  fallback and treat any value the platform sends as an override. Result payloads report
  `metadata.taskType`, defaulting to `tabular_regression`.
- **`dimer-pipeline.json` stays at the fine-tuner repository root.** The portal reads it there to
  render the preprocessing and fine-tuning fields.
- **Field-to-runtime mapping.** `datasetPreprocessing` keys are passed to the fine-tuner as
  `DIMER_PREPROCESSING_ARGS_JSON`; `modelFinetuning` keys as `DIMER_HYPERPARAMETERS_JSON`. Every
  declared key maps to concrete behavior in `train.py` — there are no silently-ignored controls.
- **`model_id` is not a hyperparameter.** The base model is selected by DIMER's **Base Model**
  field, surfaced to the runtime as `model_version` plus an optional pinned checkpoint via
  `DIMER_TABPFN_MODEL_PATH` (see [Provenance and traceability](#provenance-and-traceability)).

---

## The dataset

### Format

A directory or one top-level zip of CSV files. The full contract is in
[`TABULAR_REGRESSION_DATASET_SPEC.md`](TABULAR_REGRESSION_DATASET_SPEC.md).

```
dataset.zip
├── train.csv          (required)   one row per example; one finite numeric target column
├── val.csv            (optional)   same columns as train; a random holdout is split if absent
└── test.csv           (optional)   scored after fitting, never used for selection
```

The target column is named `target` by default; change it with the `target_column` preprocessing
field. Every other column, except those listed in `drop_columns`, is a feature; features may be
numeric or categorical. Target values must be **finite numeric when non-null**; null-target rows
are unusable and are removed before training. At least **two distinct usable target values** and
at least **50 usable training rows** are required, and at least one feature column must remain
after removing the target and `drop_columns`. Duplicate logical split names in the archive are
rejected so the validator and fine-tuner cannot resolve different files.

### How to build a dataset

TabPFN consumes a feature table, not raw records. Convert a time series or transaction log
(`entity, date, value`) into a training table by engineering one row per `(entity, date)`:

- **features** — history and context at that point: lags, rolling means and standard deviations,
  calendar fields, and any known covariates (promotions, holidays, weather, stock status);
- **target** — the numeric value to predict, e.g. the quantity a chosen number of days ahead.

[`examples/build_synthetic_dataset.py`](examples/build_synthetic_dataset.py) is a runnable
template that writes a valid dataset zip with numeric and categorical features, negative and
positive target values, and explicit train/validation/test splits:

```bash
python examples/build_synthetic_dataset.py --out /tmp/tabpfn-regression.zip
```

For forecasting problems, supply time-separated `val.csv` / `test.csv` rather than relying on the
generic random fallback holdout; the validator checks structural correctness but cannot infer
whether a split is leakage-safe.

### Row and feature ceilings

Capacity is **version-aware** (see [Generations and capacity limits](#generations-and-capacity-limits)):
the feature ceiling and row cap follow the selected generation, defaulting to the strictest tier
when the version is unknown. The default feature ceiling is **2,000** columns (a hard reject).
`max_train_rows` is a **deterministic training cap** (not a TabPFN hard limit); if the train table
exceeds it, the fine-tuner samples rows with the configured seed after the validation split is
built. Archive guards reject path traversal, suspicious compression ratios, oversized members,
excessive file counts, and excessive total expansion; a single inner ZIP is transparently
unwrapped and validated (only multiple top-level ZIPs are rejected). These limits are overridable
by platform environment variables for an intentionally larger profile.

### Data licence governs the served model

A served pipeline is bound both by the model-weight licence (which for TabPFN's default v3 is
**non-commercial**, see [The model: TabPFN](#the-model-tabpfn)) and by the licence of the data it
was trained on. A model fine-tuned on non-commercial data — for example CC BY-NC — may not be
appropriate to expose as a hosted service. Confirm the licence of both the selected TabPFN
generation and any training corpus before you enable a pipeline built from them.

---

## Configurable fields

From the fine-tuner's [`dimer-pipeline.json`](https://github.com/kurtvalcorza/tabpfn-regressor-finetuner/blob/main/dimer-pipeline.json).

Preprocessing (`datasetPreprocessing`):

| Field | Default | Purpose |
|---|---|---|
| `target_column` | `target` | Name of the finite numeric column to predict (1–128 chars) |
| `drop_columns` | — | Comma-separated columns to exclude from features (ids, raw timestamps) |
| `max_train_rows` | `100000` | Deterministic cap on training rows (range 50–100,000); larger tables are sampled to it |
| `validation_split` | `0.2` | Holdout fraction when the zip has no `val.csv` (range 0.05–0.4) |

Fine-tuning (`modelFinetuning`):

| Field | Default | Purpose |
|---|---|---|
| `fine_tune` | `true` | Update TabPFN weights (needs CUDA) or run zero-shot ICL; no GPU forces zero-shot |
| `model_version` | `v3` | TabPFN generation (`v3`, `default`, `v2.6`, `v2.5`, `v2`) |
| `epochs` | `30` | Maximum fine-tuning epochs (range 1–200) |
| `learning_rate` | `1e-5` | AdamW learning rate (range 1e-7–1e-3) |
| `weight_decay` | `0.01` | AdamW weight decay (range 0–0.2) |
| `time_limit_seconds` | `3600` | Wall-clock fine-tuning budget (range 60–21,600) |
| `early_stopping_patience` | `8` | Validation epochs without MSE improvement before early stopping (range 1–50) |
| `n_finetune_ctx_plus_query_samples` | `10000` | Max context-plus-query samples per fine-tuning meta-dataset (range 1,000–200,000) |
| `n_estimators_finetune` | `2` | Ensemble estimators inside the fine-tuning loop (range 1–16) |
| `n_estimators_validation` | `2` | Estimators used for validation during fine-tuning (range 1–16) |
| `n_estimators_final_inference` | `8` | Estimators retained for the final fitted inference model (range 1–32) |
| `prediction_batch_rows` | `4096` | Validation/test prediction chunk size, to bound peak memory (range 128–32,768) |
| `seed` | `0` | RNG seed for sampling, holdout generation, and fine-tuning (range 0–2,147,483,647) |

Regression has **no `eval_metric` enum**. The upstream TabPFN 8.1.0 fine-tuning wrapper uses
**MSE** internally for early-stopping / validation; DIMER independently reports **MAE, RMSE, R²,
and MAPE** (the latter over non-zero target rows only).

---

## Outputs

A successful run materializes the DIMER artifact layout under the run's output directory
(`DIMER_OUTPUT_DIR` = `/data/fine-tuning/<run_id>/`) alongside a `result.json` describing the run:

```
/data/fine-tuning/<run_id>/
├── artifacts/
│   ├── model.tabpfn_fit          fitted estimator state (foundation weights intentionally omitted)
│   ├── model.ckpt                companion foundation checkpoint (recombined by the loader)
│   └── artifact_manifest.json    SHA-256 of the fitted state + checkpoint
├── evaluation/report.json        validation / optional test metrics
├── logs/run-summary.json         run parameters and provenance summary
├── progress/epoch_*.json         best-effort per-epoch telemetry (never fails a run)
└── result.json                   the run envelope
```

`result.json` follows the shared tabular-FM schema (full form in [`CONTRACT.md`](CONTRACT.md)):

```jsonc
{
  "successful": true,
  "metrics": {
    "mode": "fine-tune",           // or "zero-shot-icl"
    "fineTuneEffective": true,      // false when a fine-tune request fell back to ICL
    "fineTuneSkippedReason": null,  // populated on CPU fallback
    "deviceName": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
    "validation": { "mae": 0.2739, "rmse": 0.3403, "r2": 0.9949,
                    "predictionMin": -18.26, "predictionMax": 13.54 },
    "test": { "mae": 0.2685, "rmse": 0.3468, "r2": 0.9950 }
  },
  "artifacts": {
    "modelArtifact":       { "path": "fine-tuning/<run_id>/artifacts/model.tabpfn_fit", "name": "model.tabpfn_fit", "contentType": "application/octet-stream", "sizeBytes": 75221 },
    "foundationCheckpoint":{ "path": "fine-tuning/<run_id>/artifacts/model.ckpt",        "name": "model.ckpt",        "contentType": "application/octet-stream", "sizeBytes": 233304651 },
    "evaluationReport":    { "path": "fine-tuning/<run_id>/evaluation/report.json",      "name": "report.json",       "contentType": "application/json" },
    "logArtifact":         { "path": "fine-tuning/<run_id>/logs/run-summary.json",       "name": "run-summary.json",  "contentType": "application/json" },
    "manifest":            { "path": "fine-tuning/<run_id>/artifacts/artifact_manifest.json", "name": "artifact_manifest.json", "contentType": "application/json" },
    "fittedSha256": "…", "foundationCheckpointSha256": "…",
    "reloadCheck": { "passed": true, "device": "cpu", "rows": 8 }
  },
  "provenance": { "model": { "tabpfnVersion": "8.1.0", "modelVersionRequested": "v3", "baseModelSha256": "…" },
                  "dataset": { "sha256": "…" }, "seed": 0 },
  "metadata": { "taskType": "tabular_regression", "baseModel": "…", "selectedModelId": "…",
                "device": { "selectedDevice": "cuda:0", "cudaAvailable": true, "fallbackReason": null } }
}
```

The example values above are the measured v3 smoke run from
[`PHASE2_ACCEPTANCE_REPORT.md`](PHASE2_ACCEPTANCE_REPORT.md). Predictions are reported as-is with
**no clipping** — the error matches the served artifact's behaviour, and negative predictions are
valid. When the dataset zip includes a `test.csv`, it is scored after fitting and never
influences selection.

`artifacts.modelArtifact` is the file DIMER's `export-to-repository` resolves (via the backend's
`_resolve_result_artifact`, reading `path`/`key`); in GPU-burst mode it points at the uploaded S3
model key instead of the `/data`-relative path. **No literal `best.pt` is needed** on the default
path (unlike the image pipelines). `artifacts.reloadCheck` is written only after the fine-tuner
reloads the saved artifact in-process and predicts — success is never reported for an artifact
that cannot be reloaded. On failure the same envelope is written with `successful: false` and an
`error` object.

### Serving loader

TabPFN's `.tabpfn_fit` archive intentionally omits the foundation weights; the companion
`model.ckpt` sits beside it under `artifacts/`. [`serving/load_artifact.py`](serving/load_artifact.py)
(`load_dimer_tabpfn_regressor_artifact`) is directory-parameterized: it rewrites the fitted
archive's recorded `model_path` to the colocated `model.ckpt` after DIMER relocates the artifact,
then loads the estimator. DIMER serving must use this loader (or equivalent logic).

### GPU burst (S3) mode

When `GPU_BURST_MODE` is set, the fine-tuner reads the dataset from and writes `result.json` and
the model back to S3 (`GPU_BURST_S3_BUCKET` / `GPU_BURST_DATASET_PREFIX` / `GPU_BURST_RESULT_KEY`
/ `GPU_BURST_MODEL_KEY`, via the `boto3` dependency) instead of the `/data` mount, which is not
mounted then. In burst mode `modelArtifact` points at `GPU_BURST_MODEL_KEY`. The path is inactive
unless the platform sets those variables.

---

## Reproducibility

TabPFN fine-tuning is stochastic: two runs on identical data can differ unless the seed is fixed.
`seed` is a first-class field and seeds the split, the sampling, and TabPFN itself. GPU kernel
autotuning can still leave small residual variation, so runs are reproducible in ranking but not
guaranteed byte-identical. For the strongest guarantee that every run starts from identical
weights, mount a specific checkpoint via `DIMER_TABPFN_MODEL_PATH` — the run then hashes that
exact file (`baseModelSha256`) rather than relying on a mutable cache resolution.

---

## Resource profile

Each fine-tuning run executes as a Kubernetes job. **The default DIMER deployment provisions no
GPU node pool** — GPU is opt-in and off by default — so a fine-tune request on the default profile
falls back to zero-shot ICL rather than failing. Provision a GPU profile only when task-specific
fine-tuning is actually required.

TabPFN holds the training table in memory as in-context context, so its footprint grows with rows
and features. On the Phase 2 smoke dataset (600-row synthetic, 3 features), measured peak GPU
memory was **3.0 GB for v2 and 4.1 GB for v3**; wall-clock was ~13–14 s. These are smoke-only
figures, **not production ceilings** — set the production GPU/RAM profile from measured peak usage
on representative data during the acceptance test (see [DEPLOYMENT.md](DEPLOYMENT.md)).

The fine-tuner image is built on **`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`** (CUDA 12.8,
sm_120 / Blackwell kernels). It detects CUDA at runtime: with a GPU it fine-tunes; without one it
runs zero-shot ICL on CPU. The validator is CPU-only.

---

## Provenance and traceability

This section records how the pipeline was built and how the models it produces stay auditable.

### How this pipeline was authored

The validator, fine-tuner, configuration, and documentation in this repository were drafted with
AI assistance (Anthropic Claude Opus 4.8, via Claude Code) and are pending human review before
production deployment. The following were verified by execution, not only generated (see
[`PHASE2_ACCEPTANCE_REPORT.md`](PHASE2_ACCEPTANCE_REPORT.md), 2026-08-19, RTX 5070 Ti / sm_120):

- both container scripts byte-compile; the validator passes its full check set (18/18 checks) on
  the synthetic sample and fails closed on every negative (duplicate split, non-numeric target,
  too-few-rows, constant target, path traversal, zip bomb); component unit tests pass
  (validator 9/9, fine-tuner 11/11);
- the fine-tuner trains TabPFN on a real GPU for **both v2 and v3**, writes a valid artifact, and
  that artifact **reloads and predicts in a fresh container** through `serving/load_artifact.py`
  (CUDA reload exact to 1e-6; CPU within device float noise);
- reloaded predictions span negative→positive with **no clipping**;
- two real defects found during the run (Docker PEP 668 build failure; a non-JSON-serializable
  fitted-state save) were reproduced on the unpatched build, fixed, and re-verified from the
  patched build;
- the base weights' SHA-256 is verified when a checkpoint is mounted, and `test.csv` is scored
  when present.

Not yet verified, and requiring human sign-off: the DIMER portal image build, the on-platform
**inference-serving integration** (gate 8, deferred), the production resource profile on
representative data, and — for v3/v2.5/v2.6 — a commercial licence for any non-evaluation use.
Treat the generated code as a reviewed draft, not audited production code, and do not
production-enable until the end-to-end serving check in [DEPLOYMENT.md](DEPLOYMENT.md) passes.

### Model lineage

| Field | Value |
|---|---|
| Base model | [TabPFN](https://huggingface.co/Prior-Labs/tabpfn_3) by Prior Labs |
| Package | `tabpfn==8.1.0` |
| Default generation | v3 (`Prior-Labs/tabpfn_3`, HF-gated, non-commercial) |
| Licence | v2: Prior Labs Apache-derived (commercial OK); v2.5 / v2.6 / v3: non-commercial |
| Framework | `torch 2.11.0+cu128`; base image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` |
| Provenance hook | mount a checkpoint via `DIMER_TABPFN_MODEL_PATH`; the run hashes it (`baseModelSha256`) |

The non-commercial generations (v2.5 / v2.6 / v3) must be supplied as an accepted checkpoint via
`DIMER_TABPFN_MODEL_PATH`; v3 additionally requires accepting the Hugging Face gate on
`Prior-Labs/tabpfn_3`. Only v2 auto-downloads freely under its Apache-derived licence.

### Data lineage

A trained model inherits the provenance and licence of the table it was fine-tuned on. Each
dataset should carry its source, its licence, and — for a derived table — the transformation that
produced it. The input dataset bytes are SHA-256 hashed into every run's provenance record.

### Per-run record

Every run writes a `result.json` that serves as the run's provenance record: the effective mode
and device (with any fallback reason), the target and dropped columns, the seed, time budget, and
schedule, the row counts, the resulting MAE / RMSE / R² / MAPE metrics, the fitted-state and
foundation-checkpoint SHA-256s, the resolved TabPFN generation and any mounted checkpoint hash,
and the input dataset digest. Paired with the container image tag and the component commits pinned
in [`COMPONENTS.json`](COMPONENTS.json), this forms a chain from data to served model.

---

## References

- [TabPFN v3 model card and gated checkpoints](https://huggingface.co/Prior-Labs/tabpfn_3)
  (`Prior-Labs/tabpfn_3`), Prior Labs, Hugging Face — non-commercial, gated.
- [`tabpfn`](https://pypi.org/project/tabpfn/) package (pinned `8.1.0`), PyPI.
- [Prior Labs](https://priorlabs.ai).
- Pipeline contract and runbooks in this repository: [CONTRACT.md](CONTRACT.md),
  [MODEL_CARD.md](MODEL_CARD.md), [DEPLOYMENT.md](DEPLOYMENT.md),
  [TABULAR_REGRESSION_DATASET_SPEC.md](TABULAR_REGRESSION_DATASET_SPEC.md),
  [PHASE2_ACCEPTANCE_REPORT.md](PHASE2_ACCEPTANCE_REPORT.md).

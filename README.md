# TabPFN Regressor — DIMER Pipeline

DIMER integration for fine-tuning TabPFN on tabular regression datasets.

This repository is the **integration/specification layer**. It does not duplicate deployable container source. The validator and fine-tuner live in separate repositories and immutable reviewed revisions are pinned in `COMPONENTS.json`.

## Repositories

| Component | Repository |
|---|---|
| Pipeline/specification | `tabpfn-regressor-pipeline` |
| Dataset validator | `tabpfn-regressor-dataset-validator` |
| Fine-tuner | `tabpfn-regressor-finetuner` |

## Dataset

Upload a ZIP or directory containing:

```text
train.csv   required
val.csv     optional
 test.csv   optional
```

Each table has one finite numeric target column (`target` by default) plus feature columns. Negative target values are valid. `val.csv` and `test.csv`, when supplied, must use the same schema as `train.csv`.

If `val.csv` is absent, the fine-tuner creates a deterministic random holdout. For temporal, grouped, panel, spatial, or otherwise leakage-sensitive data, provide an explicit validation/test split produced using the correct domain boundary.

See `TABULAR_REGRESSION_DATASET_SPEC.md`.

## Training modes

### Fine-tune

`fine_tune=true` uses `FinetunedTabPFNRegressor` and updates TabPFN weights for the uploaded task. CUDA is required. A requested fine-tune fails clearly if CUDA is unavailable; it never silently changes semantics.

### Zero-shot / ICL

`fine_tune=false` uses `TabPFNRegressor` without task-specific gradient updates.

The container pins `tabpfn==8.1.0` and defaults to model generation `v3`. A deployment can supply `DIMER_TABPFN_MODEL_PATH` to use a platform-managed checkpoint and obtain checkpoint-level provenance.

## Evaluation

`val.csv` is validation data and `test.csv` is held out until after fitting. Results report:

- MAE
- RMSE
- R²
- MAPE over non-zero target rows
- prediction minimum/maximum

Predictions are not clipped. In particular, the pipeline makes no non-negativity assumption about a general regression target.

## Artifacts

Successful training produces:

```text
/data/fine-tuning/<run_id>/
├── artifacts/
│   ├── model.tabpfn_fit
│   ├── model.ckpt
│   └── artifact_manifest.json
├── evaluation/report.json
├── logs/run-summary.json
├── progress/epoch_*.json    per-epoch telemetry (best-effort)
└── result.json
```

TabPFN's fitted-state archive intentionally omits foundation weights. `serving/load_artifact.py` rewrites the fitted state's recorded model path to the companion `model.ckpt` after DIMER relocates the artifact. `result.json` declares `artifacts.modelArtifact` (path relative to `/data`) that `export-to-repository` resolves; in GPU-burst mode it points at the S3 model key the run uploaded instead.

## Smoke dataset

```bash
python examples/build_synthetic_dataset.py --out /tmp/tabpfn-regression.zip
```

The fixture contains numeric and categorical features, negative and positive target values, and explicit train/validation/test splits.

## Verification gates

CPU CI covers dataset contract behavior, regression metrics, negative predictions, split isolation, archive safety, JSON validity, synthetic fixture generation, and artifact relocation. Production acceptance additionally requires a real CUDA fine-tune, clean-runtime artifact reload, and DIMER inference request.

Review the selected TabPFN package and model-weight licensing/usage terms before enabling a hosted production service.

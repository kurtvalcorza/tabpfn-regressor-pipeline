# Tabular Regression Dataset Specification

## Required layout

A dataset is a directory or one top-level ZIP containing exactly one logical `train.csv` and optionally one logical `val.csv` and `test.csv`.

Nested directories are allowed, but duplicate logical names are rejected. Archives containing path traversal, nested ZIPs, suspicious compression ratios, oversized members, excessive file counts, or excessive total expansion are rejected.

## Columns

`target_column` defaults to `target`.

- Target values must be finite numeric values when non-null.
- Null target rows are unusable and are removed before training/evaluation.
- Negative, zero, and positive target values are all valid.
- At least two distinct usable target values are required.
- At least one feature column must remain after removing the target and `drop_columns`.
- The configured default feature ceiling is 2,000 columns.

`val.csv` and `test.csv` must have the same column set as `train.csv` before preprocessing.

## Split semantics

`train.csv` is fit data.

`val.csv`, when supplied, is validation data passed to the fine-tuning loop and used for reported validation metrics. It is not mixed back into training.

If `val.csv` is absent, a deterministic random holdout is created from `train.csv` using `validation_split` and `seed`.

`test.csv`, when supplied, is never used to fit or tune the model. It is evaluated only after fitting completes.

### Leakage-sensitive data

A random holdout is not statistically appropriate for every dataset. For time series, forecasting feature tables, grouped entities, panels, geospatial blocks, repeated subjects, or any problem where nearby observations are dependent, create `val.csv` and `test.csv` using the correct temporal/group/domain boundary before upload.

The validator checks structural correctness; it cannot infer whether a split is scientifically leakage-safe.

## Operating limits

Defaults can be overridden by deployment environment variables where documented:

- minimum usable training rows: 50
- feature ceiling: 2,000
- advisory row range: 100,000
- top-level ZIP count: one
- archive/resource limits: enforced by both validator and trainer

`max_train_rows` is a deterministic training cap, not a TabPFN hard architectural limit. If the train table exceeds it, the fine-tuner samples rows with the configured seed after validation split construction.

## Metrics

The pipeline reports MAE, RMSE, R², and MAPE on non-zero target rows. MAPE is omitted (`null`) when no evaluated target is non-zero. Predictions are evaluated as emitted by TabPFN; no non-negative clipping is applied.

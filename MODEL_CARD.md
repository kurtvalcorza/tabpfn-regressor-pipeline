---
license: other
license_name: tabpfn-3-license-v1.0
model_card_spec: "1.0"
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - tabular-foundation-model
  - in-context-learning
  - tabpfn
base_model: Prior-Labs/tabpfn_3
---

# TabPFN-3 Regressor (tabpfn 8.1.0)


###### Description

TabPFN Regressor packages Prior Labs' TabPFN (Tabular Prior-data Fitted Network) through the `tabpfn==8.1.0` package, with the TabPFN-3 generation selected by default (`model_version: v3`) and v2, v2.5, and v2.6 selectable through DIMER configuration. TabPFN is a Transformer trained on a prior over synthetic tabular tasks so that it performs supervised regression in a single forward pass: the labelled training rows are the in-context support, the query rows attend to them, and the head emits a continuous estimate. Negative targets and predictions are valid; no non-negativity is assumed. The v3 checkpoint's stored inference config admits up to 1,000,000 samples and 2,000 features; the earlier generations are narrower.

At inference the model conditions on the operator's training table; adaptation happens through in-context conditioning by default and, when `fine_tune=true` and a CUDA GPU is present, through gradient fine-tuning with Prior Labs' `FinetunedTabPFNRegressor` on explicit train and validation sets, whose internal early stopping uses MSE. What this repository adds is the DIMER composition: the pipeline contract (`dimer-pipeline.json`, `CONTRACT.md`), the component manifest pinning the validator and fine-tuner commits (`COMPONENTS.json`), an artifact loader for serving (`serving/load_artifact.py`), tests, and a synthetic example builder; the validator and fine-tuner workers live in the sibling `tabpfn-regressor-dataset-validator` and `tabpfn-regressor-finetuner` repositories. The upstream weights are not modified by this repository.

#### Intended Use and Limitations

The use cases below are the ones envisioned during development; the limits are the ones the workers enforce.

###### Primary Intended Uses

Supervised prediction of a finite numeric target from tabular data where each observation is one row of mixed numeric and categorical predictor columns. The pipeline takes a `train.csv` (optionally `val.csv`/`test.csv`) with a declared target column and produces a fitted TabPFN artifact pair (`model.tabpfn_fit` + `model.ckpt`), one point prediction per row, and holdout error metrics.

Concrete application domains envisioned during development: demand and quantity estimation, price and cost estimation, continuous risk or score prediction, and scientific or engineering regression on feature tables of small to medium size, where the operator wants strong performance without hyperparameter search. The pipeline is meant to play the role of a strong zero-shot baseline, or a fine-tuned model when a GPU is available, inside DIMER. Enforced ceilings follow the selected generation (v3: 1,000,000 rows, 2,000 features; v2: 10,000 / 500), further capped by the DIMER `max_train_rows` preprocessing argument. A provided `test.csv` is used only for post-fit evaluation, never for early stopping or checkpoint selection.

###### Primary Intended Users

Machine-learning engineers, data scientists, and researchers building predictive systems from structured datasets, and DIMER integrators provisioning the workers. The envisioned deployment setting is internal enterprise or research use through the DIMER platform — CPU by default, GPU opt-in — and, because `tabpfn-3-license-v1.0` limits the TabPFN-3 weights to a Non-Commercial Purpose that its own definition says excludes production deployment and revenue generation, testing and evaluation only until a production enablement has been cleared against those terms.

The pipeline assumes its users understand dataset provenance, holdout evaluation, leakage, target scale, and distribution shift; know that the served prediction is a point estimate with no attached interval; can read `metrics.fineTuneEffective` to tell a fine-tuned run from a zero-shot fallback; and understand that random splitting is wrong for temporal or grouped data, where they must supply explicit splits. A user who would ship a zero-shot fallback believing it was fine-tuned is outside the assumed competency.

###### Out-of-scope use cases

- **Capability boundaries:** categorical classification (the sibling `tabpfn-classifier-pipeline` does that); interval or quantile output (the pipeline emits a point estimate only); image, text, object-detection, or segmentation tasks; unsupervised clustering; causal-effect estimation; time-series forecasting without tabular feature construction.
- **Input boundaries:** tables exceeding the selected generation's caps (v3: 1,000,000 rows, 2,000 features — the validator refuses, the fine-tuner re-checks); a target that is not a finite number; datasets whose rows are not exchangeable (temporal or grouped) unless explicit `val.csv`/`test.csv` splits are provided, because the default random split leaks across time and groups.
- **Decision boundaries:** autonomous high-impact decisions — pricing of credit or insurance, clinical dosing, safety margins — without application-specific validation and a human decision-maker; treating TabPFN benchmark strength as a guarantee of task-level superiority; production deployment or commercial hosted use of the TabPFN-3 weights without clearing the model-weight terms, which define Non-Commercial Purpose to exclude both.

#### Factors

TabPFN's behaviour varies with the structure of the table it is given, not with a physical capture condition; the three subsections below say what that means for groups, instruments, and environment.

###### Groups

This pipeline is not human-centric by construction: TabPFN's pretraining prior is synthetic, so no demographic group is an intrinsic development group of the foundation model. Whether any real-world tables entered the TabPFN-3 pretraining mix is not enumerated in this repository and is treated as unknown; in either case the pretraining corpus is not group-audited, demographic fairness or subgroup parity has **not** been established for the checkpoint, and the pipeline measures no subgroup metric.

Where the operator's downstream table describes people, the obligation transfers to the operator: identify the relevant groups in their own data, compute per-group MAE and RMSE on the holdout split, and check for disparate error before deployment — a regressor that is unbiased on average can still be systematically high for one group and low for another. The validator result records row counts and target statistics, not any demographic structure.

###### Instrumentation

TabPFN consumes an abstract tabular representation rather than a raw sensor stream. The instrument does not disappear because a table sits between it and the model: the operator's rows are produced by whatever systems fed the CSV — transactional databases, ETL pipelines, sensors, survey instruments — and their sampling rate, resolution, calibration, units, and encoding of missing values determine both feature quality and the scale of the target.

Instrument error reaches the model as feature or target error. Drift, miscalibration, a unit change, or a changed collection procedure between training and inference is not detectable by this pipeline; the validator checks schema, row counts, target finiteness, and generation caps, not whether a column's meaning or unit has changed. Operators should document the instrumentation of downstream datasets separately.

###### Environment

**Operating environment.** CPU is the default deployment. Fine-tuning requires a CUDA GPU — Prior Labs' official example recommends 80 GB — and when `fine_tune=true` is requested without a usable CUDA device the run does **not** fail: it falls back to zero-shot in-context learning and records `metrics.fineTuneSkippedReason` with `metrics.fineTuneEffective=false`. Prediction is chunked at `prediction_batch_rows` to bound memory. The fine-tuner records `cudaAvailable` and the device name in provenance.

**Data environment.** The reported behaviour assumes the inference rows are exchangeable with the training rows: same feature semantics, same units, same target range. Performance degrades, without warning from the pipeline, under distribution shift, heavy-tailed or sparse targets, high-cardinality categoricals, unusual missingness, and non-independent rows; predictions outside the training target range are extrapolations the model has no basis for; an apparently strong holdout score on temporal or grouped data split at random is not evidence of anything. Robustness to arbitrary distribution shift has not been established.

#### Metrics

Metrics are chosen for a point-estimate regressor whose intended use spans targets of very different scales and sign.

###### Performance Measures

The fine-tuner scores the fitted model on the validation split, and on `test.csv` when provided, (`tabpfn-regressor-finetuner/train.py`) and writes `mae`, `rmse`, `r2` (when at least two rows), and `mapePercent` with `mapeRows` — MAPE computed only over rows whose true value is non-zero — into `evaluation/report.json` and the `result.json` envelope. Upstream's fine-tuning loop separately uses MSE for its own early stopping; that value is not the reported metric.

Why these: MAE is robust to outliers and interpretable in target units; RMSE weights large errors quadratically and is the informative one when a few big misses matter more than many small ones; R² normalises against the holdout target variance so tables of different scale can be compared, at the cost of misleading on low-variance targets; MAPE gives a scale-free relative error but is undefined at zero, which is why it is restricted to non-zero rows and its row count is written beside it. Reading only MAE hides tail failures, reading only RMSE lets one outlier dominate, which is why all four are written. The pipeline reports no upstream benchmark number of its own; TabPFN's published rankings are not claimed here.

###### Decision thresholds

The pipeline applies no decision threshold and no clipping: TabPFN's `predict()` returns a raw continuous estimate — negative values included — and the served artifact returns it unchanged, so the reported error reflects what a caller will actually receive. No acceptance threshold on any metric was set during development, because the pipeline is domain-agnostic and the tolerable error is a property of the deployment.

Any cutoff that turns a prediction into an action — a reorder point, a price band, a tolerance margin — is the deployment owner's to define and to calibrate on their own held-out residuals. Set it from the asymmetric cost of over- versus under-prediction: where an under-estimate is the expensive error, place the operating value above the point prediction by a margin derived from the holdout residual distribution, and revisit it whenever the input distribution or the target scale shifts.

###### Approaches to uncertainty and variability

The pipeline's reported metrics come from a single validation split (a random holdout when `val.csv` is absent) and, when supplied, a single `test.csv`. No dispersion is reported alongside the point value: one split, one run, no confidence interval. Operators who need one should repeat the run across seeds or use cross-validation on their own side.

Sources of run-to-run variability: the holdout split, the training cap, TabPFN's internal estimator ensembling (`n_estimators_finetune`, `n_estimators_validation`, `n_estimators_final_inference`, defaults 2/2/8), and gradient fine-tuning; all are driven by the DIMER `seed` hyperparameter, which `_seed_everything` propagates to Python, NumPy, and torch (including CUDA) and which is passed as `random_state` to the estimators. Non-deterministic CUDA kernels can still produce small differences. The pipeline emits no confidence output — no interval, no quantile, no predictive variance — so there is nothing to calibrate; a caller who needs an interval must estimate one from their own holdout residuals and should treat it as valid only within the training target range. A zero-shot fallback run and a fine-tuned run are different estimators and their metrics must not be compared as if from the same procedure — `fineTuneEffective` says which one ran.

#### Ethical considerations and biases

No external ethics board reviewed this pipeline, and no clearance testing with a specific group took place; the subsections record what the developers considered and what the repository actually does.

###### Data

TabPFN's pretraining prior is synthetic; whether the TabPFN-3 generation additionally saw real-world tables is not enumerated in this repository, so the sensitivity of the pretraining data is unknown rather than ruled out. What this repository distributes: the pipeline contract, component manifest, serving loader, tests, and a synthetic example builder (`examples/build_synthetic_dataset.py`); it does **not** distribute the checkpoint — the TabPFN-3 weights are a gated Hugging Face download under `tabpfn-3-license-v1.0`, and the local `weights/` mirror is gitignored.

Fine-tuned weights may encode information derived from the uploaded dataset, so the fitted artifact inherits the dataset's confidentiality. The pipeline does not audit the operator's data for personal, sensitive, or proprietary attributes — the validator checks structure, not content — so the legality, privacy, consent, retention, and governance of downstream data, and the licensing of the fine-tuned model, remain with the application developer and data owner.

###### Human Life

The pipeline is not intended for decisions in health care, physical safety, criminal justice, legal rights, employment, credit, insurance, education access, or public benefits, and it has not been validated for any of them. The only validation performed is the contract and integration testing in `tests/` (CPU CI, which does not exercise GPU fine-tuning or production serving) and the DIMER holdout evaluation on the operator's own table; no clinical, regulatory, or independent domain validation has been carried out by the developers or by any external body.

Where such a use is foreseeable — a triage classifier on a clinical feature table, for example — it would be admissible only with independent domain validation on that operator's population, a human decision-maker between the prediction and the action, subgroup evaluation, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented in the composed workers, each inspectable in the named code:

- **Supply-chain integrity:** `tabpfn` is pinned to 8.1.0 and the component commits are pinned in `COMPONENTS.json`. The selected generation is explicit (`model_version`) and a mismatch between the DIMER-resolved and requested version raises `MODEL_IDENTITY_MISMATCH`. When DIMER mounts an approved checkpoint through `DIMER_TABPFN_MODEL_PATH`, its SHA-256 is recorded and, if the model config carries `expectedSha256`, a mismatch raises `MODEL_INTEGRITY_FAILED`. Without a mounted checkpoint the package's cached weights are used and only their digest is recorded — that path is not pinned, and the card says so.
- **Input integrity:** the validator enforces the selected generation's row and feature caps and requires a finite numeric target; `test.csv` is isolated from fine-tuning, early stopping, and checkpoint selection.
- **Statistical mitigations:** MAPE is computed only over non-zero rows with the row count recorded, so a zero-inflated target cannot produce an undefined or misleading relative error; evaluation prediction is chunked (`prediction_batch_rows`, default 4,096) to bound memory; predictions are scored raw so the reported error is the error a caller will see.
- **Reproducibility:** `seed` propagates to Python, NumPy, torch, and the estimators; the artifact manifest records target column, ordered feature columns, and per-artifact SHA-256; the dataset fingerprint is recorded.
- **Refusals:** a GPU-less fine-tune request is not silently satisfied — the run falls back to zero-shot and says so in `fineTuneSkippedReason`; the serving layer must load the actual `model.tabpfn_fit` artifact before the pipeline is considered production-ready.

###### Risks and harms

- **Extrapolation outside the training range** (model-intrinsic): a query row beyond the support of the training table receives a point estimate with no signal of its own unreliability; borne by whoever the operator's decision affects; likely under normal use as the deployment drifts.
- **Tail and sparsity failure** (model-intrinsic): heavy-tailed, intermittent, or zero-inflated targets can produce low MAE and large individual misses; borne by the operator who reads MAE alone.
- **Amplification of input bias** (model-intrinsic): a table whose target encodes a historical disparity — past pay, past pricing — yields a regressor that reproduces it; borne by the data subjects in the disadvantaged group; realised whenever such a table is used without subgroup evaluation.
- **Silent zero-shot fallback misread as fine-tuning** (use-context): a run without a GPU still succeeds; an operator who does not read `fineTuneEffective` ships a different model than they think; borne by the operator.
- **Leakage through random splitting** (use-context): temporal or grouped data split at random produces a holdout score that collapses in production; the pipeline does not detect it; borne by the operator and downstream users.
- **Automation bias** (use-context): a numerically precise estimate displaces human judgement; borne by the data subject.
- **Licence breach** (use-context): production-deploying or commercially hosting the TabPFN-3 weights without clearance, since the licence's Non-Commercial Purpose excludes both; borne by the operator.

###### Use cases

Distinct from the capability and decision boundaries listed under *Out-of-scope use cases*, the developers consider the following uses prohibited even where the model would produce a numerically plausible estimate:

- surveillance, biometric or demographic profiling, or social scoring of individuals;
- unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access, including regression on a target that proxies a protected attribute (pay, premium, or limit set by group membership);
- deceptive, manipulative, or predatory applications, including exploitative price discrimination and presenting a point estimate as a certified measurement;
- clinical dosing, safety-margin, or legal-rights determinations without the validation and oversight described under *Human Life*;
- any use outside the terms of the selected model weights — for TabPFN-3, `tabpfn-3-license-v1.0`, whose Non-Commercial Purpose excludes production deployment and revenue generation without a separate agreement — or of the DIMER deployment.

---

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

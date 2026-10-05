"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment) — E2E.

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, checkpoint
staging) from repository files; this template holds the learner-facing prose, the guided layer and the learner
cells. Every learner cell calls ``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process
in an isolated, hash-locked environment, so nothing is installed into the notebook kernel. The ARTIFACT-INFERENCE
companion has its own template, ``tools/notebook_template_artifact_inference.py``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "tabpfn-regressor-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabpfn_regressor_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Prior--Labs%2Ftabpfn__3-ffcc4d?style=flat",
        "https://huggingface.co/Prior-Labs/tabpfn_3",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-PriorLabs%2FTabPFN-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/PriorLabs/TabPFN",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2605.13986-b31b1b.svg", "https://arxiv.org/abs/2605.13986"),
]



UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}
ENVIRONMENT = {
    "package": "tabpfn_regressor_pipeline",
    "repo_name": REPO,
    "weights_key": "tabpfn-3-regressor",
    "modules": ["__init__.py", "pipeline.py"],
    "entry_module": "pipeline.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 0.3, "environment": 8.0},
    "runtime_modules": ["torch", "tabpfn", "numpy", "pandas", "scikit-learn"],
    "install_flags": ["--only-binary", ":all:"],
    "license_file": "NOTICE",
}

RUNTIME_PREREQ = (
    "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab (CPU is enough for the default sample; a T4 GPU is used automatically when present), Kaggle or a Linux Jupyter kernel. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`tabpfn` 8.1.0, `torch` 2.11.0 with its CUDA libraries, `numpy` 2.5.3, `pandas` 2.3.2, `scikit-learn` 1.9.0). About 0.3 GB of disk is needed for the checkpoint and about 8 GB for the isolated environment."
)
LICENCE_PREREQ = "- **Licence:** the TabPFN-3 weights are non-commercial (`tabpfn-3-license-v1.0`): testing, evaluation and internal benchmarking only. A bundle carries a byte copy of the checkpoint, so the same terms travel with it. Clear the licence before any production use. No credentials are needed: `Prior-Labs/tabpfn_3` is public and not access-gated."

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabpfn_regressor",
    "notebook_name": "tabpfn_regressor_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, then runs each stage below in its own process: it stages and digest-verifies the pinned TabPFN-3 checkpoint (an ungated download; the TabPFN-3 licence's non-commercial terms still apply to what you do with it), draws the deterministic synthetic regression table in code (600 rows: `x1`, `x2` and a three-level `category`; seeded `train`/`val`/`test` split, no download), validates it into an input manifest, **fits the regressor in context on the training split** (support-row registration on the pinned checkpoint, no gradient update), evaluates on the held-out split against a training-mean baseline and a linear-regression reference and writes the evaluation report, exports the artifact bundle and reloads it in a fresh process with a per-row prediction-equivalence check, scores new rows and exports machine-readable results and provenance. No repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5). No hosted run of this revision has been recorded yet."
    ),
    "byod": (
        "Both optional branches are off by default and never part of the default path. `USE_BYOD = True` in Section 4 takes your own labelled table by `BYOD_PATH` (a ZIP with `train.csv` and optional `val.csv`/`test.csv`, a single `train.csv`, or a directory holding them — paths work in Colab, Kaggle and Jupyter) or, in Colab, an upload dialog; name the numeric target in `TARGET_COLUMN` and list identifier columns in `DROP_COLUMNS` (kept beside the predictions, never given to the model). A missing target, a target that is not numeric (`n.a.`, thousands separators), duplicate columns, or a numeric feature with a few stray strings stop Section 4 with a coded message naming the file. `USE_BYOD_ROWS = True` in Section 9 scores your own unlabelled rows, with identifier columns listed in `ID_COLUMNS` kept beside the predictions. Uploads stay inside this runtime."
    ),
    "title": "TabPFN-3 Regressor — DIMER E2E tabular regression tutorial (standalone)",
    "badges": BADGES,
    "capability": "supervised tabular regression by **in-context learning** with the pinned `Prior-Labs/tabpfn_3` regressor checkpoint: validated support rows, no gradient update, evaluation against trivial and classical baselines, point predictions without intervals, and a `model.tabpfn_fit` + `model.ckpt` + `artifact_manifest.json` bundle reloaded in a fresh process",
    "intro": (
        "TabPFN-3 is a Transformer trained on a prior over synthetic tabular tasks so that it performs supervised "
        "regression in a single forward pass: `fit` registers your labelled training rows as the in-context "
        "support and performs **no gradient update**; query rows attend to that support and the head emits a "
        "predictive distribution, whose mean is the point prediction. `n_estimators` averages several passes over differently preprocessed views of the table. The "
        "carried package owns the pinned snapshot scheme, the input contract, the in-context fit / predict / evaluate "
        "calls, the artifact bundle the serving path consumes (with its pre-load validation) and the evaluation report.\n\n"
        "**Adaptation.** The production DIMER pipeline can also fine-tune TabPFN by gradient descent (Prior Labs' "
        "`FinetunedTabPFNRegressor`, default on in `dimer-pipeline.json`, on a large GPU). This tutorial does not: "
        "in-context conditioning is its adaptation — the same path production falls back to without a large GPU. That is "
        "a proposed RUN7 deviation, recorded in the evaluation report and pending the maintainer's approval. The TabPFN-3 "
        "weights are released under `tabpfn-3-license-v1.0`, whose Non-Commercial Purpose excludes production "
        "deployment, so this tutorial is testing-and-evaluation material. Sample metrics on synthetic data are tutorial "
        "sanity evidence only; predictions are **point estimates only**, with no prediction interval, and the pipeline "
        "ships no tolerance band."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** what in-context learning means for TabPFN: what `fit` does and does not do (Section 6).\n"
        "2. **Diagnose** an invalid table from a coded validation finding, and explain why an identifier column must never "
        "be a feature (Sections 4, 5).\n"
        "3. **Compare** TabPFN with a training-mean baseline and a linear-regression reference on the same rows, and say "
        "when a linear model already sits at the noise floor (Section 7).\n"
        "4. **Interpret** MAE, RMSE and R², and explain why MAPE is omitted when targets come close to zero (Sections 6, 7).\n"
        "5. **Verify** that the exported bundle rebuilds the same predictions in a fresh process (Section 8).\n"
        "6. **Apply** the bundle to new rows and keep your identifiers beside the predictions (Section 9).\n"
        "7. **Predict**, run and **explain** the effect of the ensemble size in an optional activity (Section 10).\n"
        "8. **Write** an evidence-based conclusion that names the baselines and the limits of one synthetic table "
        "(Conclusion)."
    ),
    "exclusions": (
        "gradient fine-tuning (see **Adaptation** above), classification, prediction intervals, the v2 / v2.5 / v2.6 "
        "generations (only the pinned v3 checkpoint is carried), temporal or grouped splitting, or any quality claim "
        "beyond one holdout of one table."
    ),
    "prerequisites": [
        RUNTIME_PREREQ,
        "- **Knowledge:** basic Python and pandas, the train/val/test convention, and how to read a printed dictionary. The metrics and in-context learning are explained where they are first used, and the glossary collects them.",
        "- **Model file:** `tabpfn-v3-regressor-v3_default.ckpt` (222 MiB) plus the licence, README and config of the snapshot, staged at a fixed revision and digest-verified in Section 3.",
        "- **Data:** the default path draws a deterministic synthetic regression table (600 rows: numeric `x1`, `x2`, a three-level `category`, and a target that is linear in them plus noise; `train.csv`/`val.csv`/`test.csv`) in code and needs no download and no private data. BYOD (one labelled ZIP, `train.csv` or directory) is off by default. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
        LICENCE_PREREQ,
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python and pandas, "
                "and who want to see a tabular foundation model used end to end: validated data, in-context fitting, honest "
                "evaluation against simple references, and a reusable bundle checked across a fresh boundary. No experience with "
                "transformers is assumed; the glossary below explains every term.\n\n"
                "**Running it.** Choose *Runtime → Run all*. The default path needs no edit, no upload, no account, no token and "
                "no runtime restart. Section 2 builds an isolated environment, which takes the longest. You can also run one cell "
                "at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–10) are the machine-learning workflow. *Infrastructure cells* "
                "(Sections 1–3) are collapsed and titled **Infrastructure**; you may run them without studying their "
                "implementation.\n\n"
                "**Form controls.** `USE_BYOD`, `BYOD_PATH`, `TARGET_COLUMN`, `DROP_COLUMNS`, `TEXT_COLUMNS` and "
                "`VALIDATION_SPLIT` (Section 4); `N_ESTIMATORS` and `SEED` (Section 6); `USE_BYOD_ROWS`, `NEW_DATA_PATH` and "
                "`ID_COLUMNS` (Section 9); `RUN_ACTIVITY` and `ACTIVITY_N_ESTIMATORS` (Section 10). Leave them at their defaults for the first "
                "run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 6 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Validate** | a table with a numeric target (train / val / test) | `validate_inputs` with coded findings; identifiers set aside | an input manifest; 420 support, 90 validation and 90 test rows |\n"
                "| **Fit in context** | the support rows | TabPFN-3 registering them as context (`n_estimators=4`) | a point prediction for any query row |\n"
                "| **Evaluate** | validation and test rows | `regression_metrics`, training-mean baseline, linear regression | MAE / RMSE / R², an evaluation report |\n"
                "| **Package** | fitted state + checkpoint + manifest | `save_artifact`, `validate_artifact_bundle`, `from_artifact` | a bundle that rebuilds the same predictions |\n"
                "| **Score** | new unlabelled rows (+ identifiers) | the rebuilt estimator | `prediction` per row, identifiers kept |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux x86_64, GPU and disk; a run directory | the machine |\n"
                "| 2. Carry the code, build the environment | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify the model | [Engineering] | the 222 MiB checkpoint downloaded at a fixed revision, digest-checked | identity and digest |\n"
                "| 4. Prepare the dataset | [Concept] | the synthetic table (or your files); identifiers set aside | split sizes, target summary |\n"
                "| 5. Validate | [Evaluation practice] | input manifest; a refusal probe; the range warning; training-mean baseline | the manifest |\n"
                "| 6. Fit in context and evaluate | [Concept] | validation and test metrics | the metrics |\n"
                "| 7. Baselines and report | [Evaluation practice] | linear-regression reference, the MAPE rule, the evaluation report | the principal result |\n"
                "| 8. Export and fresh reload | [Engineering] | bundle exported, rebuilt in a new process, predictions compared | the reload check and the digests |\n"
                "| 9. Score new rows | [Engineering] | eight rows (or your file) scored by the rebuilt bundle | the output contract |\n"
                "| 10. Optional activity | [Concept] | a different ensemble size (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Run all, then read Sections 6, 7 and 8 and the conclusion."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **In-context learning (ICL)** | Predicting from labelled rows shown to the model at prediction time; `fit` only registers them. |\n"
                "| **Support set** | The labelled rows the model reads; here the 420-row training split. |\n"
                "| **Ensemble members (`n_estimators`)** | Passes over differently preprocessed views of the table, averaged into one prediction. |\n"
                "| **Identifier column** | A key that names a row but carries no signal; it must never be a feature. |\n"
                "| **Coded finding** | A validation result with a stable code (`TARGET_NOT_NUMERIC`, `TARGET_MISSING`, …) and the observed value. |\n"
                "| **Predictive mean** | TabPFN predicts a distribution per row; `prediction` is its mean. No interval is exported. |\n"
                "| **Training-mean baseline** | Always predict the mean target of the support rows (R² ≈ 0). |\n"
                "| **Linear regression** | Standardised numeric features plus one-hot categoricals, fitted by least squares: the classical reference. |\n"
                "| **MAE / RMSE** | Mean absolute error and root mean squared error, in target units; RMSE weighs large misses more. |\n"
                "| **R²** | 1 − MSE / variance of the target: 0 is the mean baseline, 1 is perfect. |\n"
                "| **MAPE** | Mean absolute percentage error; it divides by the target, so it explodes near zero and is omitted then. |\n"
                "| **Noise floor** | The error left when the model is exactly right: here the synthetic noise, standard deviation 0.35. |\n"
                "| **Extrapolation** | Predicting a target outside the range of the support rows' targets. |\n"
                "| **Fitted archive (`model.tabpfn_fit`)** | The fitted estimator state without the foundation weights. |\n"
                "| **Foundation checkpoint (`model.ckpt`)** | A byte copy of the pinned TabPFN-3 checkpoint, bound by its SHA-256. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n"
                "| **BYOD** | Bring Your Own Data. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Prepare the dataset: the synthetic sample or your own · [Concept]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`. The expected input is one "
                "`train.csv` with a declared numeric target column, plus optional `val.csv` and `test.csv` with identical "
                "columns (explicit splits are preserved, never re-split). With `USE_BYOD = False` the carried package draws the "
                "deterministic synthetic sample (`build_synthetic_dataset`: `4·x1 − 1.5·x2 + category effect + noise`, seed 42). With "
                "`USE_BYOD = True` the stage reads `BYOD_PATH` — a ZIP (member by member with the archive-safety rules: bare file "
                "names, expanded-size and compression-ratio ceilings, never `extractall`), a single `train.csv`, or a directory — "
                "or, in Colab with an empty path, the file you choose in the upload dialog. Without a `val.csv`, "
                "`random_holdout` draws a seeded holdout of `VALIDATION_SPLIT` — correct only for independent rows.\n\n"
                "**Identifiers are not features.** List key columns (a patient or record reference) in `DROP_COLUMNS`: they stay "
                "out of the model and beside the predictions in Section 9. A model given a unique key can memorise it; on new rows "
                "the key is meaningless. The synthetic sample has none, so the default is empty.\n\n"
                "The stage validates the table **before** any split or summary print, so a renamed target is `TARGET_MISSING` and "
                "a target read as text (`n.a.`, a unit, thousands separators such as `1,000.0`) is `TARGET_NOT_NUMERIC` naming "
                "the file — not a raw `TypeError` from a summary print. A column that is numeric except for a few stray strings is refused naming the "
                "column and the values (list it in `TEXT_COLUMNS` if it really is categorical). The stage first removes this "
                "notebook's earlier exports from `outputs/`."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "TARGET_COLUMN = 'target'  # @param {{type:\"string\"}}\n"
                "DROP_COLUMNS = []  # @param {{type:\"raw\"}}\n"
                "TEXT_COLUMNS = []  # @param {{type:\"raw\"}}\n"
                "VALIDATION_SPLIT = 0.2  # @param {{type:\"number\"}}\n\n"
                "def upload_one(what, field):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'{{field}} is empty, and the upload dialog exists only in Google Colab: set {{field}} to {{what}} in this runtime.') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received. Run this cell again and choose {{what}}, or set {{field}}.')\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'Upload exactly one file ({{what}}); got {{sorted(uploaded)}}.')\n"
                "    upload_name, payload = next(iter(uploaded.items()))\n"
                "    path = ROOT / 'inputs' / Path(upload_name).name\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    path.write_bytes(payload)\n"
                "    return str(path)\n\n"
                "byod_path = ''\n"
                "if USE_BYOD:\n"
                "    byod_path = BYOD_PATH or upload_one('one dataset ZIP or train.csv', 'BYOD_PATH')\n"
                "run_stage('data', use_byod=USE_BYOD, byod_path=byod_path, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS, text_columns=TEXT_COLUMNS, validation_split=VALIDATION_SPLIT)"
            ),
        },
        {
            "md": (
                "**What to notice:** `sample_kind: 'synthetic'`, 420 / 90 / 90 rows, the training target from −18.3 to 13.1 "
                "(mean −0.52, standard deviation 4.57), and the dataset digest."
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest · [Evaluation practice]\n\n"
                "`validate_inputs` is the package's public validation stage: unique column names, a numeric, finite, non-constant "
                "target with no missing values, at least `MIN_TRAIN_ROWS` rows, at most `MAX_FEATURES` features and "
                "`MAX_TRAIN_ROWS` rows, finite numeric features, identical schemas across splits. It returns an **input manifest** "
                "(schema, target statistics per split, numeric/categorical feature counts, the train-table digest, findings), "
                "written to `outputs/{stem}_input_manifest.json`. A `TARGET_OUT_OF_TRAINING_RANGE` warning means some validation "
                "or test targets lie outside the training targets' range: those rows test **extrapolation**, which in-context "
                "regressors do poorly; on a random split of a small table a few such rows are normal, and the stage records the "
                "explanation beside the warning. To show what rejection looks like, the stage validates a probe with the target "
                "renamed and records the package's own `TARGET_MISSING` finding. The training-mean baseline is computed on the "
                "validation rows."
            ),
            "code": "run_stage('validate')",
        },
        {
            "md": (
                "**What to notice:** two numeric features and one categorical; the `TARGET_OUT_OF_TRAINING_RANGE` warning on "
                "`val.csv` (its maximum 14.17 is above the training maximum 13.08) with its explanation; the `renamed-target-probe` "
                "finding; and the training-mean baseline — RMSE 5.01, R² ≈ 0, MAPE omitted because the targets cross zero.\n\n"
                "**Checkpoint:** why does the baseline print `mape_omitted` instead of a MAPE value?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "MAPE divides each error by its target. This target crosses zero, so a few rows have targets like 0.09; an error "
                "of 0.3 there counts as 330 %, and those rows dominate the average. In the review, a linear regression with R² "
                "0.99 showed a test MAPE of 0.79 — a contradiction created by the metric, not the model. When any target is close "
                "to zero relative to the training range, MAPE is omitted and the reason is printed.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Fit in context and evaluate · [Concept]\n\n"
                "`fit` registers the 420 support rows as context — no gradient update, the pinned checkpoint is unchanged. "
                "`evaluate` then scores the validation and test rows with **MAE** and **RMSE** (target units) and **R²**, computed "
                "by the package's `regression_metrics`; **MAPE** is reported only when no target is close to zero. Ensemble "
                "averaging is set by `N_ESTIMATORS` (default 4) and the preprocessing randomness by `SEED`.\n\n"
                "**Predict before running:** the training-mean baseline has validation RMSE 5.01 and a linear regression 0.37, "
                "close to the noise standard deviation of 0.35 (Section 7). Can TabPFN do better than the noise floor?"
            ),
            "code": (
                "N_ESTIMATORS = 4  # @param {{type:\"integer\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n"
                "run_stage('condition', n_estimators=N_ESTIMATORS, seed=SEED)"
            ),
        },
        {
            "md": (
                "**What to notice:** `mode: zero-shot-icl`, the device, the validation and test metrics (no MAPE, with the reason), "
                "and the printed adaptation note. No recorded run of this revision exists yet, so read your own numbers against "
                "Section 7.\n\n"
                "**Checkpoint:** `fit` took seconds and changed no weight. What, then, did the model learn from your rows?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Nothing was learned in the gradient sense. TabPFN was trained once, on synthetic tasks, to behave like a learning "
                "algorithm: given labelled rows and a query in the same forward pass, it outputs the predictive distribution a good "
                "learner would. `fit` stores and preprocesses your rows so each prediction can attend to them. That is why the "
                "bundle must carry the fitted state (which holds the support rows) and why quality depends entirely on them.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Baselines → evaluation report · [Evaluation practice]\n\n"
                "Two references on the same rows: the **training-mean baseline** (always the support rows' mean target) and a "
                "**standardised linear regression** with the categorical feature one-hot encoded. The synthetic target *is* linear "
                "in its features plus noise of standard deviation 0.35, so the linear model already sits at the noise floor — no "
                "model can beat it by more than chance on this table. `evaluation_report` is the package's public evaluation "
                "stage: verdict `sample-sanity` (one holdout of one synthetic table, no dispersion estimate), the metric entries, "
                "the baseline, the caveats, the MAPE rule and the proposed RUN7 deviation, written to "
                "`outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** if TabPFN's validation RMSE is 0.40 and the linear model's 0.37, what do you report?"
            ),
            "code": "run_stage('report')",
        },
        {
            "md": (
                "**What to notice:** the linear reference — validation MAE 0.298 / RMSE 0.366 / R² 0.9946, test 0.266 / 0.339 / "
                "0.9943 — beside TabPFN's line from your run, `mape_omitted`, and the interpretation line.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "That both models are at the noise floor and the difference is noise: with 90 rows, RMSE moves by a few hundredths "
                "between splits. The honest lesson of this table is that a flexible foundation model is not needed when the truth "
                "is linear; a BYOD table with non-linear structure is where the comparison becomes informative.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Export the artifact bundle and verify it in a fresh process · [Engineering]\n\n"
                "`save_artifact` writes `model.tabpfn_fit` (the fitted estimator state, without the foundation weights), "
                "`model.ckpt` (a byte copy of the verified checkpoint) and `artifact_manifest.json` (target and feature columns, "
                "training-target statistics, both SHA-256 digests, the base-model identity, the ensemble settings, and the feature kinds the "
                "companion uses to refuse text in numeric columns); `zip_artifact_bundle` writes "
                "`outputs/{stem}_artifact.zip`. The `export` stage prints the ZIP and fitted-archive digests — the **trusted "
                "digests** to give the companion notebook.\n\n"
                "The `reload` stage runs in a **fresh process**: it clears and fills `outputs/artifact-reload/`, checks the bundle "
                "with `validate_artifact_bundle` (manifest schema, member names, sizes, digests, the checkpoint equal to the pinned "
                "one), rebuilds the estimator with `from_artifact` (no refit, no download), and requires its predictions on "
                "**every** validation row to equal the exporting process's within `rtol=1e-5`, `atol=1e-6`, recording the largest "
                "difference. Comparing predictions row by row is a much stronger check than comparing one scalar such as MAE, "
                "which two different models can share.\n\n"
                "**Predict before running:** if the reload reproduced the validation MAE exactly but one prediction differed by "
                "0.01, would this check pass?"
            ),
            "code": "run_stage('export')\nrun_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** the three bundle members, the two digests and the ZIP digest; then `maxAbsPredictionDifference` "
                "(0 or a few 1e-8 on the same device) and the `PASSED` line. Re-running this cell alone works: the reload directory "
                "is cleared first.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "No. A 0.01 difference is far above `atol=1e-6`, so the check fails even if the MAE matched — exactly the silent "
                "drift a one-scalar comparison would miss. A bundle that cannot reproduce its own predictions should not be "
                "shipped.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 9. Score new rows with the rebuilt bundle · [Engineering]\n\n"
                "By default the `predict` stage takes the first eight rows of `test.csv` (held out from the support), removes the "
                "target, writes them to `outputs/{stem}_new_rows.csv` — the companion notebook's input — and scores them with the "
                "estimator rebuilt from the bundle. Tick `USE_BYOD_ROWS` and set `NEW_DATA_PATH` (or, in Colab, leave it empty to "
                "upload) to score your own unlabelled CSV. `validate_new_rows` requires exactly the fitted feature columns (any "
                "order) and no target or `prediction` column; identifier columns listed in `ID_COLUMNS` (or `DROP_COLUMNS`) are "
                "kept beside the predictions, so the output joins back on your own key; an undeclared extra column is refused with "
                "a hint. The output adds `prediction`, the point estimate in target units, with no interval. `outputs/{stem}_result.json` records predictions, metrics, baselines, the reload check, the "
                "input manifest, the dataset and bundle digests, the notebook's source, the model identity and licence, and the "
                "runtime."
            ),
            "code": (
                "USE_BYOD_ROWS = False  # @param {{type:\"boolean\"}}\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "ID_COLUMNS = []  # @param {{type:\"raw\"}}\n\n"
                "new_data_file = ''\n"
                "if USE_BYOD_ROWS:\n"
                "    new_data_file = NEW_DATA_PATH or upload_one('one unlabelled CSV', 'NEW_DATA_PATH')\n"
                "run_stage('predict', new_data_path=new_data_file, id_columns=ID_COLUMNS)"
            ),
        },
        {
            "md": (
                "**What to notice:** eight rows with `x1`, `x2`, `category` and `prediction`."
            ),
        },
        {
            "md": (
                "## 10. Optional activity: how much does the ensemble size matter? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** with `ACTIVITY_N_ESTIMATORS = 1` instead of 4, will "
                "validation RMSE change by more than a few hundredths, and how far will individual predictions move? **Change:** "
                "tick `RUN_ACTIVITY` (try 1, then 8). **Run** this cell. **Observe** the canonical and changed validation metrics "
                "and `max_abs_prediction_change`. **Explain** what the ensemble buys. The activity refits in "
                "context, writes only to `outputs/activity/`, and stops if any canonical output changed."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_N_ESTIMATORS = 1  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', n_estimators=ACTIVITY_N_ESTIMATORS)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** the two settings, both metric sets and the largest prediction change; "
                "`canonical_outputs_unchanged: True`.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "Each ensemble member sees the table through a different preprocessing and feature order; averaging them steadies "
                "the predictions. On a table at the noise floor the RMSE barely moves — the residual is noise either way — while "
                "individual predictions can shift by a fraction of the noise standard deviation.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel. |\n"
        "| `Not enough free disk` | the isolated environment needs about 8 GB | Start a fresh runtime; an environment built from the same lock is reused. |\n"
        "| `Carried file integrity failure` | a carried file was edited in the notebook | Open a fresh copy from the repository. |\n"
        "| `uv … mismatch`, `URLError`, or `CalledProcessError` from `uv` | network or a transient PyPI error | Re-run the Section 2 install cell. Never remove a pin or a hash. |\n"
        "| `The run directory … has no carried files, or the isolated environment is gone` | Section 1 run with `NEW_RUN_DIRECTORY` ticked | Run Sections 1–3 again, or *Run all*. |\n"
        "| `RuntimeError: Stage '…' failed (exit 1): …` | the stage's own error follows the colon | Find it below; fix the cause and re-run from that cell. |\n"
        "| A Hub download error in Section 3, or `… sha256 … != manifest` | a transient failure or a corrupted download | Re-run Section 3; delete the partial file under `weights/` if the digest fails. Never edit the manifest. |\n"
        "| `[TARGET_MISSING] … target column … not present` | your label column has another name | Set `TARGET_COLUMN`. |\n"
        "| `[TARGET_NOT_NUMERIC]` | text in the target (`n.a.`, units, thousands separators) | Fix the values; remove separators (1510.0, not 1,510.0). |\n"
        "| `[TARGET_MISSING_VALUES]`, `[TARGET_NON_FINITE]`, `[CONSTANT_TARGET]`, `[TOO_FEW_ROWS]` | the target breaks the contract | Fill or remove the rows; supply a varying target. |\n"
        "| `[DUPLICATE_COLUMNS]`, `[RESERVED_COLUMNS]`, `[SCHEMA_MISMATCH]` | the table breaks the input contract | Fix the named columns. |\n"
        "| `TARGET_OUT_OF_TRAINING_RANGE` (a warning) | evaluation targets outside the training range | Expected for a few rows on a random split; extrapolated rows are scored. |\n"
        "| `column … is numeric except for N value(s)` | stray text in a numeric column | Fix the values, or list the column in `TEXT_COLUMNS`. |\n"
        "| `DROP_COLUMNS … are not in the header` | a typo in `DROP_COLUMNS` | Use the exact column names. |\n"
        "| `BYOD_PATH … does not exist`, `BYOD_PATH is empty, and the upload dialog exists only in Google Colab`, `The upload was cancelled or empty` | no file supplied | Set the path, or run the cell again and choose a file. |\n"
        "| `The reloaded artifact does not reproduce the exporting process's predictions` | a corrupted bundle or a changed runtime | Re-run Sections 6–8; do not ship the bundle. |\n"
        "| `new rows: [SCHEMA_MISMATCH] … If [...] are identifiers, list them in ID_COLUMNS` | your rows carry an identifier or lack a feature | List identifiers in `ID_COLUMNS`; supply exactly the fitted features. |\n\n"
        "## Interpretation and limits\n\n"
        "`prediction` is TabPFN's point estimate (the mean of its predictive distribution) with no interval and no shipped "
        "tolerance band. The evaluation report's `sample-sanity` verdict names what it is: one holdout of one synthetic table "
        "with no dispersion estimate — a plumbing check that must not be generalised. The synthetic target is linear plus "
        "noise, so a linear regression already reaches validation RMSE 0.37 against a noise floor of 0.35; any gap between "
        "TabPFN and it is noise here. MAPE is omitted when targets come close to zero, because it would contradict R². Random "
        "splitting assumes independent rows; temporal, grouped or patient-level data need leakage-safe splits you supply. "
        "In-context learning is not fine-tuning: quality depends entirely on the support rows, the bundle carries them, and "
        "targets outside their range are extrapolations. Digest equality proves the checkpoint bytes are the ones pinned at "
        "the immutable revision; it does not by itself prove who published them.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire "
        "and digest-verify the pinned TabPFN-3 checkpoint, validate the demonstrated table into an input manifest with coded "
        "refusals, fit in context, compute sample metrics against a trivial and a classical baseline, export the artifact "
        "bundle, rebuild equivalent predictions from it in a fresh process, score new rows with identifiers kept, and emit "
        "the shown machine-readable outputs — without the repository being reachable. It does **not** establish benchmark "
        "superiority, generalisation, fairness, robustness, calibrated uncertainty, safety for high-consequence decisions, "
        "production fitness, or anything about fine-tuned TabPFN models.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three to five sentences, using the numbers your run printed:\n\n"
        "1. **Result:** TabPFN's validation and test MAE, RMSE and R² beside the training mean and the linear regression.\n"
        "2. **Reading:** what the noise floor of 0.35 allows you to claim, and why MAPE was omitted.\n"
        "3. **Reuse:** what the Section 8 reload proved, and why predictions were compared row by row.\n"
        "4. **Limits:** the one limitation you would fix first (for example repeated splits, real non-linear data, intervals).\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "On the 90-row validation and 90-row test splits of the synthetic table, TabPFN-3 conditioned in context on 420 rows "
        "scored the MAE, RMSE and R² printed in Sections 6 and 7, against RMSE 5.0 for the training mean and 0.37 / 0.34 for "
        "a linear regression, which already sits at the noise floor of 0.35 because the target is linear. Any difference "
        "between TabPFN and the linear model is noise on this table, so the run demonstrates the workflow, not an advantage. "
        "MAPE was omitted because the targets cross zero. The exported bundle rebuilt identical predictions in a fresh "
        "process. Before any use I would evaluate on repeated splits of a real, non-linear table and add prediction "
        "intervals — and clear the non-commercial licence.\n\n"
        "</details>\n\n"
        "**Next experiments:** run the activity with 1 and 8 estimators; enable `USE_BYOD` with a small non-linear table of "
        "your own and compare TabPFN with the linear reference there; drop `val.csv` from your ZIP and watch the seeded "
        "holdout take over; hand `outputs/{stem}_artifact.zip`, its printed digests and `outputs/{stem}_new_rows.csv` to the "
        "companion artifact-inference notebook in a fresh session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/PriorLabs/TabPFN\n"
        "- TabPFN-3 technical report: https://arxiv.org/abs/2605.13986"
    ),
}

"""Stage runner for the standalone TabPFN-3 regressor E2E tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process and starts from files only: the
verified snapshot under ``--weights``, the partitions and records of earlier stages under ``RUN_DIR/state`` (tables as
pandas ``orient="table"`` JSON, so no Parquet engine is needed), and the learner-facing exports under ``--outputs``.
In-context conditioning (``fit``) registers the support rows and is repeated by every stage that needs the model. On
failure a stage writes ``RUN_DIR/state/<stage>.error.json``, which the notebook re-raises in the kernel.

Stages: weights → data → validate → condition → report → export → reload → predict, plus the optional ``activity``.
The ``data`` and ``validate`` stages import no model library, so CI exercises them directly.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

STEM = "tabpfn_regressor"
PACKAGE = "tabpfn_regressor_pipeline"
SEED = 42
NEW_ROWS = 8
RELOAD_RTOL, RELOAD_ATOL = 1e-5, 1e-6
DEFAULT_DROP_COLUMNS: list[str] = []  # the synthetic sample has no identifier column; list yours (TPR-S2)
MAPE_FLOOR = 0.05  # MAPE is omitted when any evaluation target is within 5 % of the training range of zero (TPR-m2)
NUMERIC_LIKE_SHARE = 0.9  # an object column whose values are >= 90 % numbers is treated as a numeric column with typos
RUN7_DEVIATION = (
    "in-context conditioning is this tutorial's adaptation; the production pipeline's gradient fine-tuning "
    "(Prior Labs' FinetunedTabPFNRegressor, default on, large-GPU) is not run here — the same fallback production "
    "uses without a large GPU. Proposed RUN7 deviation, pending maintainer approval."
)


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_frame(self, name: str, frame) -> None:
        (self.state / name).write_text(frame.to_json(orient="table", index=False), encoding="utf-8")

    def read_frame(self, name: str, needed_by: str):
        import pandas as pd

        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return pd.read_json(io.StringIO(path.read_text(encoding="utf-8")), orient="table")

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def coded(P, exc: Exception, name: str) -> ValueError:
    """An InputRejected re-raised with its code, the file and the observed value, so the kernel shows the rule."""
    finding = getattr(exc, "finding", None)
    if not finding:
        return ValueError(f"{name}: {exc}")
    observed = finding.get("observed")
    hint = " Set TARGET_COLUMN to the name of your label column." if finding.get("code") == "TARGET_MISSING" else ""
    if finding.get("code") == "TARGET_NOT_NUMERIC":
        hint = " A regression target must be numeric: fix text values such as 'n.a.' (leave the cell empty only if you mean to drop the row) and remove thousands separators (1510.0, not 1,510.0)."
    return ValueError(f"{name}: [{finding['code']}] {finding['message']} (observed: {observed}).{hint}")


# --------------------------------------------------------------------------------------------------
# data (TPR-m3, TPR-S2)
# --------------------------------------------------------------------------------------------------


def check_numeric_like(frame, name: str, skip: list[str]) -> None:
    """Refuse a column that is numeric except for a few stray strings, naming the file, the column and the values."""
    import pandas as pd

    for column in frame.columns:
        if column in skip or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        numeric = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
        if float(numeric.notna().mean()) >= NUMERIC_LIKE_SHARE:
            bad = values[numeric.isna()].astype(str).unique().tolist()[:5]
            raise ValueError(f"{name}: column {column!r} is numeric except for {int(numeric.isna().sum())} value(s) {bad}; it would be treated as a category and lose its numeric meaning. Fix those values (leave missing cells empty), or list the column in TEXT_COLUMNS if it really is categorical.")


def clear_outputs(run: Run) -> list[str]:
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def read_byod(P, path: Path) -> dict[str, Any]:
    if not str(path) or not path.exists():
        raise FileNotFoundError(f"BYOD_PATH {str(path)!r} does not exist in this runtime: upload the file or correct the path.")
    if path.is_dir():
        return P.read_dataset_dir(path)
    if path.suffix.lower() == ".zip":
        return P.read_dataset_zip(path)
    if path.suffix.lower() == ".csv":
        import pandas as pd

        return {"train.csv": pd.read_csv(path)}
    raise ValueError(f"{path.name}: BYOD must be a dataset ZIP (train.csv, optional val.csv/test.csv), a train.csv, or a directory holding them.")


def stage_weights(run: Run) -> None:
    P = package(run.root)
    snapshot = run.weights / P.MODEL_KEY
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / P.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / P.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified_files": len(verified["files"]), "revision": verified["revision"], "checkpoint": P.WEIGHTS_FILE, "sha256": P.WEIGHTS_SHA256})
    run.write_state("weights.json", {"snapshot": str(snapshot), "checkpoint": str(snapshot / P.WEIGHTS_FILE)})


def stage_data(run: Run) -> None:
    P = package(run.root)
    opts = run.options
    use_byod = bool(opts.get("use_byod", False))
    target = opts.get("target_column") or "target"
    drop = list(opts["drop_columns"]) if opts.get("drop_columns") is not None else list(DEFAULT_DROP_COLUMNS)
    text_columns = list(opts.get("text_columns") or [])
    split = float(opts.get("validation_split", P.DEFAULT_VALIDATION_SPLIT))
    removed = clear_outputs(run)
    for name in ("train.json", "val.json", "test.json", "data.json", "validated.json", "condition.json", "export_reference.json"):
        (run.state / name).unlink(missing_ok=True)
    if removed:
        print({"removed_previous_outputs": removed})
    if use_byod:
        path = Path(opts.get("byod_path") or "")
        frames = read_byod(P, path)
        origin = {"type": "user-supplied dataset (BYOD)", "file": path.name}
        kind, digest_source = "BYOD", path
    else:
        zip_path = P.build_synthetic_dataset(run.state / "synthetic.zip", rows=600, seed=SEED)
        frames = P.read_dataset_zip(zip_path)
        origin = {"type": "deterministic synthetic tutorial sample", "generator": "build_synthetic_dataset", "rows": 600, "features": 3, "seed": SEED}
        kind, digest_source = "synthetic", zip_path
    header = list(map(str, frames["train.csv"].columns))
    unknown = [c for c in drop if c not in header and c not in DEFAULT_DROP_COLUMNS]
    if unknown:
        raise ValueError(f"train.csv: DROP_COLUMNS {unknown} are not in the header {header}.")
    drop = [c for c in drop if c in header]  # the default identifiers are simply absent from most BYOD tables
    if target in drop:
        raise ValueError(f"DROP_COLUMNS must not contain the target column {target!r}.")
    for name, frame in frames.items():
        check_numeric_like(frame, name, [target, *drop, *text_columns])
    train, val, test = frames["train.csv"], frames.get("val.csv"), frames.get("test.csv")
    features_only = lambda f: f.drop(columns=[c for c in drop if c in f.columns]) if f is not None else None  # noqa: E731
    # Validate BEFORE any split or summary print so a text target is TARGET_NOT_NUMERIC, not a TypeError (TPR-m3).
    try:
        P.validate_inputs(features_only(train), target, val=features_only(val), test=features_only(test), names=[origin["type"]])
    except P.InputRejected as exc:
        raise coded(P, exc, "dataset") from None
    split_origin = "explicit train/val/test from the dataset"
    if val is None:
        train, val = P.random_holdout(train, target, split, seed=SEED)
        split_origin = f"seeded random holdout of {split} drawn from train.csv (independent rows assumed)"
    for key, frame in (("train", train), ("val", val), ("test", test)):
        if frame is not None:
            run.write_frame(f"{key}.json", frame.reset_index(drop=True))
    record = {"sample_kind": kind, **origin, "dataset_sha256": sha256_file(digest_source) if digest_source.is_file() else None, "target": target, "drop_columns": drop, "text_columns": text_columns, "splits": split_origin, "rows": {"train": len(train), "val": len(val), "test": None if test is None else len(test)}}
    run.write_state("data.json", record)
    print(record)
    print("training target:", rounded(train[target].describe()[["count", "mean", "std", "min", "max"]].to_dict(), 3))
    if drop:
        print(f"DROP_COLUMNS {drop}: kept beside the predictions as identifiers, never given to the model.")


def stage_validate(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "validate")
    target, drop = data["target"], data["drop_columns"]
    train = run.read_frame("train.json", "validate")
    val = run.read_frame("val.json", "validate")
    test = run.read_frame("test.json", "validate") if (run.state / "test.json").is_file() else None
    strip = lambda f: f.drop(columns=drop) if f is not None else None  # noqa: E731
    print({"ceilings": {"MAX_TRAIN_ROWS": P.MAX_TRAIN_ROWS, "MAX_FEATURES": P.MAX_FEATURES, "MIN_TRAIN_ROWS": P.MIN_TRAIN_ROWS}, "decision_rule": P.DECISION_RULE, "model_version": P.MODEL_VERSION})
    try:
        manifest = P.validate_inputs(strip(train), target, val=strip(val), test=strip(test), names=[data["type"]])
    except P.InputRejected as exc:
        raise coded(P, exc, "dataset") from None
    for finding in manifest["findings"]:
        if finding.get("code") == "TARGET_OUT_OF_TRAINING_RANGE":
            finding["explanation"] = "some evaluation targets lie outside the training targets' range, so those rows test extrapolation; on a random split of a small table a few such rows are expected"
    try:
        P.validate_inputs(strip(train).rename(columns={target: "label"}), target)
    except P.InputRejected as exc:
        manifest["findings"].append({"input": "renamed-target-probe", **exc.finding})
    entry = manifest["inputs"][0]
    entry["drop_columns"] = drop
    run.write_output(f"{STEM}_input_manifest.json", manifest)
    features = entry["feature_columns"]
    baseline = mape_policy(P.mean_baseline(train[target], val[target]), val[target], train[target])
    run.write_state("validated.json", {"features": features, "baseline": baseline})
    print(json.dumps(rounded({k: entry[k] for k in ("id", "mode", "target_column", "numeric_features", "categorical_features", "target_stats", "splits", "drop_columns")}, 3), indent=2))
    print("findings:", json.dumps(manifest["findings"], indent=2, default=str))
    print("training-mean baseline on val.csv", rounded(baseline))


# --------------------------------------------------------------------------------------------------
# model stages
# --------------------------------------------------------------------------------------------------


def mape_policy(metrics: dict[str, Any], y_eval, y_train) -> dict[str, Any]:
    """TPR-m2: MAPE divides by the target, so near zero it explodes. It is dropped (with the reason) when any evaluation
    target lies within MAPE_FLOOR of the training range of zero, e.g. when the targets straddle zero."""
    import numpy as np

    y = np.asarray(y_eval, dtype=float)
    span = float(np.ptp(np.asarray(y_train, dtype=float))) or 1.0
    out = dict(metrics)
    if "mape" in out and float(np.min(np.abs(y))) < MAPE_FLOOR * span:
        out.pop("mape")
        out["mape_omitted"] = f"some targets are within {MAPE_FLOOR:.0%} of the training range of zero (min |y| = {float(np.min(np.abs(y))):.3g}); MAPE would be dominated by those rows"
    return out


def tables(run: Run, needed_by: str):
    data = run.read_state("data.json", needed_by)
    val_state = run.read_state("validated.json", needed_by)
    train = run.read_frame("train.json", needed_by)
    val = run.read_frame("val.json", needed_by)
    test = run.read_frame("test.json", needed_by) if (run.state / "test.json").is_file() else None
    return data, val_state, train, val, test


def fitted(run: Run, train, data, features, n_estimators: int | None = None):
    P = package(run.root)
    opts = run.read_state("condition.json", "model")["settings"] if (run.state / "condition.json").is_file() and n_estimators is None else {}
    n = int(n_estimators or opts.get("n_estimators", run.options.get("n_estimators", P.DEFAULT_N_ESTIMATORS)))
    seed = int(opts.get("seed", run.options.get("seed", SEED)))
    pipe = P.TabPFNRegressorPipeline.from_pretrained(weights_dir=run.weights / P.MODEL_KEY, n_estimators=n, random_state=seed)
    pipe.fit(train[features], train[data["target"]], target_column=data["target"])
    return pipe


def stage_condition(run: Run) -> None:
    P = package(run.root)
    data, v, train, val, test = tables(run, "condition")
    n = int(run.options.get("n_estimators", P.DEFAULT_N_ESTIMATORS))
    seed = int(run.options.get("seed", SEED))
    if not 1 <= n <= 32:
        raise ValueError(f"N_ESTIMATORS must be from 1 to 32, got {n}")
    (run.state / "condition.json").unlink(missing_ok=True)
    pipe = fitted(run, train, data, v["features"], n_estimators=n)
    target = data["target"]
    metrics = mape_policy(pipe.evaluate(val[v["features"]], val[target]), val[target], train[target])
    test_metrics = None if test is None else mape_policy(pipe.evaluate(test[v["features"]], test[target]), test[target], train[target])
    record = {"settings": {"n_estimators": n, "seed": seed}, "validation": metrics, "test": test_metrics, "device": pipe.device, "adaptation": "zero-shot-icl (in-context conditioning, no gradient update)", "run7_deviation": RUN7_DEVIATION}
    run.write_state("condition.json", record)
    print(json.dumps(rounded({"mode": record["adaptation"], "device": pipe.device, "n_estimators": n, "seed": seed, "validation": metrics, "test": test_metrics}), indent=2))
    print("Adaptation:", RUN7_DEVIATION)


def classical_reference(train, val, test, target: str, features: list[str]) -> dict[str, Any]:
    """Standardised linear regression with one-hot categoricals (TPR-m2): on a linear table it sits near the noise floor."""
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    prep = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
    model = make_pipeline(prep, LinearRegression()).fit(train[features].fillna(train[numeric].mean()), train[target].astype(float))

    def score(frame):
        pred = model.predict(frame[features].fillna(train[numeric].mean()))
        y = frame[target].to_numpy(dtype=float)
        err = y - pred
        return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2))), "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2))}

    out = {"id": "standardised_linear_regression", "features": "numeric standardised, categoricals one-hot", "validation": score(val)}
    if test is not None:
        out["test"] = score(test)
    return out


def stage_report(run: Run) -> None:
    P = package(run.root)
    data, v, train, val, test = tables(run, "report")
    cond = run.read_state("condition.json", "report")
    target = data["target"]
    reference = classical_reference(train, val, test, target, v["features"])
    report = P.evaluation_report(cond["validation"], baseline=v["baseline"], n_validation=len(val), target_column=target, sample_kind=data["sample_kind"])
    report["test_metrics"] = cond["test"]
    report["classical_reference"] = reference
    if "mape_omitted" in cond["validation"]:
        report["mape_omitted"] = cond["validation"]["mape_omitted"]
    report["adaptation_deviation"] = RUN7_DEVIATION
    std = float(train[target].std())
    report["interpretation"] = f"{len(val)} validation rows; compare RMSE with the training target's standard deviation ({std:.3g}) and with the linear reference. A gap smaller than the split-to-split noise of one holdout is not evidence; no dispersion over splits was measured."
    run.write_output(f"{STEM}_evaluation_report.json", report)
    print(json.dumps(rounded({k: report.get(k) for k in ("verdict", "reason", "decision_rule", "n_validation", "classical_reference", "mape_omitted", "interpretation")}), indent=2))
    if cond["validation"]["rmse"] > v["baseline"]["rmse"]:
        print("WARNING: in-context TabPFN does not beat the training-mean baseline on this holdout; inspect the data before drawing any conclusion.")


def stage_export(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    data, v, train, val, _test = tables(run, "export")
    pipe = fitted(run, train, data, v["features"])
    art = run.out / "artifact"
    shutil.rmtree(art, ignore_errors=True)
    manifest = pipe.save_artifact(art)
    import pandas as pd

    # Record which features the estimator saw as numeric, so the companion can refuse text in them (TPCA-m2).
    manifest["featureKinds"] = {c: ("numeric" if pd.api.types.is_numeric_dtype(train[c]) else "categorical") for c in v["features"]}
    (art / P.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    zip_path = run.out / f"{STEM}_artifact.zip"
    zip_sha = P.zip_artifact_bundle(art, zip_path)
    if manifest["foundationCheckpointSha256"] != P.WEIGHTS_SHA256:
        raise RuntimeError("the bundled checkpoint is not the pinned foundation checkpoint")
    run.write_state("export_reference.json", {"predictions": np.asarray(pipe.predict_values(val[v["features"]]), dtype=float).tolist(), "mae": run.read_state("condition.json", "export")["validation"]["mae"], "device": pipe.device})
    record = {"artifact": sorted(p.name for p in art.iterdir()), "fittedEstimatorSha256": manifest["fittedEstimatorSha256"], "foundationCheckpointSha256": manifest["foundationCheckpointSha256"], "bundle_zip": zip_path.name, "bundle_zip_sha256": zip_sha}
    print(record)
    print(f"Trusted digests for the companion notebook: EXPECTED_ZIP_SHA256 = '{zip_sha}', EXPECTED_FITTED_SHA256 = '{manifest['fittedEstimatorSha256']}'.")
    run.write_state("export.json", record)


def stage_reload(run: Run) -> None:
    """TPR-m4: a fresh process rebuilds the estimator from the bundle and its predictions must equal the exporting
    process's on every validation row within an explicit tolerance; the fresh directory is cleared first."""
    import numpy as np

    P = package(run.root)
    data, v, _train, val, _test = tables(run, "reload")
    reference = run.read_state("export_reference.json", "reload")
    fresh_dir = run.out / "artifact-reload"
    shutil.rmtree(fresh_dir, ignore_errors=True)
    shutil.copytree(run.out / "artifact", fresh_dir)
    checked = P.validate_artifact_bundle(fresh_dir, expected_checkpoint_sha256=P.WEIGHTS_SHA256)
    fresh = P.TabPFNRegressorPipeline.from_artifact(fresh_dir, device=reference["device"], expected_checkpoint_sha256=P.WEIGHTS_SHA256)
    if fresh.feature_columns != v["features"]:
        raise RuntimeError("reloaded artifact disagrees with the validated schema")
    preds = np.asarray(fresh.predict_values(val[v["features"]]), dtype=float)
    expected = np.asarray(reference["predictions"], dtype=float)
    diff = float(np.max(np.abs(preds - expected)))
    reload_check = {"rows": int(len(val)), "maxAbsPredictionDifference": diff, "rtol": RELOAD_RTOL, "atol": RELOAD_ATOL, "recordedMae": reference["mae"], "recordedModelPath": checked["recordedModelPath"], "process": "fresh (separate from the exporting stage)"}
    np.testing.assert_allclose(preds, expected, rtol=RELOAD_RTOL, atol=RELOAD_ATOL, err_msg="The reloaded artifact does not reproduce the exporting process's predictions. Do not ship this artifact.")
    report_path = run.out / f"{STEM}_evaluation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["reload_check"] = reload_check
        run.write_output(report_path.name, report)
    run.write_state("reload.json", reload_check)
    print(json.dumps(rounded(reload_check, 8), indent=2))
    print(f"Fresh-boundary verification PASSED: predictions equivalent on {len(val)} validation rows (rtol={RELOAD_RTOL}, atol={RELOAD_ATOL}).")


def new_rows_frame(data: dict, val, test):
    """The first eight rows of test.csv (else val.csv) without the target (TPR-S1)."""
    source, label = (test, "test.csv") if test is not None else (val, "val.csv")
    rows = source.drop(columns=[data["target"]]).head(NEW_ROWS).reset_index(drop=True)
    return rows, f"first {NEW_ROWS} rows of {label} (held out from the in-context support), target removed"


def stage_predict(run: Run) -> None:
    import importlib.metadata
    import platform

    import pandas as pd

    P = package(run.root)
    data, v, train, val, test = tables(run, "predict")
    cond = run.read_state("condition.json", "predict")
    reload_check = run.read_state("reload.json", "predict")
    export = run.read_state("export.json", "predict")
    fresh = P.TabPFNRegressorPipeline.from_artifact(run.out / "artifact", device=cond["device"], expected_checkpoint_sha256=P.WEIGHTS_SHA256)
    path = run.options.get("new_data_path") or ""
    drop = list(dict.fromkeys([*data["drop_columns"], *(run.options.get("id_columns") or [])]))
    if path:
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"NEW_DATA_PATH {path!r} does not exist in this runtime.")
        rows = pd.read_csv(file)
        origin = f"user-supplied CSV (BYOD): {file.name}"
    else:
        rows, origin = new_rows_frame(data, val, test)
        rows.to_csv(run.out / f"{STEM}_new_rows.csv", index=False)
    passthrough = [c for c in rows.columns if c in drop]
    try:
        features = P.validate_new_rows(rows.drop(columns=passthrough), v["features"], target_column=data["target"])
    except P.InputRejected as exc:
        error = coded(P, exc, Path(path).name if path else "new rows")
        extra = (exc.finding.get("observed") or {}).get("extra") if isinstance(exc.finding.get("observed"), dict) else None
        raise ValueError(f"{error} If {extra} are identifiers, list them in ID_COLUMNS.") if extra else error from None
    check_numeric_like(features, Path(path).name if path else "new rows", [])
    scored = fresh.predict(features)
    out = pd.concat([rows[passthrough].reset_index(drop=True), scored.drop(columns=["row_id"]).reset_index(drop=True)], axis=1) if passthrough else scored
    out.to_csv(run.out / f"{STEM}_predictions.csv", index=False)
    report = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    import torch

    payload = {
        "predictions": out.to_dict(orient="records"),
        "metrics": {"validation": cond["validation"], "test": cond["test"]},
        "training_mean_baseline": v["baseline"],
        "evaluation_report": report,
        "fresh_boundary_reload": reload_check,
        "input_manifest": manifest,
        "dataset": {k: data[k] for k in ("sample_kind", "type", "dataset_sha256", "target", "drop_columns", "splits", "rows")} | {"feature_columns": v["features"]},
        "artifact": export,
        "inference": {"mode": "zero-shot-icl", "adaptation": cond["run7_deviation"], **cond["settings"], "decision_rule": P.DECISION_RULE, "new_rows_origin": origin, "scored_rows": int(len(out)), "identifier_columns": passthrough},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "model_file": P.WEIGHTS_FILE,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "tabpfn": importlib.metadata.version("tabpfn"), "pandas": pd.__version__, "scikit_learn": importlib.metadata.version("scikit-learn"), "device": fresh.device},
    }
    run.write_output(f"{STEM}_result.json", payload)
    print("scored rows drawn from:", origin)
    if not path:
        print(f"Their unlabelled copy, with {passthrough or 'no'} identifier column(s), is outputs/{STEM}_new_rows.csv: the companion notebook's input.")
    print(out.round(4).to_string(index=False))


def stage_activity(run: Run) -> None:
    """Optional activity: refit with a different ensemble size and compare validation metrics and predictions with
    the canonical run; writes only to outputs/activity/."""
    import numpy as np

    package(run.root)
    data, v, train, val, _test = tables(run, "activity")
    n = int(run.options.get("n_estimators", 1))
    if not 1 <= n <= 32:
        raise ValueError(f"ACTIVITY_N_ESTIMATORS must be from 1 to 32, got {n}")
    cond = run.read_state("condition.json", "activity")
    before = {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()}
    changed = fitted(run, train, data, v["features"], n_estimators=n)
    metrics = mape_policy(changed.evaluate(val[v["features"]], val[data["target"]]), val[data["target"]], train[data["target"]])
    reference = run.read_state("export_reference.json", "activity") if (run.state / "export_reference.json").is_file() else None
    record = {"changed": {"n_estimators": [cond["settings"]["n_estimators"], n]}, "validation": {"canonical": cond["validation"], "changed": metrics}, "validation_rows": len(val)}
    if reference is not None:
        diff = np.abs(np.asarray(changed.predict_values(val[v["features"]]), dtype=float) - np.asarray(reference["predictions"], dtype=float))
        record["max_abs_prediction_change"] = float(diff.max())
        record["mean_abs_prediction_change"] = float(diff.mean())
    run.write_output(f"activity/{STEM}_activity_n_estimators_{n}.json", record)
    if {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()} != before:
        raise RuntimeError("The activity changed a canonical output; it must write only to outputs/activity/.")
    print(json.dumps(rounded(record), indent=2))
    print({"canonical_outputs_unchanged": True})


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "validate": stage_validate,
    "condition": stage_condition,
    "report": stage_report,
    "export": stage_export,
    "reload": stage_reload,
    "predict": stage_predict,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

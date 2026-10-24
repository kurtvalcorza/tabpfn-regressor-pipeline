"""TabPFN-3 tabular regression — public tutorial API (DIMER pipeline, standalone-notebook carrier).

Inference-only: the pinned TabPFN-3 regressor checkpoint conditions on the labelled training rows **in context**
(``fit`` registers the support set; no gradient update) and predicts query rows in a forward pass. The DIMER
fine-tuning path of this pipeline lives in the private ``tabpfn-regressor-finetuner`` worker and is NOT carried
here (private code; TabPFN-3 weights are non-commercial). This module owns the pinned snapshot scheme, the input
contract (DAT24), the ICL fit / predict / evaluate calls, the ``model.tabpfn_fit`` + ``model.safetensors`` +
``artifact_manifest.json`` bundle (format v2) the serving path consumes, its pre-load validation, and the evaluation report
(EVAL21). ``tabpfn`` / ``torch`` are imported lazily inside the functions that need them.

Serving reads only safetensors: ``from_pretrained`` converts the digest-checked pinned ``.ckpt`` once
(``torch.load(weights_only=True)``) into ``CONVERTED_FILE``, checks it against ``CONVERTED_SHA256``, and the estimator
is built from that file, which tabpfn 8.1.0 opens with ``safetensors.safe_open`` (no ``torch.load``). Once the
converted file is in place the source ``.ckpt`` is no longer needed. Bundles carry that file as ``model.safetensors``
(format v2); v1 bundles (``model.ckpt``) stay readable, ``weights_only=True`` only, with a ``DeprecationWarning``.
"""
# ruff: noqa: E501  -- contract dictionaries and messages are kept on single lines
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from . import safetensors_weights as _sw

MODEL_ID = "Prior-Labs/tabpfn_3"
MODEL_REVISION = "24a16a89d245878b846555110985634aa2e656d7"
# Non-commercial licence: testing, evaluation and internal benchmarking only (no production deployment).
MODEL_LICENSE = "tabpfn-3-license-v1.0"
MODEL_KEY = "tabpfn-3-regressor"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "tabpfn-v3-regressor-v3_default.ckpt"
WEIGHTS_SHA256 = "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"
WEIGHTS_BYTES = 233289807
# The served file: WEIGHTS_FILE converted by safetensors_weights.convert_checkpoint (tabpfn's own safetensors layout,
# sorted header). Byte-reproducible; the digest below is what two independent conversions produced.
CONVERTED_FILE = "tabpfn-v3-regressor-v3_default.safetensors"
CONVERTED_SHA256 = "edff9a12e3ccd8c0d3ad11f6fdf5b5fe14e36e5a23025049f3bd9f3e4c4e332c"
CONVERTED_BYTES = 233186068
MODEL_VERSION = "v3"
TABPFN_VERSION = "8.1.0"

TASK_TYPE = "tabular_regression"
# Ceilings of the selected generation (contract/model-versions.json, `v3`): rows, features (maxClasses is unused by regression).
MAX_TRAIN_ROWS = 1_000_000
MAX_FEATURES = 2000
MIN_TRAIN_ROWS = 10
DEFAULT_N_ESTIMATORS = 4
DEFAULT_SEED = 42
DEFAULT_VALIDATION_SPLIT = 0.2
MAX_ZIP_EXPANDED_BYTES = 512 * 1024 * 1024  # BYOD ZIP expansion ceiling (512 MiB)
MAX_ZIP_RATIO = 200  # compression-bomb guard: expanded / compressed
DECISION_RULE = "point-estimate"  # predict() returns TabPFN's point estimate (mean of the predictive distribution); no intervals are shipped
METRIC_IDS = ("mae", "rmse", "r2", "mape")
# Bundle format v2 (safetensors_weights): model.safetensors; v1 (model.ckpt) is read-only and deprecated.
ARTIFACT_SCHEMA_VERSION = _sw.BUNDLE_SCHEMA_VERSION
LEGACY_ARTIFACT_SCHEMA_VERSION = _sw.LEGACY_BUNDLE_SCHEMA_VERSION
FITTED_NAME = _sw.FITTED_ESTIMATOR
CHECKPOINT_NAME = _sw.BUNDLE_CHECKPOINT
LEGACY_CHECKPOINT_NAME = _sw.LEGACY_BUNDLE_CHECKPOINT
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"


# --------------------------------------------------------------------------- snapshot scheme


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# tabpfn 8.1.0 reads checkpoints with torch.load and weights_only=None (tabpfn/checkpoint.py:73). That is restricted only because
# torch >= 2.6 turns None into True; an older torch, or TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD set to a truthy value, makes it a full unpickle.
_TORCH_TRUTHY = {"1", "y", "yes", "true"}
_MIN_TORCH = (2, 6)


def require_weights_only_load(torch_version: str | None = None) -> None:
    """Refuse to continue unless a TabPFN checkpoint would load with weights_only=True. ``torch_version`` is for tests."""
    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "").strip().lower() in _TORCH_TRUTHY:
        raise RuntimeError("Refusing to load a TabPFN checkpoint: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is set, which would make torch.load unpickle it without weights_only; unset it.")
    if torch_version is None:
        import torch

        torch_version = torch.__version__
    match = re.match(r"(\d+)\.(\d+)", torch_version)
    if match is None or (int(match[1]), int(match[2])) < _MIN_TORCH:
        raise RuntimeError(f"Refusing to load a TabPFN checkpoint: torch {torch_version} is older than 2.6, so torch.load would not default to weights_only=True; install torch>=2.6.")


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("modelId"), manifest.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError(f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, expected {MODEL_ID}@{MODEL_REVISION}")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("manifest has no file entries")
    return manifest


def verify_snapshot(weights_dir: str | Path | None = None, *, require_checkpoint: bool = True) -> dict[str, Any]:
    """Re-hash every manifest entry under ``weights_dir``; raise on any missing file, size or digest mismatch.

    ``require_checkpoint=False`` skips the source ``.ckpt`` (used once the verified conversion is in place)."""
    root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
    manifest = _read_manifest(root)
    checkpoint_ok = not require_checkpoint
    for entry in manifest["files"]:
        if not require_checkpoint and entry["path"] == WEIGHTS_FILE:
            continue
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {path}")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] == WEIGHTS_FILE:
            checkpoint_ok = digest == WEIGHTS_SHA256 and size == WEIGHTS_BYTES
    if not checkpoint_ok:
        raise ValueError(f"manifest does not pin {WEIGHTS_FILE} at sha256 {WEIGHTS_SHA256} / {WEIGHTS_BYTES} bytes")
    return manifest


def stage_missing_files(weights_dir: str | Path | None = None, *, allow_download: bool = False, downloader: Callable[..., Any] | None = None, skip: Sequence[str] = ()) -> list[str]:
    """Fetch the manifest entries absent from ``weights_dir`` (revision-pinned, never ``main``); refuse without ``allow_download``."""
    root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
    manifest = _read_manifest(root)
    missing = [entry["path"] for entry in manifest["files"] if entry["path"] not in skip and not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(f"snapshot at {root} is missing {missing}; pass allow_download=True to stage them from {MODEL_ID}@{MODEL_REVISION[:12]}")
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    for rel in missing:
        downloader(repo_id=MODEL_ID, filename=rel, revision=MODEL_REVISION, local_dir=str(root))
        if not (root / rel).is_file():
            raise FileNotFoundError(f"download did not produce {root / rel}")
    return missing


def converted_is_verified(weights_dir: str | Path | None = None) -> bool:
    """Whether ``CONVERTED_FILE`` is present under ``weights_dir`` with the pinned size and SHA-256."""
    target = Path(weights_dir or DEFAULT_WEIGHTS_DIR) / CONVERTED_FILE
    return target.is_file() and target.stat().st_size == CONVERTED_BYTES and _sw.sha256_file(target) == CONVERTED_SHA256


def pinned_checkpoint_sha256(layout: int) -> str:
    """The checkpoint digest a pretrained (zero-shot) bundle must carry: the conversion for v2, the source for v1."""
    return CONVERTED_SHA256 if layout == ARTIFACT_SCHEMA_VERSION else WEIGHTS_SHA256


def ensure_converted(weights_dir: str | Path | None = None) -> Path:
    """Return the verified ``CONVERTED_FILE`` under ``weights_dir``, converting the pinned checkpoint once if it is absent.

    An existing file is used only when its size and SHA-256 match the pins; otherwise the source ``.ckpt`` (re-hashed
    against ``WEIGHTS_SHA256``) is read with ``torch.load(weights_only=True)`` and the result must equal
    ``CONVERTED_SHA256`` before it is moved into place."""
    root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
    target = root / CONVERTED_FILE
    if converted_is_verified(root):
        return target
    require_weights_only_load()
    _sw.convert_checkpoint(root / WEIGHTS_FILE, target, expected_sha256=WEIGHTS_SHA256, source_name=WEIGHTS_FILE, expected_converted_sha256=CONVERTED_SHA256)
    return target


# --------------------------------------------------------------------------- datasets


def build_synthetic_dataset(out: str | Path, rows: int = 600, seed: int = DEFAULT_SEED) -> Path:
    """Deterministic train/val/test regression ZIP (mirrors examples/build_synthetic_dataset.py); tutorial data only."""
    from sklearn.model_selection import train_test_split

    if rows < 100:
        raise ValueError("rows must be at least 100")
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.uniform(-2.0, 2.0, size=rows)
    category = rng.choice(["a", "b", "c"], size=rows)
    category_effect = pd.Series(category).map({"a": -2.0, "b": 0.5, "c": 3.0}).to_numpy()
    noise = rng.normal(scale=0.35, size=rows)
    target = 4.0 * x1 - 1.5 * x2 + category_effect + noise - 1.0
    frame = pd.DataFrame({"x1": x1, "x2": x2, "category": category, "target": target})
    train, remainder = train_test_split(frame, test_size=0.3, random_state=seed)
    val, test = train_test_split(remainder, test_size=0.5, random_state=seed)
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, part in (("train.csv", train), ("val.csv", val), ("test.csv", test)):
            archive.writestr(name, part.reset_index(drop=True).to_csv(index=False))
    return out


def _safe_member(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"unsafe archive member: {name!r}")
    return path.name


def safe_extract_zip(zip_path: str | Path, destination: str | Path, max_expanded_bytes: int = MAX_ZIP_EXPANDED_BYTES, max_ratio: int = MAX_ZIP_RATIO) -> list[str]:
    """Extract a ZIP member by member (bare file names only, no directories/absolute/traversing paths); never ``extractall``."""
    zip_path, destination = Path(zip_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        expanded = sum(info.file_size for info in infos)
        compressed = max(1, sum(info.compress_size for info in infos))
        if expanded > max_expanded_bytes:
            raise ValueError(f"archive expands to {expanded} bytes > ceiling {max_expanded_bytes}")
        if expanded / compressed > max_ratio:
            raise ValueError(f"archive compression ratio {expanded / compressed:.0f} > ceiling {max_ratio}")
        for info in infos:
            name = _safe_member(info.filename)
            with archive.open(info) as source, open(destination / name, "wb") as target:
                shutil.copyfileobj(source, target)
            written.append(name)
    return written


def read_dataset_zip(zip_path: str | Path, *, max_expanded_bytes: int = MAX_ZIP_EXPANDED_BYTES, max_ratio: int = MAX_ZIP_RATIO) -> dict[str, pd.DataFrame]:
    """Read ``train.csv`` (+ optional ``val.csv`` / ``test.csv``) from a ZIP with archive-safety checks; never ``extractall``."""
    zip_path = Path(zip_path)
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        expanded = sum(info.file_size for info in infos)
        compressed = max(1, sum(info.compress_size for info in infos))
        if expanded > max_expanded_bytes:
            raise ValueError(f"archive expands to {expanded} bytes > ceiling {max_expanded_bytes}")
        if expanded / compressed > max_ratio:
            raise ValueError(f"archive compression ratio {expanded / compressed:.0f} > ceiling {max_ratio}")
        for info in infos:
            name = _safe_member(info.filename)
            if name in ("train.csv", "val.csv", "test.csv"):
                if name in frames:
                    raise ValueError(f"archive carries {name} more than once")
                frames[name] = pd.read_csv(io.BytesIO(archive.read(info.filename)))
    if "train.csv" not in frames:
        raise ValueError("archive has no train.csv")
    return frames


def read_dataset_dir(dataset_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read a ZIP (``*.zip``) or loose ``train.csv`` / ``val.csv`` / ``test.csv`` from a directory."""
    root = Path(dataset_dir)
    zips = sorted(root.glob("*.zip"))
    if zips:
        if len(zips) > 1:
            raise ValueError(f"expected one ZIP in {root}, found {len(zips)}")
        return read_dataset_zip(zips[0])
    frames = {name: pd.read_csv(root / name) for name in ("train.csv", "val.csv", "test.csv") if (root / name).is_file()}
    if "train.csv" not in frames:
        raise ValueError(f"{root} has neither a ZIP nor train.csv")
    return frames


def random_holdout(train: pd.DataFrame, target_column: str, validation_split: float = DEFAULT_VALIDATION_SPLIT, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded random split for a single ``train.csv``; wrong for temporal/grouped data (supply explicit splits instead)."""
    from sklearn.model_selection import train_test_split

    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be in (0, 1)")
    if target_column not in train.columns:
        raise ValueError(f"target column {target_column!r} not present")
    fit_part, val_part = train_test_split(train, test_size=validation_split, random_state=seed)
    return fit_part.reset_index(drop=True), val_part.reset_index(drop=True)


# --------------------------------------------------------------------------- input contract (DAT24)

INPUT_SCHEMA = {
    "format": "CSV table(s): train.csv with an optional val.csv / test.csv (ZIP or loose files); one row per observation",
    "target": "one numeric target column (declared name), no missing or non-finite values, >= MIN_TRAIN_ROWS training rows, non-constant",
    "features": "every other column; numeric or categorical (strings); unique column names; identical schema across splits",
    "ceilings": {"MAX_TRAIN_ROWS": MAX_TRAIN_ROWS, "MAX_FEATURES": MAX_FEATURES, "MIN_TRAIN_ROWS": MIN_TRAIN_ROWS},
    "splits": "explicit val.csv/test.csv are preserved; without val.csv a seeded random holdout is drawn (independent rows assumed)",
    "reserved_columns": "no `prediction` column in inputs",
}


class InputRejected(ValueError):
    """Raised by ``validate_inputs`` with a structured finding."""

    def __init__(self, finding: dict[str, Any]) -> None:
        super().__init__(finding["message"])
        self.finding = finding


def _reject(code: str, message: str, observed: Any = None) -> InputRejected:
    return InputRejected({"code": code, "verdict": "rejected", "message": message, "observed": observed})


def _check_frame(name: str, frame: pd.DataFrame, target_column: str, feature_columns: list[str] | None) -> list[str]:
    columns = list(frame.columns)
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise _reject("DUPLICATE_COLUMNS", f"{name}: duplicate column names", duplicates)
    if target_column not in columns:
        raise _reject("TARGET_MISSING", f"{name}: target column {target_column!r} not present", columns)
    features = [c for c in columns if c != target_column]
    reserved = [c for c in features if c == "prediction"]
    if reserved:
        raise _reject("RESERVED_COLUMNS", f"{name}: reserved output columns present", reserved)
    if feature_columns is not None and features != feature_columns:
        raise _reject("SCHEMA_MISMATCH", f"{name}: feature columns differ from train.csv", {"expected": feature_columns, "observed": features})
    if frame[target_column].isna().any():
        raise _reject("TARGET_MISSING_VALUES", f"{name}: target has missing values", int(frame[target_column].isna().sum()))
    if not pd.api.types.is_numeric_dtype(frame[target_column]):
        raise _reject("TARGET_NOT_NUMERIC", f"{name}: target column {target_column!r} is not numeric", str(frame[target_column].dtype))
    if not np.isfinite(frame[target_column].to_numpy(dtype=float)).all():
        raise _reject("TARGET_NON_FINITE", f"{name}: target has infinite values", target_column)
    if len(frame) == 0:
        raise _reject("EMPTY_SPLIT", f"{name}: no rows", 0)
    return features


def validate_inputs(train: pd.DataFrame, target_column: str, *, val: pd.DataFrame | None = None, test: pd.DataFrame | None = None, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Apply the input contract to the supplied splits and return the DAT24 input manifest (raises ``InputRejected``)."""
    features = _check_frame("train.csv", train, target_column, None)
    if not features:
        raise _reject("NO_FEATURES", "train.csv has no feature columns", list(train.columns))
    if len(features) > MAX_FEATURES:
        raise _reject("TOO_MANY_FEATURES", f"train.csv has {len(features)} features > MAX_FEATURES {MAX_FEATURES}", len(features))
    if len(train) > MAX_TRAIN_ROWS:
        raise _reject("TOO_MANY_ROWS", f"train.csv has {len(train)} rows > MAX_TRAIN_ROWS {MAX_TRAIN_ROWS}", len(train))
    if len(train) < MIN_TRAIN_ROWS:
        raise _reject("TOO_FEW_ROWS", f"train.csv has {len(train)} rows < MIN_TRAIN_ROWS {MIN_TRAIN_ROWS}", len(train))
    target_values = train[target_column].to_numpy(dtype=float)
    if float(np.std(target_values)) == 0.0:
        raise _reject("CONSTANT_TARGET", "train.csv target is constant; nothing to regress", float(target_values[0]))
    findings: list[dict[str, Any]] = []
    target_stats = {"mean": float(np.mean(target_values)), "std": float(np.std(target_values)), "min": float(np.min(target_values)), "max": float(np.max(target_values))}
    splits: dict[str, Any] = {"train": {"rows": int(len(train)), "target": target_stats}}
    for name, frame in (("val.csv", val), ("test.csv", test)):
        if frame is None:
            continue
        _check_frame(name, frame, target_column, features)
        values = frame[target_column].to_numpy(dtype=float)
        if values.min() < target_stats["min"] or values.max() > target_stats["max"]:
            findings.append({"input": name, "verdict": "warning", "code": "TARGET_OUT_OF_TRAINING_RANGE", "message": f"{name} targets fall outside the training range; extrapolation is being scored", "observed": {"min": float(values.min()), "max": float(values.max())}})
        splits[name.split(".")[0]] = {"rows": int(len(frame)), "target": {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(values.min()), "max": float(values.max())}}
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    for c in numeric:
        values = train[c].dropna().to_numpy(dtype=float)
        if values.size and not np.isfinite(values).all():
            raise _reject("NON_FINITE_FEATURE", f"train.csv numeric feature {c!r} contains infinite values", c)
    if (target_values == 0).mean() > 0.5:
        findings.append({"input": "train.csv", "verdict": "warning", "code": "ZERO_HEAVY_TARGET", "message": "more than half of the training targets are zero; mape is computed over non-zero rows only", "observed": float((target_values == 0).mean())})
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "dataset-0", "mode": "in-context-fit", "target_column": target_column, "feature_columns": features, "numeric_features": len(numeric), "categorical_features": len(features) - len(numeric), "target_stats": target_stats, "splits": splits, "train_sha256": sha256_hex(train.to_csv(index=False).encode("utf-8"))}],
        "verdict": "accepted",
        "findings": findings,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_version": MODEL_VERSION,
    }


def validate_new_rows(frame: pd.DataFrame, feature_columns: Sequence[str], *, target_column: str | None = None) -> pd.DataFrame:
    """Rows for inference must carry exactly the artifact's feature columns and no target/output columns."""
    columns = list(frame.columns)
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise _reject("DUPLICATE_COLUMNS", "new rows: duplicate column names", duplicates)
    reserved = [c for c in [target_column, "prediction"] if c and c in columns]
    if reserved:
        raise _reject("RESERVED_COLUMNS", "new rows: remove target/prediction columns before inference", reserved)
    missing = [c for c in feature_columns if c not in columns]
    extra = [c for c in columns if c not in feature_columns]
    if missing or extra:
        raise _reject("SCHEMA_MISMATCH", "new rows: feature schema mismatch", {"missing": missing, "extra": extra})
    if len(frame) == 0:
        raise _reject("EMPTY_INPUT", "new rows: no rows", 0)
    frame = frame.loc[:, list(feature_columns)].copy()
    for c in frame.select_dtypes(include=np.number).columns:
        values = frame[c].dropna().to_numpy(dtype=float)
        if values.size and not np.isfinite(values).all():
            raise _reject("NON_FINITE_FEATURE", f"new rows: numeric feature {c!r} contains infinite values", c)
    return frame


# --------------------------------------------------------------------------- metrics / baseline


def regression_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    """MAE, RMSE, R² and MAPE (the latter over non-zero targets only; absent when every target is zero)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape or yt.size == 0:
        raise ValueError("y_true and y_pred must be equal-length, non-empty 1-D arrays")
    out = {"mae": float(mean_absolute_error(yt, yp)), "rmse": float(np.sqrt(mean_squared_error(yt, yp))), "r2": float(r2_score(yt, yp)) if yt.size > 1 and float(np.std(yt)) > 0 else float("nan")}
    nonzero = yt != 0
    if nonzero.any():
        out["mape"] = float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])))
    return out


def mean_baseline(train_targets: Sequence[Any], eval_targets: Sequence[Any]) -> dict[str, Any]:
    """Always predict the training mean, scored on the evaluation rows."""
    train_targets = np.asarray(train_targets, dtype=float)
    eval_targets = np.asarray(eval_targets, dtype=float)
    if train_targets.size == 0 or eval_targets.size == 0:
        raise ValueError("baseline needs training and evaluation targets")
    mean = float(train_targets.mean())
    metrics = regression_metrics(eval_targets, np.full(eval_targets.shape, mean))
    return {"trainMean": mean, **metrics}


# --------------------------------------------------------------------------- artifact bundle


def _safe_fitted_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            _safe_member(name)
        if "init_params.json" not in names:
            raise ValueError("fitted archive lacks init_params.json")
        return json.loads(archive.read("init_params.json"))


def manifest_digest(manifest: dict[str, Any], key: str) -> str:
    """Both manifest shapes: flat ``<key>Sha256`` (this module, the worker) or nested ``sha256[<key>]``."""
    value = manifest.get(key + "Sha256") or (manifest.get("sha256") or {}).get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"manifest has no SHA-256 for {key}")
    return value


def validate_artifact_bundle(artifact_dir: str | Path, *, expected_fitted_sha256: str = "", expected_checkpoint_sha256: str = "", pinned_base: bool = False) -> dict[str, Any]:
    """Check manifest layout, member names, sizes, digests and archive safety BEFORE any model state is deserialised.

    The layout (v2 ``model.safetensors`` or deprecated v1 ``model.ckpt``) is decided from the manifest alone; a mixed or
    unknown layout is refused before any other file is opened. ``pinned_base`` binds the checkpoint to the pinned base
    for its layout (``pinned_checkpoint_sha256``)."""
    root = Path(artifact_dir)
    manifest_path = root / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{ARTIFACT_MANIFEST_NAME} missing in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("taskType") != TASK_TYPE:
        raise ValueError(f"unsupported manifest: taskType={manifest.get('taskType')}")
    layout = _sw.check_bundle_layout(manifest)
    _sw.check_bundle_files(root, layout)
    if pinned_base:
        pinned = pinned_checkpoint_sha256(layout)
        if expected_checkpoint_sha256 and expected_checkpoint_sha256.strip().lower() != pinned:
            raise ValueError("expected_checkpoint_sha256 disagrees with the pinned base for this bundle layout")
        expected_checkpoint_sha256 = pinned
    for key in ("targetColumn", "featureColumns"):
        if key not in manifest:
            raise ValueError(f"manifest lacks {key}")
    digests: dict[str, str] = {}
    for key, override in (("fittedEstimator", expected_fitted_sha256), ("foundationCheckpoint", expected_checkpoint_sha256)):
        name = manifest.get(key)
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"manifest {key} must be a bare file name, got {name!r}")
        path = root / name
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError(f"{name} is missing or implausibly small")
        digest = sha256_file(path)
        if digest != manifest_digest(manifest, key):
            raise ValueError(f"{name} SHA-256 {digest} != manifest {manifest_digest(manifest, key)}")
        if override and digest != override.strip().lower():
            raise ValueError(f"{name} SHA-256 {digest} != expected {override.strip().lower()}")
        if key == "foundationCheckpoint" and layout == ARTIFACT_SCHEMA_VERSION and path.stat().st_size != manifest["foundationCheckpointBytes"]:
            raise ValueError(f"{name} is {path.stat().st_size} bytes != manifest foundationCheckpointBytes {manifest['foundationCheckpointBytes']}")
        digests[key] = digest
    init_params = _safe_fitted_archive(root / manifest["fittedEstimator"])
    return {**manifest, "verifiedSha256": digests, "recordedModelPath": init_params.get("model_path"), "layout": layout}


def rewrite_model_path(fitted_archive: Path, checkpoint: Path, destination: Path) -> Path:
    """Copy a ``.tabpfn_fit`` archive while pointing its ``init_params.json`` ``model_path`` at ``checkpoint`` (original untouched)."""
    fitted_archive, checkpoint, destination = fitted_archive.resolve(), checkpoint.resolve(), destination.resolve()
    if not fitted_archive.is_file():
        raise FileNotFoundError(f"fitted estimator not found: {fitted_archive}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"companion checkpoint not found: {checkpoint}")
    saw_init = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(fitted_archive, "r") as source, zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            _safe_member(info.filename)
            payload = source.read(info.filename)
            if info.filename == "init_params.json":
                params = json.loads(payload.decode("utf-8"))
                params["model_path"] = str(checkpoint)
                payload = json.dumps(params, sort_keys=True).encode("utf-8")
                saw_init = True
            target.writestr(info, payload)
    if not saw_init:
        destination.unlink(missing_ok=True)
        raise ValueError("fitted artifact does not contain init_params.json")
    return destination


def zip_artifact_bundle(artifact_dir: str | Path, zip_path: str | Path) -> str:
    """Zip the three bundle members (manifest, fitted archive, the layout's checkpoint) flat for transport; returns the ZIP SHA-256.

    Deterministic: members in sorted order, each with a fixed timestamp (1980-01-01 00:00:00) and mode 0644, so the same
    member bytes always give the same ZIP digest. The ZIP is written to a temporary file and moved into place atomically.
    """
    root, zip_path = Path(artifact_dir), Path(zip_path)
    layout = _sw.check_bundle_layout(json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8")))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted((ARTIFACT_MANIFEST_NAME, FITTED_NAME, _sw.bundle_checkpoint_name(layout))):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with (root / name).open("rb") as source, archive.open(info, "w") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    partial.replace(zip_path)
    return sha256_file(zip_path)


def _coerce_non_json_init_params(model: Any) -> None:
    """tabpfn 8.1.0 serialises ``get_params()`` as JSON; stringify the values that are not JSON-encodable (e.g. a Path)."""
    for param, value in model.get_params(deep=False).items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            setattr(model, param, str(value))


# --------------------------------------------------------------------------- pipeline


@dataclass
class TabPFNRegressorPipeline:
    """The verified TabPFN-3 weights plus, after ``fit`` or ``from_artifact``, an in-context-fitted estimator.

    ``weights_path`` is the safetensors file the estimator loads (``CONVERTED_FILE`` from ``from_pretrained``; a v2
    artifact's ``model.safetensors`` from ``from_artifact``). ``save_artifact`` copies it into the bundle. A v1 artifact
    has no such file (``weights_path`` is its ``model.ckpt``) and cannot be re-exported."""

    weights_path: Path
    device: str = "cpu"
    source: str = "local-snapshot"
    n_estimators: int = DEFAULT_N_ESTIMATORS
    random_state: int = DEFAULT_SEED
    target_column: str | None = None
    feature_columns: list[str] | None = None
    target_stats: dict[str, float] | None = None
    _model: Any = None

    @classmethod
    def from_pretrained(cls, device: str | None = None, weights_dir: str | Path | None = None, allow_download: bool = False, *, n_estimators: int = DEFAULT_N_ESTIMATORS, random_state: int = DEFAULT_SEED) -> TabPFNRegressorPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if converted_is_verified(root):  # the source .ckpt is only needed for the one-time conversion
            stage_missing_files(root, allow_download=allow_download, skip=(WEIGHTS_FILE,))
            verify_snapshot(root, require_checkpoint=False)
        else:
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
        converted = ensure_converted(root)
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(weights_path=converted, device=device, n_estimators=n_estimators, random_state=random_state)

    def _build_estimator(self) -> Any:
        require_weights_only_load()
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(model_path=str(self.weights_path), device=self.device, n_estimators=self.n_estimators, random_state=self.random_state, show_progress_bar=False)

    def fit(self, X: pd.DataFrame, y: Sequence[Any], *, target_column: str = "target", estimator_factory: Callable[[], Any] | None = None) -> TabPFNRegressorPipeline:
        """In-context fit: register the training rows as the support set. No gradient update is performed."""
        model = (estimator_factory or self._build_estimator)()
        targets = pd.Series(np.asarray(list(y), dtype=float), name=target_column)
        model.fit(X, targets)
        self._model = model
        self.target_column = target_column
        self.feature_columns = list(X.columns)
        values = targets.to_numpy()
        self.target_stats = {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}
        self.source = "in-context-fit"
        return self

    def _require(self) -> Any:
        if self._model is None or self.feature_columns is None:
            raise RuntimeError("no fitted estimator: call fit() or from_artifact() first")
        return self._model

    def predict_values(self, X: pd.DataFrame) -> np.ndarray:
        model = self._require()
        X = validate_new_rows(X, self.feature_columns or [], target_column=self.target_column)
        return np.asarray(model.predict(X), dtype=float).reshape(-1)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """``prediction`` = TabPFN's point estimate per row (no intervals are shipped)."""
        values = self.predict_values(X)
        out = pd.DataFrame({"row_id": np.asarray(X.index), "prediction": values})
        out.attrs["decision_rule"] = DECISION_RULE
        return out

    def evaluate(self, X: pd.DataFrame, y: Sequence[Any]) -> dict[str, float]:
        return regression_metrics(np.asarray(list(y), dtype=float), self.predict_values(X))

    def save_artifact(self, artifact_dir: str | Path, *, saver: Callable[[Any, Path], None] | None = None) -> dict[str, Any]:
        """Write a v2 bundle: ``model.tabpfn_fit`` + ``model.safetensors`` (copy of the served safetensors weights) + ``artifact_manifest.json``."""
        model = self._require()
        checkpoint = Path(self.weights_path)
        if checkpoint.suffix != ".safetensors":
            raise ValueError(f"a v2 bundle carries {CHECKPOINT_NAME}; {checkpoint.name} is not a safetensors file (a v1 artifact cannot be re-exported)")
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        if saver is None:
            from tabpfn.model_loading import save_fitted_tabpfn_model

            def saver(estimator: Any, path: Path) -> None:
                _coerce_non_json_init_params(estimator)
                save_fitted_tabpfn_model(estimator, path)

        saver(model, root / FITTED_NAME)
        shutil.copyfile(checkpoint, root / CHECKPOINT_NAME)
        manifest = {
            "taskType": TASK_TYPE,
            "targetColumn": self.target_column,
            "featureColumns": list(self.feature_columns or []),
            "targetStats": dict(self.target_stats or {}),
            **_sw.v2_manifest_fields(TASK_TYPE, sha256_file(root / CHECKPOINT_NAME), (root / CHECKPOINT_NAME).stat().st_size),
            "fittedEstimatorSha256": sha256_file(root / FITTED_NAME),
            "portableLoader": "tabpfn_regressor_pipeline.TabPFNRegressorPipeline.from_artifact",
            "baseModel": {"modelId": MODEL_ID, "revision": MODEL_REVISION, "file": WEIGHTS_FILE, "sha256": WEIGHTS_SHA256, "convertedFile": CONVERTED_FILE, "convertedSha256": CONVERTED_SHA256, "modelVersion": MODEL_VERSION, "license": MODEL_LICENSE},
            "mode": "zero-shot-icl",
            "nEstimators": self.n_estimators,
            "randomState": self.random_state,
        }
        (root / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path, device: str | None = None, *, expected_fitted_sha256: str = "", expected_checkpoint_sha256: str = "", pinned_base: bool = False, loader: Callable[[Path, str], Any] | None = None) -> TabPFNRegressorPipeline:
        """Validate the bundle, then reconstruct the estimator from the fitted archive + companion weights (no refit, no download).

        v2: the fitted archive's ``model_path`` is pointed at ``model.safetensors``. v1 (deprecated): ``model.ckpt`` is read
        with ``torch.load(weights_only=True)`` into a temporary safetensors file, with a ``DeprecationWarning``; tabpfn
        never unpickles a checkpoint on either path."""
        root = Path(artifact_dir)
        manifest = validate_artifact_bundle(root, expected_fitted_sha256=expected_fitted_sha256, expected_checkpoint_sha256=expected_checkpoint_sha256, pinned_base=pinned_base)
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        if loader is None:
            require_weights_only_load()
            from tabpfn.model_loading import load_fitted_tabpfn_model

            def loader(fitted: Path, dev: str) -> Any:
                return load_fitted_tabpfn_model(fitted, device=dev)

        checkpoint = root / manifest["foundationCheckpoint"]
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = checkpoint
            if manifest["layout"] == LEGACY_ARTIFACT_SCHEMA_VERSION:
                require_weights_only_load()
                weights = Path(temp_dir) / "legacy-v1.safetensors"
                _sw.legacy_checkpoint_to_safetensors(checkpoint, weights, expected_sha256=manifest["verifiedSha256"]["foundationCheckpoint"])
            rewritten = rewrite_model_path(root / manifest["fittedEstimator"], weights, Path(temp_dir) / FITTED_NAME)
            model = loader(rewritten, device)
        if not hasattr(model, "predict"):
            raise RuntimeError("reconstructed estimator has no predict method")
        return cls(weights_path=checkpoint, device=device, source="artifact", n_estimators=int(manifest.get("nEstimators", DEFAULT_N_ESTIMATORS)), random_state=int(manifest.get("randomState", DEFAULT_SEED)), target_column=manifest["targetColumn"], feature_columns=list(manifest["featureColumns"]), target_stats=dict(manifest.get("targetStats") or {}), _model=model)


# --------------------------------------------------------------------------- evaluation report (EVAL21)


def evaluation_report(metrics: dict[str, float] | None, *, baseline: dict[str, Any] | None = None, n_validation: int = 0, target_column: str | None = None, sample_kind: str = "synthetic", reload_check: dict[str, Any] | None = None, split_name: str = "val.csv") -> dict[str, Any]:
    """Evaluation stage: the metrics are honest about what they are (single holdout, tutorial sample) or ``not-measurable``."""
    base = {
        "task": TASK_TYPE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_version": MODEL_VERSION,
        "decision_rule": DECISION_RULE,
        "adaptation": "in-context conditioning only (no gradient update); the private-worker fine-tune path is not carried",
        "sample_kind": sample_kind,
        "n_validation": int(n_validation),
        "split": split_name,
        "target_column": target_column,
        "reload_check": reload_check,
        "caveats": ["single holdout, no dispersion estimate", "point estimates only; no prediction intervals are shipped", "mape is computed over non-zero targets only", "synthetic sample metrics are plumbing evidence only" if sample_kind == "synthetic" else "BYOD metrics are one holdout of one table"],
    }
    if not metrics or n_validation == 0:
        return {**base, "metrics": [], "verdict": "not-measurable", "reason": "no labelled validation split was scored", "needs": "a labelled, leakage-safe holdout from the deployment domain scored with mae / rmse / r2 against mean_baseline; repeated splits for any dispersion estimate"}
    entries = [{"id": k, "value": float(v)} for k, v in metrics.items() if k in METRIC_IDS]
    return {**base, "metrics": entries, "baselines": {"training_mean": baseline} if baseline else {}, "verdict": "sample-sanity", "reason": f"{n_validation} validation row(s) from one holdout; tutorial evidence, not a benchmark", "needs": "a domain-representative labelled test set, subgroup breakdowns, residual analysis on held-out data and repeated splits for any generalisable claim"}

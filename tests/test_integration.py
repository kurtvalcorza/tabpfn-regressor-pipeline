"""Integration tier: exercises the relocation loader with real TabPFN.

Fits a real TabPFN regressor (v2, Apache, CPU), persists a DIMER artifact, moves
it to a clean directory, and reloads it through the pipeline's own
`load_dimer_tabpfn_regressor_artifact`. This proves the portable loader
reconstructs a working estimator after DIMER relocates the artifact away from
the training cache — the exact production serving path. Marked `integration`
so it only runs in the integration CI job.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _coerce_non_json_init_params(model) -> None:
    # Mirrors the finetuner's save_artifacts workaround for TabPFN 8.1.0, which
    # str-coerces only torch.dtype and would otherwise fail json.dump on the
    # estimator's device / RegressorModelSpecs model_path.
    for param, value in model.get_params(deep=False).items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            setattr(model, param, str(value))


def test_relocation_loader_roundtrips_real_artifact(tmp_path: Path) -> None:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import save_fitted_tabpfn_model, save_tabpfn_model

    from serving.load_artifact import load_dimer_tabpfn_regressor_artifact

    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(80, 3))
    y_train = x_train @ np.array([2.0, -1.5, 0.5]) - 1.0 + rng.normal(scale=0.1, size=80)
    x_test = rng.normal(size=(20, 3))
    train = pd.DataFrame(x_train, columns=["a", "b", "c"])
    test = pd.DataFrame(x_test, columns=["a", "b", "c"])

    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device="cpu", random_state=0, show_progress_bar=False
    )
    model.fit(train, pd.Series(y_train))
    reference = np.asarray(model.predict(test), dtype=float)

    train_dir = tmp_path / "train_out"
    train_dir.mkdir()
    _coerce_non_json_init_params(model)
    save_fitted_tabpfn_model(model, train_dir / "model.tabpfn_fit")
    save_tabpfn_model(model, train_dir / "model.ckpt")

    # Relocate to a clean serving directory (paths differ from the training cache).
    serve_dir = tmp_path / "clean_serving_env"
    serve_dir.mkdir()
    for name in ("model.tabpfn_fit", "model.ckpt"):
        shutil.copy(train_dir / name, serve_dir / name)

    reloaded = load_dimer_tabpfn_regressor_artifact(serve_dir, device="cpu")
    preds = np.asarray(reloaded.predict(test), dtype=float)

    assert preds.shape[0] == 20
    # Same fitted estimator after relocation -> predictions match within float noise.
    assert np.max(np.abs(preds - reference)) < 1e-2

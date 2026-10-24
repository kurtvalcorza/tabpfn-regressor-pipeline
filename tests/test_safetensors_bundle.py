# ruff: noqa: E501
"""ARTIFACT_SPEC 2.0 SER9 (b): the bundle carries ``model.safetensors``, the faithful conversion of ``model.ckpt``.

The offline tests (no torch) cover the manifest provenance, the standard-library safetensors structure check, the
validator's refusals and the byte-reproducible header. The ``integration`` tests convert real checkpoints with torch +
safetensors and prove the conversion tensor-for-tensor (SER8), including a real TabPFN reload from the safetensors
file while ``torch.load`` is disabled.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabpfn_regressor_pipeline import pipeline as api

ROOT = Path(__file__).resolve().parents[1]


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]], metadata: dict[str, str] | None = None) -> None:
    """Minimal standard-library safetensors writer for offline tests (entries in the given order, no sorting)."""
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset, blobs = 0, []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    text = json.dumps(header).encode("utf-8")
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def fake_converter(checkpoint: Path, destination: Path) -> dict:
    write_safetensors(destination, {"w": ("F32", [2, 2], struct.pack("<4f", 1, 2, 3, 4)), "b": ("F32", [2], struct.pack("<2f", 5, 6))}, {"config": json.dumps({"embed_dim": 4})})
    return {"tool": "fake", "toolVersion": "0", "tensorCount": 2}


class FakeRegressor:
    def fit(self, X: pd.DataFrame, y) -> FakeRegressor:
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


def fake_saver(model: FakeRegressor, path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("init_params.json", json.dumps({"model_path": "fake", "mean": model.mean_}))
        archive.writestr("pad.bin", b"\0" * 2048)


def fake_loader(fitted: Path, device: str) -> FakeRegressor:
    with zipfile.ZipFile(fitted) as zf:
        params = json.loads(zf.read("init_params.json"))
    model = FakeRegressor()
    model.mean_ = params["mean"]
    return model


def pin_checkpoint(monkeypatch: pytest.MonkeyPatch, checkpoint: Path) -> None:
    """Test seam: treat ``checkpoint`` as the pinned upstream file (``save_artifact`` refuses any other)."""
    monkeypatch.setattr(api, "WEIGHTS_SHA256", api.sha256_file(checkpoint))


@pytest.fixture()
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkpoint = tmp_path / "upstream.ckpt"
    checkpoint.write_bytes(b"c" * 4096)
    pin_checkpoint(monkeypatch, checkpoint)
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
    pipe = api.TabPFNRegressorPipeline(weights_path=checkpoint).fit(frame, [1.0, 2.0, 3.0, 4.0], estimator_factory=FakeRegressor)
    art = tmp_path / "artifact"
    pipe.save_artifact(art, saver=fake_saver, converter=fake_converter)
    return art


def _rewrite_manifest(art: Path, **changes) -> None:
    path = art / api.ARTIFACT_MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            manifest.pop(key, None)
        else:
            manifest[key] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_records_the_conversion_and_its_provenance(bundle: Path) -> None:
    manifest = json.loads((bundle / api.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["safetensorsWeights"] == api.SAFETENSORS_NAME == "model.safetensors"
    assert manifest["safetensorsWeightsSha256"] == api.sha256_file(bundle / api.SAFETENSORS_NAME)
    conversion = manifest["safetensorsConversion"]
    assert conversion["derived_from"] == {"file": api.CHECKPOINT_NAME, "sha256": api.sha256_file(bundle / api.CHECKPOINT_NAME)}
    assert conversion["ser9Clause"] == "b" and conversion["tool"] == "fake" and conversion["toolVersion"] == "0"
    checked = api.validate_artifact_bundle(bundle)
    assert set(checked["verifiedSha256"]) == {"fittedEstimator", "foundationCheckpoint", "safetensorsWeights"}


def test_contract_schema_accepts_the_new_manifest(bundle: Path) -> None:
    import jsonschema

    schema = json.loads((ROOT / "contract" / "schemas" / "artifact-manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads((bundle / api.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8")), schema)
    assert {"safetensorsWeights", "safetensorsWeightsSha256", "safetensorsConversion"} <= set(schema["properties"])


def test_validator_refuses_a_missing_safetensors_file(bundle: Path) -> None:
    (bundle / api.SAFETENSORS_NAME).unlink()
    with pytest.raises(ValueError, match="model.safetensors is missing"):
        api.validate_artifact_bundle(bundle)


def test_validator_refuses_a_manifest_without_the_conversion_unless_legacy(bundle: Path) -> None:
    _rewrite_manifest(bundle, safetensorsWeights=None, safetensorsWeightsSha256=None, safetensorsConversion=None)
    with pytest.raises(ValueError, match="SER9"):
        api.validate_artifact_bundle(bundle)
    legacy = api.validate_artifact_bundle(bundle, require_safetensors=False)
    assert "safetensorsWeights" not in legacy["verifiedSha256"]


def test_validator_refuses_a_tampered_safetensors_file(bundle: Path) -> None:
    path = bundle / api.SAFETENSORS_NAME
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="model.safetensors SHA-256"):
        api.validate_artifact_bundle(bundle)
    # A re-hashed manifest over a file that is not safetensors (here a pickle) is still refused, before any load.
    path.write_bytes(b"\x80\x04\x95" + b"\0" * 64)
    _rewrite_manifest(bundle, safetensorsWeightsSha256=api.sha256_file(path))
    with pytest.raises(ValueError, match="safetensors header"):
        api.validate_artifact_bundle(bundle)


def test_validator_refuses_a_conversion_not_derived_from_this_checkpoint(bundle: Path) -> None:
    manifest = json.loads((bundle / api.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    conversion = dict(manifest["safetensorsConversion"], derived_from={"file": api.CHECKPOINT_NAME, "sha256": "0" * 64})
    _rewrite_manifest(bundle, safetensorsConversion=conversion)
    with pytest.raises(ValueError, match="not recorded as derived from"):
        api.validate_artifact_bundle(bundle)
    _rewrite_manifest(bundle, safetensorsWeights="../model.safetensors")
    with pytest.raises(ValueError, match="must be 'model.safetensors'"):
        api.validate_artifact_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda h: h["w"].update(data_offsets=[0, 12]), "spans"),
        (lambda h: h["b"].update(data_offsets=[8, 16]), "gap or overlap"),
        (lambda h: h["w"].update(dtype="PICKLE"), "dtype"),
        (lambda h: h.update(__metadata__={"config": 1}), "strings to strings"),
    ],
)
def test_header_check_refuses_malformed_files(tmp_path: Path, mutate, message: str) -> None:
    good = tmp_path / "good.safetensors"
    fake_converter(tmp_path / "unused", good)
    parsed = api.read_safetensors_header(good)
    assert sorted(parsed["tensors"]) == ["b", "w"] and parsed["metadata"] == {"config": '{"embed_dim": 4}'}
    data = good.read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + n])
    mutate(header)
    text = json.dumps(header).encode("utf-8")
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(struct.pack("<Q", len(text)) + text + data[8 + n :])
    with pytest.raises(ValueError, match=message):
        api.read_safetensors_header(bad)
    (tmp_path / "short").write_bytes(data[: 8 + n + 4])
    with pytest.raises(ValueError, match="data section"):
        api.read_safetensors_header(tmp_path / "short")


def test_canonical_header_makes_the_bytes_independent_of_metadata_order(tmp_path: Path) -> None:
    tensors = {"z": ("F32", [1], struct.pack("<f", 1)), "a": ("I64", [], struct.pack("<q", 7))}
    one, two = tmp_path / "one.safetensors", tmp_path / "two.safetensors"
    write_safetensors(one, tensors, {"k1": "1", "k2": '{"x":[1,2]}', "k3": "null"})
    write_safetensors(two, tensors, {"k3": "null", "k1": "1", "k2": '{"x":[1,2]}'})
    assert one.read_bytes() != two.read_bytes()
    for path in (one, two):
        api.canonicalize_safetensors_header(path)
        assert struct.unpack("<Q", path.read_bytes()[:8])[0] % 8 == 0
    assert one.read_bytes() == two.read_bytes()
    assert api.read_safetensors_header(one)["metadata"] == {"k1": "1", "k2": '{"x":[1,2]}', "k3": "null"}


def test_bundle_zip_carries_the_safetensors_member_and_reload_uses_it(bundle: Path, tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    api.zip_artifact_bundle(bundle, archive)
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == sorted([api.ARTIFACT_MANIFEST_NAME, api.FITTED_NAME, api.CHECKPOINT_NAME, api.SAFETENSORS_NAME])
    seen = {}

    def loader(fitted: Path, device: str) -> FakeRegressor:
        with zipfile.ZipFile(fitted) as zf:
            seen.update(json.loads(zf.read("init_params.json")))
        model = FakeRegressor()
        model.mean_ = seen["mean"]
        return model

    reloaded = api.TabPFNRegressorPipeline.from_artifact(bundle, device="cpu", loader=loader)
    assert Path(seen["model_path"]).name == api.SAFETENSORS_NAME and reloaded.weights_path.name == api.SAFETENSORS_NAME


def _legacy(art: Path) -> Path:
    """Turn ``art`` into a pre-conversion bundle like the pinned v1 sample: no ``model.safetensors``."""
    path = art / api.ARTIFACT_MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("safetensorsWeights", "safetensorsWeightsSha256", "safetensorsConversion"):
        manifest.pop(key)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (art / api.SAFETENSORS_NAME).unlink()
    return art


def test_a_reloaded_bundle_reexports_the_authentic_checkpoint(bundle: Path, tmp_path: Path) -> None:
    reloaded = api.TabPFNRegressorPipeline.from_artifact(bundle, device="cpu", loader=fake_loader)
    again = tmp_path / "again"
    manifest = reloaded.save_artifact(again, saver=fake_saver, converter=fake_converter)
    assert api.sha256_file(again / api.CHECKPOINT_NAME) == api.sha256_file(bundle / api.CHECKPOINT_NAME) == api.WEIGHTS_SHA256
    assert (again / api.SAFETENSORS_NAME).read_bytes() == (bundle / api.SAFETENSORS_NAME).read_bytes()
    assert manifest["foundationCheckpointSha256"] == api.WEIGHTS_SHA256
    api.validate_artifact_bundle(again)
    assert reloaded.weights_path.name == api.SAFETENSORS_NAME and reloaded.checkpoint_path == bundle / api.CHECKPOINT_NAME


def test_a_reloaded_legacy_bundle_reexports_its_checkpoint(bundle: Path, tmp_path: Path) -> None:
    legacy = _legacy(bundle)
    with pytest.raises(ValueError, match="safetensorsWeights"):
        api.TabPFNRegressorPipeline.from_artifact(legacy, device="cpu", loader=fake_loader)
    reloaded = api.TabPFNRegressorPipeline.from_artifact(legacy, device="cpu", loader=fake_loader, require_safetensors=False)
    again = tmp_path / "again"
    reloaded.save_artifact(again, saver=fake_saver, converter=fake_converter)
    assert api.sha256_file(again / api.CHECKPOINT_NAME) == api.sha256_file(legacy / api.CHECKPOINT_NAME) == api.WEIGHTS_SHA256
    api.validate_artifact_bundle(again)  # the re-export is a full SER9 (b) bundle
    assert reloaded.weights_path == reloaded.checkpoint_path == legacy / api.CHECKPOINT_NAME


def test_export_refuses_a_checkpoint_that_is_not_the_pinned_one(bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"d" * 4096)
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    pipe = api.TabPFNRegressorPipeline(weights_path=other).fit(frame, [1.0, 2.0, 3.0, 4.0], estimator_factory=FakeRegressor)
    with pytest.raises(ValueError, match="not the pinned"):
        pipe.save_artifact(tmp_path / "fresh", saver=fake_saver, converter=fake_converter)
    assert not (tmp_path / "fresh").exists()
    reloaded = api.TabPFNRegressorPipeline.from_artifact(bundle, device="cpu", loader=fake_loader)
    monkeypatch.setattr(api, "WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        reloaded.save_artifact(tmp_path / "again", saver=fake_saver, converter=fake_converter)


def test_plain_data_refuses_live_objects() -> None:
    assert api._plain_data({"t": (None, "safepower"), "n": [1, 2.5, True]}, "x") == {"t": [None, "safepower"], "n": [1, 2.5, True]}
    with pytest.raises(ValueError, match="not plain data"):
        api._plain_data({"p": Path("/x")}, "config")
    with pytest.raises(ValueError, match="non-finite"):
        api._plain_data(float("nan"), "config.x")


# --------------------------------------------------------------------------- integration (torch + safetensors)


def _synthetic_checkpoint(path: Path, *, scale: float = 1.0) -> None:
    import torch

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4) * scale
    state = {
        "encoder.weight": base,
        "encoder.view": base[1],  # shares storage with encoder.weight: the conversion must clone it
        "head.bias": torch.tensor([0.5, -0.25], dtype=torch.bfloat16),
        "borders": torch.linspace(-1, 1, 5, dtype=torch.float64),
        "count": torch.tensor(3, dtype=torch.int64),
    }
    torch.save({"state_dict": state, "config": {"embed_dim": 4, "name": "toy", "dropout": 0.0}, "inference_config": {"Y": (None, "safepower"), "N": None}, "architecture_name": "toy", "optimizer_state": None}, path)


@pytest.mark.integration
def test_conversion_is_faithful_and_carries_no_pickle(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file

    ckpt, out = tmp_path / "model.ckpt", tmp_path / "model.safetensors"
    _synthetic_checkpoint(ckpt)
    provenance = api.convert_checkpoint_to_safetensors(ckpt, out)
    assert provenance["tensorCount"] == 5 and provenance["tool"] == "safetensors" and provenance["checkpointLoad"] == "torch.load(weights_only=True)"
    original = torch.load(ckpt, weights_only=True)
    converted = load_file(out)
    for name, tensor in original["state_dict"].items():
        assert converted[name].dtype == tensor.dtype and converted[name].shape == tensor.shape and torch.equal(converted[name], tensor)
    header = api.read_safetensors_header(out)
    assert json.loads(header["metadata"]["inference_config"]) == {"Y": [None, "safepower"], "N": None}
    assert set(header["metadata"]) == {"config", "inference_config", "architecture_name", "optimizer_state"}
    raw = out.read_bytes()
    assert raw[8:9] == b"{" and b"\x80\x02" not in raw[:2] and not raw.startswith(b"PK")  # JSON header, no pickle/zip
    assert api.verify_safetensors_conversion(ckpt, out)["tensorCount"] == 5


@pytest.mark.integration
def test_conversion_bytes_are_reproducible_across_interpreters(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.ckpt"
    _synthetic_checkpoint(ckpt)
    script = "import sys; from tabpfn_regressor_pipeline import pipeline as api; api.convert_checkpoint_to_safetensors(sys.argv[1], sys.argv[2])"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": "random"}
    digests = set()
    for i in range(3):
        out = tmp_path / f"run{i}.safetensors"
        subprocess.run([sys.executable, "-c", script, str(ckpt), str(out)], check=True, env=env)
        digests.add(api.sha256_file(out))
    assert len(digests) == 1


@pytest.mark.integration
def test_verification_refuses_a_conversion_that_disagrees_with_the_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt, other, out = tmp_path / "model.ckpt", tmp_path / "other.ckpt", tmp_path / "model.safetensors"
    _synthetic_checkpoint(ckpt)
    pin_checkpoint(monkeypatch, ckpt)
    _synthetic_checkpoint(other, scale=2.0)
    api.convert_checkpoint_to_safetensors(other, out)
    with pytest.raises(ValueError, match="values differ"):
        api.verify_safetensors_conversion(ckpt, out)
    # Inside a bundle: the manifest is internally consistent, but the tensors are not the checkpoint's.
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    pipe = api.TabPFNRegressorPipeline(weights_path=ckpt).fit(frame, [1.0, 2.0, 3.0, 4.0], estimator_factory=FakeRegressor)
    art = tmp_path / "artifact"
    pipe.save_artifact(art, saver=fake_saver)
    api.validate_artifact_bundle(art, verify_conversion=True)
    (art / api.SAFETENSORS_NAME).write_bytes(out.read_bytes())
    _rewrite_manifest(art, safetensorsWeightsSha256=api.sha256_file(out))
    api.validate_artifact_bundle(art)  # digest-level checks alone cannot see this
    with pytest.raises(ValueError, match="values differ"):
        api.validate_artifact_bundle(art, verify_conversion=True)


@pytest.mark.integration
def test_real_tabpfn_reloads_from_safetensors_without_torch_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import save_tabpfn_model

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = x["a"] * 2.0 - x["b"] + rng.normal(scale=0.1, size=100)
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0, show_progress_bar=False, n_estimators=2)
    model.fit(x[:80], y[:80])
    reference = np.asarray(model.predict(x[80:]), dtype=float)
    ckpt = tmp_path / "upstream.ckpt"
    save_tabpfn_model(model, ckpt)
    pin_checkpoint(monkeypatch, ckpt)
    pipe = api.TabPFNRegressorPipeline(weights_path=ckpt, device="cpu", n_estimators=2, random_state=0, target_column="target", feature_columns=["a", "b", "c"], target_stats={"mean": 0.0}, _model=model)
    art = tmp_path / "artifact"
    pipe.save_artifact(art)
    api.validate_artifact_bundle(art, verify_conversion=True)

    real_load = torch.load
    opened: list = []

    def refuse_files(source, *args, **kwargs):
        # model.tabpfn_fit (SER9 (a) fitted state) is joblib and unpickles in-memory torch storages; that is not a
        # checkpoint file. Any torch.load of a file path here would be the checkpoint being unpickled.
        if isinstance(source, (str, os.PathLike)):
            opened.append(str(source))
            raise AssertionError(f"torch.load opened {source} during reload")
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", refuse_files)
    reloaded = api.TabPFNRegressorPipeline.from_artifact(art, device="cpu")
    assert reloaded.weights_path.name == api.SAFETENSORS_NAME and opened == []
    assert np.max(np.abs(reloaded.predict_values(x[80:]) - reference)) < 1e-5


@pytest.mark.integration
def test_a_reloaded_bundle_reexports_through_the_real_converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "upstream.ckpt"
    _synthetic_checkpoint(ckpt)
    pin_checkpoint(monkeypatch, ckpt)
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    pipe = api.TabPFNRegressorPipeline(weights_path=ckpt).fit(frame, [1.0, 2.0, 3.0, 4.0], estimator_factory=FakeRegressor)
    art = tmp_path / "artifact"
    pipe.save_artifact(art, saver=fake_saver)
    reloaded = api.TabPFNRegressorPipeline.from_artifact(art, device="cpu", loader=fake_loader)
    again = tmp_path / "again"
    reloaded.save_artifact(again, saver=fake_saver)  # torch.load(weights_only=True) of the authentic model.ckpt
    assert api.sha256_file(again / api.CHECKPOINT_NAME) == api.sha256_file(ckpt) == api.WEIGHTS_SHA256
    assert (again / api.SAFETENSORS_NAME).read_bytes() == (art / api.SAFETENSORS_NAME).read_bytes()
    api.validate_artifact_bundle(again, verify_conversion=True)
    # legacy (pre-conversion) bundle: weights come from model.ckpt and the re-export gains the conversion
    legacy = _legacy(art)
    reloaded = api.TabPFNRegressorPipeline.from_artifact(legacy, device="cpu", loader=fake_loader, require_safetensors=False)
    third = tmp_path / "third"
    reloaded.save_artifact(third, saver=fake_saver)
    assert api.sha256_file(third / api.CHECKPOINT_NAME) == api.WEIGHTS_SHA256
    assert (third / api.SAFETENSORS_NAME).read_bytes() == (again / api.SAFETENSORS_NAME).read_bytes()
    api.validate_artifact_bundle(third, verify_conversion=True)

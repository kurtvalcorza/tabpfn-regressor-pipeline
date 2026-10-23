"""Refused user input is shown as one clean line (NOTEBOOK_SPEC 3.0 DAT19/DAT20).

Each refusal path runs the generated notebook's own kernel code (the ``run_stage`` helper of the Section 2 cell and the
learner cell) against the real stage runner in a child process, as the notebook does, with only ``google.colab``'s
upload dialog stood in. For every path the cell must raise ``Refused`` (an exception, so Run all stops at that cell),
render as exactly one ``Refused: <file>: <reason>`` line through IPython's ``_render_traceback_`` hook, name the file
the learner supplied (not the internal ``train.csv``), and print no ``Traceback`` from the stage. A genuine failure still
raises ``RuntimeError`` with the stage's traceback. The Section 4 case is the browser kit's ``refuse.csv``: the
diabetes table (unscaled) whose target ``progression`` holds three text values (``about 150``, ``high``, ``?``). CI's dependencies suffice (no torch, no tabpfn);
the last test also runs a real IPython kernel when ``ipykernel`` is installed.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tutorials" / "tabpfn_regressor_colab.ipynb"
STEM = "tabpfn_regressor"


def _cells() -> list[str]:
    notebook = json.loads(E2E.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(needle: str) -> str:
    found = [c for c in _cells() if needle in c]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


def _kernel(tmp_path: Path) -> dict:
    """The notebook's own stage helper, bound to a run directory that carries the package and the stage runner."""
    run = tmp_path / "run"
    (run / "state").mkdir(parents=True)
    shutil.copytree(ROOT / "src", run / "src")
    shutil.copyfile(ROOT / "tools" / "tutorial_stages.py", run / "tutorial_stages.py")
    install = _cell("def run_stage(stage, **options):")
    helper = install[install.index("REFUSED_EXIT = 3"):]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    ns = {"ROOT": run, "OUTPUTS": tmp_path / "outputs", "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": env, "Path": Path, "json": json, "subprocess": subprocess, "shutil": shutil}
    exec(compile(helper, "<run_stage>", "exec"), ns)
    return ns


@contextlib.contextmanager
def _colab(*uploads: dict):
    queue = list(uploads)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: queue.pop(0))
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _run_all(ns: dict, cells: list[str]) -> tuple[BaseException | None, str, int]:
    """Run cells in order as Run all does: the first cell that raises stops the run. Returns the exception, the cell
    output (stdout plus the exception as IPython renders it) and how many cells ran."""
    out, ran, raised = io.StringIO(), 0, None
    with contextlib.redirect_stdout(out):
        for source in cells:
            ran += 1
            try:
                exec(compile(source, f"<cell {ran}>", "exec"), ns)
            except BaseException as exc:  # noqa: BLE001 -- what IPython would show is asserted below
                raised = exc
                break
    shown = out.getvalue()
    if raised is not None and hasattr(raised, "_render_traceback_"):
        shown += "\n".join(raised._render_traceback_()) + "\n"
    return raised, shown, ran


def _assert_refused(raised, shown: str, ran: int, file: str) -> str:
    assert type(raised).__name__ == "Refused", (raised, shown)
    lines = raised._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith(f"Refused: {file}: "), lines
    assert [line for line in shown.splitlines() if "Refused:" in line] == lines, shown
    assert "Traceback" not in shown, shown
    assert ran == 1, "Run all must stop at the refusing cell"
    return lines[0]


TEXT_ROWS = {3: "about 150", 17: "high", 40: "?"}  # data rows of the kit's refuse.csv (CSV lines 5, 19 and 42)


def _diabetes(path: Path, text_targets: bool = False, label: str = "progression") -> bytes:
    from sklearn.datasets import load_diabetes

    frame = load_diabetes(as_frame=True, scaled=False).frame.rename(columns={"target": label})
    if text_targets:
        frame[label] = frame[label].astype(object)
        for row, value in TEXT_ROWS.items():
            frame.loc[row, label] = value
    frame.to_csv(path, index=False)
    return path.read_bytes()


def _section4(**fields) -> str:
    cell = _cell("USE_BYOD = False  # @param")
    for name, value in {"USE_BYOD": True, **fields}.items():
        cell = _set(cell, name, value)
    return cell


def _accepted(tmp_path: Path, ns: dict) -> None:
    payload = _diabetes(tmp_path / "accept.csv")
    with _colab({"accept.csv": payload}):
        raised, shown, _ = _run_all(ns, [_section4(TARGET_COLUMN="progression"), "run_stage('validate')"])
    assert raised is None, shown
    for name in ("condition.json", "reload.json", "export.json"):  # the model stages are not run here
        (ns["ROOT"] / "state" / name).write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------- Section 4: the learner's dataset


def test_browser_kit_refuse_csv_is_one_clean_line_naming_the_upload(tmp_path) -> None:
    """The kit's Pass B: refuse.csv uploaded with TARGET_COLUMN = 'progression' (three of its targets are text)."""
    ns = _kernel(tmp_path)
    payload = _diabetes(tmp_path / "refuse.csv", text_targets=True)
    stale = ns["OUTPUTS"] / f"{STEM}_input_manifest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    with _colab({"refuse.csv": payload}):
        raised, shown, ran = _run_all(ns, [_section4(TARGET_COLUMN="progression"), "run_stage('validate')"])
    line = _assert_refused(raised, shown, ran, "refuse.csv")
    assert "[TARGET_NOT_NUMERIC]" in line and "A regression target must be numeric" in line and "run the Section 4 cell again" in line
    assert "'about 150' (line 5), 'high' (line 19), '?' (line 42)" in line  # DAT19: the offending values and lines
    assert "train.csv" not in line  # DAT19: the file the learner uploaded, not the internal split name
    assert not stale.exists() and not (ns["ROOT"] / "state" / "data.json").exists()  # no earlier result survives
    error = json.loads((ns["ROOT"] / "state" / "data.error.json").read_text(encoding="utf-8"))
    assert error["refused"] is True and error["file"] == "refuse.csv"


@pytest.mark.parametrize(
    ("fields", "upload", "file", "needle"),
    [
        ({"TARGET_COLUMN": "progression", "DROP_COLUMNS": ["patient_id"]}, "ok", "refuse.csv", "DROP_COLUMNS ['patient_id'] are not in the header"),
        ({"TARGET_COLUMN": "progression"}, "stray", "refuse.csv", "column 'bmi' is numeric except for 1 value(s) ['abc']"),
        ({"TARGET_COLUMN": "target"}, "ok", "refuse.csv", "[TARGET_MISSING]"),
        ({"TARGET_COLUMN": "progression"}, "zip", "refuse.zip/train.csv", "[TARGET_NOT_NUMERIC]"),
        ({"TARGET_COLUMN": "progression"}, "notazip", "refuse.zip", "cannot be read as a dataset"),
        ({"TARGET_COLUMN": "progression"}, "text", "refuse.txt", "BYOD must be a dataset ZIP"),
        ({"TARGET_COLUMN": "progression", "BYOD_PATH": "/nowhere/refuse.csv"}, None, "/nowhere/refuse.csv", "does not exist in this runtime"),
    ],
    ids=["drop-columns", "stray-string", "target-missing", "zip-member", "corrupt-zip", "wrong-format", "missing-path"],
)
def test_section4_refusals_are_one_clean_line(tmp_path, fields, upload, file, needle) -> None:
    ns = _kernel(tmp_path)
    good = _diabetes(tmp_path / "good.csv")
    if upload == "stray":
        lines = good.decode().splitlines()
        cells = lines[5].split(",")
        cells[2] = "abc"
        lines[5] = ",".join(cells)
        uploads = [{"refuse.csv": ("\n".join(lines) + "\n").encode()}]
    elif upload == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("train.csv", _diabetes(tmp_path / "v.csv", text_targets=True))
        uploads = [{"refuse.zip": buffer.getvalue()}]
    elif upload == "notazip":
        uploads = [{"refuse.zip": b"not a zip"}]
    elif upload == "text":
        uploads = [{"refuse.txt": good}]
    elif upload == "ok":
        uploads = [{"refuse.csv": good}]
    else:
        uploads = []
    with _colab(*uploads):
        raised, shown, ran = _run_all(ns, [_section4(**fields), "run_stage('validate')"])
    line = _assert_refused(raised, shown, ran, file)
    assert needle in line, line


def test_section4_two_uploaded_files_are_refused_naming_both(tmp_path) -> None:
    ns = _kernel(tmp_path)
    with _colab({"a.csv": b"x\n1\n", "b.csv": b"x\n1\n"}):
        raised, shown, ran = _run_all(ns, [_section4(TARGET_COLUMN="progression"), "run_stage('validate')"])
    line = _assert_refused(raised, shown, ran, "a.csv, b.csv")
    assert "upload exactly one file" in line


# ---------------------------------------------------------------- Section 9: the learner's own rows


@pytest.mark.parametrize(
    ("rows", "needle"),
    [
        ("age,sex,bmi\n59.0,2.0,32.1\n", "[SCHEMA_MISMATCH]"),
        ("patient,age,sex,bmi,bp,s1,s2,s3,s4,s5,s6\n1,59.0,2.0,32.1,101.0,157.0,93.2,38.0,4.0,4.8598,87.0\n", "If ['patient'] are identifiers, list them in ID_COLUMNS"),
        ("age,sex,bmi,bp,s1,s2,s3,s4,s5,s6\n59.0,2.0,abc,101.0,157.0,93.2,38.0,4.0,4.8598,87.0\n" + "48.0,1.0,21.6,87.0,183.0,103.2,70.0,3.0,3.8918,69.0\n" * 12, "column 'bmi' is numeric except for 1 value(s) ['abc']"),
    ],
    ids=["schema", "identifier-hint", "stray-string"],
)
def test_section9_own_rows_refusals_are_one_clean_line(tmp_path, rows, needle) -> None:
    ns = _kernel(tmp_path)
    _accepted(tmp_path, ns)
    stale = ns["OUTPUTS"] / f"{STEM}_predictions.csv"
    stale.write_text("old", encoding="utf-8")
    cell = _set(_cell("USE_BYOD_ROWS = False  # @param"), "USE_BYOD_ROWS", True)
    with _colab({"own_rows.csv": rows.encode()}):
        raised, shown, ran = _run_all(ns, [cell, "run_stage('activity')"])
    line = _assert_refused(raised, shown, ran, "own_rows.csv")
    assert needle in line and "Section 9 cell again" in line
    assert not stale.exists()  # DAT20: no earlier prediction left on disk to be read as this one


# ---------------------------------------------------------------- Section 10: your own artifact ZIP and rows


def test_section10_own_artifact_zip_is_refused(tmp_path) -> None:
    ns = _kernel(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "not an artifact")
    cell = _set(_cell("ARTIFACT_SOURCE = 'Pinned sample artifact'"), "ARTIFACT_SOURCE", "My own artifact ZIP")
    with _colab({"my_artifact.zip": buffer.getvalue()}):
        raised, shown, ran = _run_all(ns, [cell, "run_stage('external_reconstruct')"])
    line = _assert_refused(raised, shown, ran, "my_artifact.zip")
    assert "is not a usable artifact" in line and "run this cell again" in line


def test_section10_wrong_digest_for_own_artifact_is_refused(tmp_path) -> None:
    ns = _kernel(tmp_path)
    cell = _set(_set(_cell("ARTIFACT_SOURCE = 'Pinned sample artifact'"), "ARTIFACT_SOURCE", "My own artifact ZIP"), "EXPECTED_ZIP_SHA256", "0" * 64)
    with _colab({"my_artifact.zip": b"PK\x05\x06" + b"\x00" * 18}):
        raised, shown, ran = _run_all(ns, [cell])
    assert "Trusted digest mismatch" in _assert_refused(raised, shown, ran, "my_artifact.zip")


def test_section10_own_rows_are_refused_before_the_artifact_is_loaded(tmp_path) -> None:
    ns = _kernel(tmp_path)
    manifest = {"featureColumns": ["a", "b"], "targetColumn": "target"}
    (ns["ROOT"] / "state" / "external_artifact.json").write_text(json.dumps({"source": "own", "zip": "mine.zip", "manifest": manifest}), encoding="utf-8")
    artifact_cell = _cell("ARTIFACT_SOURCE = 'Pinned sample artifact'")
    upload_helper = artifact_cell[artifact_cell.index("def external_upload"):artifact_cell.index("if ARTIFACT_SOURCE ==")]
    exec(compile(upload_helper, "<external_upload>", "exec"), ns)
    cell = _set(_cell("USE_OWN_ROWS = False  # @param"), "USE_OWN_ROWS", True)
    with _colab({"own_rows.csv": b"a,c\n1,2\n"}):
        raised, shown, ran = _run_all(ns, [cell])
    assert "[SCHEMA_MISMATCH]" in _assert_refused(raised, shown, ran, "own_rows.csv")


# ---------------------------------------------------------------- genuine failures keep their traceback


def test_a_genuine_failure_is_not_a_refusal(tmp_path) -> None:
    """A stage that fails for a reason other than the learner's input still raises RuntimeError with its traceback."""
    ns = _kernel(tmp_path)
    raised, shown, _ = _run_all(ns, ["run_stage('validate')"])
    assert type(raised) is RuntimeError and "failed (exit 1)" in str(raised)
    assert "Traceback" in shown and "Refused:" not in shown


def test_stage_runner_exit_codes(tmp_path) -> None:
    """The child process: a refusal exits REFUSED_EXIT (3) with no traceback on its streams; a crash exits 1."""
    ns = _kernel(tmp_path)
    run = ns["ROOT"]
    csv = tmp_path / "refuse.csv"
    _diabetes(csv, text_targets=True)
    base = [sys.executable, str(run / "tutorial_stages.py"), "--root", str(run), "--outputs", str(tmp_path / "outputs"), "--weights", str(tmp_path / "weights")]
    refused = subprocess.run([*base, "--stage", "data", "--options", json.dumps({"use_byod": True, "byod_path": str(csv), "target_column": "progression"})], capture_output=True, text=True, env=ns["ENV"])
    assert refused.returncode == 3 and "Traceback" not in refused.stdout + refused.stderr
    crashed = subprocess.run([*base, "--stage", "validate"], capture_output=True, text=True, env=ns["ENV"])
    assert crashed.returncode == 1 and "Traceback" in crashed.stderr


def test_synthetic_default_path_rejection_is_not_a_refusal(tmp_path) -> None:
    """Only the learner's input is refused cleanly: the default sample failing its contract is a bug, kept as an error."""
    ns = _kernel(tmp_path)
    raised, shown, _ = _run_all(ns, [_set(_section4(TARGET_COLUMN="no_such_column"), "USE_BYOD", False)])
    assert type(raised) is RuntimeError and "[TARGET_MISSING]" in str(raised) and "Refused:" not in shown


# ---------------------------------------------------------------- the real kernel (when ipykernel is installed)


def test_real_kernel_shows_one_line_and_stops_run_all(tmp_path) -> None:
    pytest.importorskip("ipykernel")
    nbclient = pytest.importorskip("nbclient")
    import nbformat

    ns = _kernel(tmp_path)
    payload = _diabetes(tmp_path / "refuse.csv", text_targets=True)
    setup = (
        "import json, os, shutil, subprocess, sys, types\nfrom pathlib import Path\n"
        f"ROOT = Path({str(ns['ROOT'])!r}); OUTPUTS = Path({str(ns['OUTPUTS'])!r}); WEIGHTS = Path({str(ns['WEIGHTS'])!r})\n"
        "PYTHON = Path(sys.executable)\nENV = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP')}\n"
        f"colab = types.ModuleType('google.colab'); colab.files = types.SimpleNamespace(upload=lambda: {{'refuse.csv': {payload!r}}})\n"
        "google = types.ModuleType('google'); google.colab = colab; sys.modules.update({'google': google, 'google.colab': colab})\n"
    )
    install = _cell("def run_stage(stage, **options):")
    cells = [setup, install[install.index("REFUSED_EXIT = 3"):], _section4(TARGET_COLUMN="progression"), "print('AFTER THE REFUSAL')"]
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(c) for c in cells])
    client = nbclient.NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}})
    with pytest.raises(nbclient.exceptions.CellExecutionError):
        client.execute()
    refused = notebook.cells[2]
    errors = [o for o in refused.outputs if o.output_type == "error"]
    text = "\n".join(o.get("text", "") for o in refused.outputs if o.output_type == "stream") + "\n".join("\n".join(o.traceback) for o in errors)
    assert len(errors) == 1 and errors[0].ename == "Refused"
    assert errors[0].traceback == [errors[0].traceback[0]] and errors[0].traceback[0].startswith("Refused: refuse.csv: [TARGET_NOT_NUMERIC]")
    assert text.count("Refused:") == 1 and "Traceback" not in text
    assert notebook.cells[3].outputs == [] and notebook.cells[3].execution_count is None  # Run all stopped

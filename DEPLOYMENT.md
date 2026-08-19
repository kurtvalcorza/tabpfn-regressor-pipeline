# Deployment Runbook

## Components

Deploy the two component repositories pinned by `COMPONENTS.json`:

1. `tabpfn-regressor-dataset-validator` — CPU validator container
2. `tabpfn-regressor-finetuner` — CUDA-capable training container

Do not copy source from either repository into this integration repo. Update the component manifest only after the referenced revisions pass review/CI.

## DIMER pipeline identity

Recommended metadata:

- task type: Custom / Other if native tabular regression is unavailable
- internal task identity: `tabular_regression`
- validator: `tabpfn-regressor-dataset-validator`
- fine-tuner: `tabpfn-regressor-finetuner`

The fine-tuner root contains `dimer-pipeline.json`, which defines user-visible preprocessing and fine-tuning controls.

## Model delivery

For development, TabPFN can resolve the selected generation through its normal cache/download mechanism.

For controlled production deployment, mount an approved checkpoint into the training container and set:

```text
DIMER_TABPFN_MODEL_PATH=/path/to/tabpfn-regressor.ckpt
```

The path must exist inside the container. The run records its SHA-256 when available.

Review the package and exact model-weight terms before enabling a hosted service.

## GPU requirement

`fine_tune=true` is a hard CUDA path. If DIMER schedules the container without CUDA, the run fails with an explicit error. It does not downgrade to zero-shot.

`fine_tune=false` can be used for ICL-only operation and may be scheduled separately according to tested resource requirements.

## Dataset resource limits

Both validator and trainer independently enforce archive/member/expansion/file-count limits. Suggested defaults are embedded in the containers and can be overridden with DIMER environment configuration when the platform contract requires different limits.

## Artifact contract

Training output must preserve these files together:

```text
model.tabpfn_fit
model.ckpt
artifact_manifest.json
```

The `.tabpfn_fit` archive intentionally omits foundation weights. DIMER serving must use `serving/load_artifact.py`, or equivalent logic, so the restored estimator points to the relocated companion `model.ckpt`.

## Acceptance test

Before enabling the pipeline:

1. Build both component images from the exact commits in `COMPONENTS.json`.
2. Generate `examples/build_synthetic_dataset.py` output.
3. Validate it through the real validator image.
4. Fine-tune on a CUDA worker with a pinned/approved TabPFN checkpoint.
5. Confirm `result.json` contains validation and test metrics and artifact hashes.
6. Move the entire artifact directory to a clean serving environment.
7. Load using `load_dimer_tabpfn_regressor_artifact`.
8. Submit at least one inference batch containing rows whose expected output spans negative/positive values.
9. Verify no clipping or hidden transformation occurs.
10. Record the container image digests, component commits, TabPFN package version, base-checkpoint hash, output-checkpoint hash, and dataset hash.

Production enablement should remain blocked until this end-to-end acceptance test succeeds.

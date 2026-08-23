# TabPFN regressor contract v1

This directory is the machine-readable authority for the DIMER-facing regression worker contract introduced by `NAIRA-SEU/dimer-backend#4`.

`contract-v1.json` owns task identity, parameter constraints, TabPFN generation limits, dataset normalization/fingerprinting, and result/artifact requirements. Worker images vendor the same semantic contract. Missing `DIMER_CONTRACT_VERSION` is treated as v1 during migration; explicit unsupported versions fail closed. `DIMER_MODEL_CONFIG_JSON` is authoritative when it identifies a concrete generation; a conflicting legacy `model_version` is rejected.

`sha256-path-content-v1` fingerprints the logical dataset after path normalization and accepted single-nested-ZIP unwrapping, so equivalent directory/ZIP/nested-ZIP representations produce the same identity. DIMER must eventually persist the successful validation fingerprint and bind training to the same dataset; that backend work is documentation-only here.

`component-candidates.json` records immutable worker commits under review while release `COMPONENTS.json` remains unchanged. Vendored snapshots are checked semantically. With `DIMER_COMPONENT_TOKEN`, the conformance script fetches the exact private commits and verifies their recorded Git blob SHAs and semantic contract equality. Release pins advance only after component CI/review and conformance pass.

DIMER-side follow-ups remain external: pass validator/finetuner the same resolved context, support native `tabular_regression`, resolve model identity once, enforce activation compatibility, bind validation to training dataset identity, and complete serving E2E.

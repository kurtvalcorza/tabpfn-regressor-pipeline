# Development checks

```bash
pip install -r requirements-test.txt
python -m py_compile examples/build_synthetic_dataset.py serving/load_artifact.py
PYTHONPATH=. pytest -q -m "not integration"
ruff check .
python tools/validate_release_assets.py
python tools/build_notebook.py --check
python tools/build_notebook.py --check --template tools/notebook_template_artifact_inference.py
```

The standalone tutorials are generated from `tools/notebook_template*.py` and `src/tabpfn_regressor_pipeline/pipeline.py`; never edit the `.ipynb` files by hand.

When component CI/review passes, update `COMPONENTS.json` to the immutable merged validator and fine-tuner SHAs before deployment review. Do not copy component source into this repository.

# Development checks

```bash
pip install -r requirements-test.txt
python -m py_compile examples/build_synthetic_dataset.py serving/load_artifact.py
PYTHONPATH=. pytest -q
```

When component CI/review passes, update `COMPONENTS.json` to the immutable merged validator and fine-tuner SHAs before deployment review. Do not copy component source into this repository.

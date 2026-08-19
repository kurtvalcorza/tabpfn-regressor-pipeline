from examples.build_synthetic_dataset import build


def test_builder_returns_disjoint_splits_with_negative_targets() -> None:
    train, val, test = build(seed=7, rows=300)
    assert len(train) + len(val) + len(test) == 300
    assert list(train.columns) == ["x1", "x2", "category", "target"]
    assert train.target.min() < 0
    assert train.target.max() > 0
    assert len(train) > len(val) > 0
    assert len(test) > 0

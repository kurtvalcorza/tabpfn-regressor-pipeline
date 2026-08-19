from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build(seed: int, rows: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("synthetic-regression.zip"))
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 100:
        raise SystemExit("--rows must be at least 100")
    train, val, test = build(args.seed, args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train.csv", train.to_csv(index=False))
        archive.writestr("val.csv", val.to_csv(index=False))
        archive.writestr("test.csv", test.to_csv(index=False))
    print(f"wrote {args.out}: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()

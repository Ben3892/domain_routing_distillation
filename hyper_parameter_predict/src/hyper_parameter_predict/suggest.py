"""Rank candidate hyperparameter configurations using a saved model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("ranked_candidates.csv"))
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact = joblib.load(args.model)
    candidates = pd.read_csv(args.candidates)
    features = artifact["features"]
    missing = sorted(set(features) - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate file is missing features: {missing}")
    ranked = candidates.copy()
    ranked["predicted_validation_loss"] = artifact["estimator"].predict(
        candidates[features]
    )
    ranked = ranked.sort_values("predicted_validation_loss").head(args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print(ranked.to_string(index=False))
    print(f"\nSaved {len(ranked)} candidates to {args.output}")


if __name__ == "__main__":
    main()


"""Command-line training entry point."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "distill_hparam_matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .modeling import (
    EvaluationResult,
    available_model_names,
    evaluate_and_fit,
    partial_dependence_table,
)
from .schema import parse_feature_list, validate_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict validation loss from distillation hyperparameters."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    parser.add_argument("--target", default="validation_loss")
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated feature columns; default: 12 domain-specific parameters.",
    )
    parser.add_argument(
        "--group-column",
        default="experiment_id",
        help="Keep repeated seeds/configs in the same fold; pass an empty string to disable.",
    )
    parser.add_argument(
        "--models",
        default=",".join(available_model_names()),
        help="Comma-separated model names.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--search-iterations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    features = parse_feature_list(args.features)
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(model_names) - set(available_model_names()))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    if not model_names:
        raise ValueError("--models cannot be empty")

    frame = pd.read_csv(args.data)
    frame = validate_frame(frame, features, args.target, args.group_column)
    x = frame[features]
    y = frame[args.target]
    groups = frame[args.group_column] if args.group_column else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    results: List[EvaluationResult] = []
    failures: Dict[str, str] = {}
    for name in model_names:
        print(f"[{name}] nested cross-validation and final fit", flush=True)
        try:
            result = evaluate_and_fit(
                name=name,
                x=x,
                y=y,
                groups=groups,
                cv_folds=args.cv_folds,
                search_iterations=args.search_iterations,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
        except ImportError as exc:
            failures[name] = str(exc)
            print(f"[{name}] skipped: {exc}", flush=True)
            continue
        results.append(result)
        artifact = _artifact(result, features, args.target, args.group_column)
        joblib.dump(artifact, model_dir / f"{name}.joblib")
        print(f"[{name}] OOF MAE={result.metrics['mae']:.6f}", flush=True)

    if not results:
        raise RuntimeError(f"no model completed successfully: {failures}")
    results.sort(key=lambda item: item.metrics["mae"])
    best = results[0]
    joblib.dump(
        _artifact(best, features, args.target, args.group_column),
        args.output_dir / "best_model.joblib",
    )
    _write_reports(args, frame, features, results, failures, best)
    print(
        f"Best model: {best.name}; artifacts written to {args.output_dir}",
        flush=True,
    )


def _artifact(
    result: EvaluationResult,
    features: List[str],
    target: str,
    group_column: str,
) -> Dict[str, object]:
    return {
        "model_name": result.name,
        "estimator": result.estimator,
        "features": features,
        "target": target,
        "group_column": group_column,
        "cv_metrics": result.metrics,
        "best_params": result.best_params,
    }


def _write_reports(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    features: List[str],
    results: List[EvaluationResult],
    failures: Dict[str, str],
    best: EvaluationResult,
) -> None:
    metrics = pd.DataFrame(
        [{"model": result.name, **result.metrics} for result in results]
    ).sort_values("mae")
    metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)

    fold_rows = []
    prediction_columns = {
        "row_index": frame.index,
        "actual_validation_loss": frame[args.target],
    }
    for result in results:
        fold_rows.extend(
            {"model": result.name, **fold} for fold in result.fold_metrics
        )
        prediction_columns[f"predicted_{result.name}"] = result.oof_prediction
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_columns).to_csv(
        args.output_dir / "oof_predictions.csv", index=False
    )

    pdp = partial_dependence_table(best.estimator, frame[features], features)
    pdp.to_csv(args.output_dir / "partial_dependence.csv", index=False)
    _plot_partial_dependence(pdp, args.output_dir / "partial_dependence.png")

    summary = {
        "best_model": best.name,
        "selection_metric": "nested cross-validation OOF MAE (lower is better)",
        "rows": len(frame),
        "features": features,
        "target": args.target,
        "group_column": args.group_column or None,
        "cv_folds": args.cv_folds,
        "search_iterations": args.search_iterations,
        "best_params": {result.name: result.best_params for result in results},
        "skipped_models": failures,
        "note": "PDP values are predictive associations, not causal effects.",
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _plot_partial_dependence(pdp: pd.DataFrame, output: Path) -> None:
    features = pdp["feature"].unique().tolist()
    columns = 3
    rows = (len(features) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows), squeeze=False)
    for axis, feature in zip(axes.flat, features):
        subset = pdp[pdp["feature"] == feature]
        if pd.api.types.is_numeric_dtype(subset["value"]):
            axis.plot(subset["value"], subset["delta_from_baseline"], marker="o")
        else:
            axis.bar(subset["value"].astype(str), subset["delta_from_baseline"])
            axis.tick_params(axis="x", rotation=45)
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_title(feature)
        axis.set_ylabel("predicted loss delta")
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    figure.suptitle("Best-model partial dependence (association, not causation)")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

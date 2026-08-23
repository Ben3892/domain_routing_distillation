"""Dataset schema and validation helpers."""

from typing import Iterable, List

import numpy as np
import pandas as pd

DOMAINS = ("stem", "general", "math", "code")
PARAMETER_NAMES = ("temperature", "topk", "lm_loss_weight")
DEFAULT_FEATURES = tuple(
    f"{domain}_{parameter}"
    for domain in DOMAINS
    for parameter in PARAMETER_NAMES
)


def parse_feature_list(raw: str) -> List[str]:
    """Parse a comma-separated feature list, or use the canonical 12 features."""
    if not raw:
        return list(DEFAULT_FEATURES)
    features = [item.strip() for item in raw.split(",") if item.strip()]
    if not features:
        raise ValueError("--features cannot be empty")
    if len(features) != len(set(features)):
        raise ValueError("--features contains duplicate columns")
    return features


def validate_frame(
    frame: pd.DataFrame,
    features: Iterable[str],
    target: str,
    group_column: str = "",
) -> pd.DataFrame:
    """Validate required columns and constraints, returning a defensive copy."""
    features = list(features)
    required = features + [target]
    if group_column:
        required.append(group_column)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if not features:
        raise ValueError("at least one feature is required")

    clean = frame.copy()
    clean[target] = pd.to_numeric(clean[target], errors="coerce")
    for column in set(features).intersection(DEFAULT_FEATURES):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean[target].isna().any() or not np.isfinite(clean[target]).all():
        raise ValueError(f"target column {target!r} contains missing/non-finite values")
    if group_column and clean[group_column].isna().any():
        raise ValueError(f"group column {group_column!r} contains missing values")

    for domain in (*DOMAINS, ""):
        prefix = f"{domain}_" if domain else ""
        temperature = f"{prefix}temperature"
        topk = f"{prefix}topk"
        weight = f"{prefix}lm_loss_weight"
        for column in (temperature, topk, weight):
            if column in features:
                clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if temperature in features:
            _require_finite(clean, temperature)
            if (clean[temperature] <= 0).any():
                raise ValueError(f"{temperature} must be > 0")
        if topk in features:
            _require_finite(clean, topk)
            if (clean[topk] < 1).any() or not np.allclose(
                clean[topk], np.round(clean[topk])
            ):
                raise ValueError(f"{topk} must contain positive integers")
        if weight in features:
            _require_finite(clean, weight)
            if ((clean[weight] < 0) | (clean[weight] > 1)).any():
                raise ValueError(f"{weight} must be in [0, 1]")
    return clean


def _require_finite(frame: pd.DataFrame, column: str) -> None:
    if frame[column].isna().any() or not np.isfinite(frame[column]).all():
        raise ValueError(f"feature {column!r} contains missing/non-finite values")

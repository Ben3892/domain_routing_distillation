"""Model construction, leakage-safe evaluation, and relationship analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class EvaluationResult:
    name: str
    estimator: Pipeline
    best_params: Dict[str, Any]
    metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float]]
    oof_prediction: np.ndarray


def available_model_names() -> Tuple[str, ...]:
    return ("random_forest", "lightgbm", "xgboost", "mlp")


def make_model(
    name: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    seed: int,
    n_jobs: int,
) -> Tuple[Pipeline, Dict[str, Iterable[Any]]]:
    """Build a preprocessing + regressor pipeline and a compact search space."""
    preprocessing = _make_preprocessor(numeric_features, categorical_features)

    if name == "random_forest":
        regressor = RandomForestRegressor(
            random_state=seed, n_estimators=400, n_jobs=1
        )
        parameters = {
            "model__n_estimators": [250, 400, 700],
            "model__max_depth": [None, 6, 10, 16],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [0.5, 0.75, 1.0],
        }
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise ImportError(
                "LightGBM is unavailable; install the 'boosting' extra"
            ) from exc
        regressor = LGBMRegressor(
            objective="regression_l1",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        parameters = {
            "model__n_estimators": [200, 400, 700],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__num_leaves": [7, 15, 31, 63],
            "model__max_depth": [-1, 5, 8, 12],
            "model__min_child_samples": [5, 10, 20, 40],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
        }
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError(
                "XGBoost is unavailable; install the 'boosting' extra"
            ) from exc
        regressor = XGBRegressor(
            objective="reg:absoluteerror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        parameters = {
            "model__n_estimators": [200, 400, 700],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [3, 5, 7, 10],
            "model__min_child_weight": [1, 3, 7],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__reg_lambda": [0.5, 1.0, 5.0],
        }
    elif name == "mlp":
        regressor = TransformedTargetRegressor(
            regressor=MLPRegressor(
                random_state=seed,
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
            ),
            transformer=StandardScaler(),
        )
        parameters = {
            "model__regressor__hidden_layer_sizes": [
                (32,),
                (64,),
                (64, 32),
                (128, 64),
            ],
            "model__regressor__activation": ["relu", "tanh"],
            "model__regressor__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "model__regressor__learning_rate_init": [1e-4, 3e-4, 1e-3],
        }
    else:
        raise ValueError(f"unknown model {name!r}; choose from {available_model_names()}")

    pipeline = Pipeline([("preprocess", preprocessing), ("model", regressor)])
    return pipeline, parameters


def evaluate_and_fit(
    name: str,
    x: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    cv_folds: int,
    search_iterations: int,
    seed: int,
    n_jobs: int,
) -> EvaluationResult:
    """Run nested CV for an unbiased comparison, then tune on all rows."""
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in x.columns if column not in numeric]
    base_estimator, search_space = make_model(
        name, numeric, categorical, seed, n_jobs
    )
    outer_splits = make_splits(x, groups, cv_folds, seed)
    oof = np.full(len(x), np.nan, dtype=float)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_test, y_test = x.iloc[test_index], y.iloc[test_index]
        train_groups = groups.iloc[train_index] if groups is not None else None
        inner_splits = make_splits(x_train, train_groups, cv_folds, seed + fold)
        search = _make_search(
            clone(base_estimator),
            search_space,
            inner_splits,
            search_iterations,
            seed + fold,
            n_jobs,
        )
        search.fit(x_train, y_train)
        prediction = search.best_estimator_.predict(x_test)
        oof[test_index] = prediction
        fold_metrics.append(
            {"fold": fold, **regression_metrics(y_test.to_numpy(), prediction)}
        )

    if np.isnan(oof).any():
        raise RuntimeError("internal error: some rows did not receive an OOF prediction")

    final_splits = make_splits(x, groups, cv_folds, seed + 10_000)
    final_search = _make_search(
        clone(base_estimator),
        search_space,
        final_splits,
        search_iterations,
        seed + 10_000,
        n_jobs,
    )
    final_search.fit(x, y)
    return EvaluationResult(
        name=name,
        estimator=final_search.best_estimator_,
        best_params=final_search.best_params_,
        metrics=regression_metrics(y.to_numpy(), oof),
        fold_metrics=fold_metrics,
        oof_prediction=oof,
    )


def make_splits(
    x: pd.DataFrame,
    groups: Optional[pd.Series],
    folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("cv_folds must be >= 2")
    if groups is not None:
        unique_groups = groups.nunique()
        if unique_groups < folds:
            raise ValueError(
                f"group column has {unique_groups} unique values, fewer than {folds} folds"
            )
        splitter = GroupKFold(n_splits=folds)
        return list(splitter.split(x, groups=groups))
    if len(x) < folds:
        raise ValueError(f"dataset has {len(x)} rows, fewer than {folds} folds")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(x))


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy()
    if np.std(actual_rank) == 0 or np.std(predicted_rank) == 0:
        spearman = float("nan")
    else:
        spearman = float(np.corrcoef(actual_rank, predicted_rank)[0, 1])
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "spearman": spearman,
    }


def partial_dependence_table(
    estimator: Pipeline,
    x: pd.DataFrame,
    features: Sequence[str],
    max_grid_points: int = 20,
) -> pd.DataFrame:
    """Compute empirical 1-D PDPs by marginally replacing each feature."""
    baseline = float(np.mean(estimator.predict(x)))
    records: List[Dict[str, Any]] = []
    for feature in features:
        values = _feature_grid(x[feature], max_grid_points)
        for value in values:
            modified = x.copy()
            modified[feature] = value
            prediction = float(np.mean(estimator.predict(modified)))
            records.append(
                {
                    "feature": feature,
                    "value": value,
                    "mean_predicted_validation_loss": prediction,
                    "delta_from_baseline": prediction - baseline,
                }
            )
    return pd.DataFrame.from_records(records)


def _feature_grid(series: pd.Series, maximum: int) -> List[Any]:
    unique = series.dropna().unique().tolist()
    if len(unique) <= maximum:
        try:
            return sorted(unique)
        except TypeError:
            return unique
    if pd.api.types.is_numeric_dtype(series):
        quantiles = np.linspace(0.02, 0.98, maximum)
        return np.unique(series.quantile(quantiles).to_numpy()).tolist()
    counts = series.value_counts()
    return counts.head(maximum).index.tolist()


def _make_preprocessor(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_features),
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def _make_search(
    estimator: Pipeline,
    parameters: Dict[str, Iterable[Any]],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    requested_iterations: int,
    seed: int,
    n_jobs: int,
) -> RandomizedSearchCV:
    combinations = int(np.prod([len(list(values)) for values in parameters.values()]))
    iterations = min(requested_iterations, combinations)
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameters,
        n_iter=iterations,
        scoring="neg_mean_absolute_error",
        cv=splits,
        refit=True,
        random_state=seed,
        n_jobs=n_jobs,
        error_score="raise",
    )

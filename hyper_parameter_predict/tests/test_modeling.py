import numpy as np
import pandas as pd

from hyper_parameter_predict.modeling import make_splits, regression_metrics


def test_group_split_has_no_overlap():
    x = pd.DataFrame({"x": range(12)})
    groups = pd.Series(np.repeat(np.arange(6), 2))
    for train, test in make_splits(x, groups, folds=3, seed=7):
        assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))


def test_regression_metrics_are_exact_for_perfect_predictions():
    values = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(values, values)
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1
    assert metrics["spearman"] == 1


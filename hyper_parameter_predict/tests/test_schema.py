import pandas as pd
import pytest

from hyper_parameter_predict.schema import DEFAULT_FEATURES, validate_frame


def valid_frame():
    row = {}
    for feature in DEFAULT_FEATURES:
        if feature.endswith("temperature"):
            row[feature] = 1.0
        elif feature.endswith("topk"):
            row[feature] = 32
        else:
            row[feature] = 0.5
    row.update(experiment_id="exp-1", validation_loss=1.23)
    return pd.DataFrame([row])


def test_valid_frame_is_accepted():
    clean = validate_frame(
        valid_frame(), DEFAULT_FEATURES, "validation_loss", "experiment_id"
    )
    assert clean.shape == (1, 14)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("stem_temperature", 0, "must be > 0"),
        ("general_topk", 1.5, "positive integers"),
        ("math_lm_loss_weight", 1.1, "must be in [0, 1]"),
    ],
)
def test_invalid_hyperparameter_is_rejected(column, value, message):
    frame = valid_frame()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        validate_frame(frame, DEFAULT_FEATURES, "validation_loss", "experiment_id")

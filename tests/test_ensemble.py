import numpy as np
import pandas as pd
import pytest

from polymer_prediction.constants import TARGETS
from polymer_prediction.ensemble import blend_predictions


def write_inputs(tmp_path):
    paths = [tmp_path / f"{name}.csv" for name in ("graph", "catboost", "xgboost")]
    for i, path in enumerate(paths):
        frame = pd.DataFrame({"id": [2, 1], **{target: [2.0 + i, 1.0 + i] for target in TARGETS}})
        if i == 1:
            frame = frame.iloc[::-1]
        frame.to_csv(path, index=False)
    return paths


@pytest.mark.parametrize("source", range(3))
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_is_rejected_before_writing(tmp_path, source, invalid):
    paths = write_inputs(tmp_path)
    frame = pd.read_csv(paths[source])
    frame.loc[0, TARGETS[0]] = invalid
    frame.to_csv(paths[source], index=False)
    output = tmp_path / "submission.csv"
    output.write_text("previous valid submission\n")

    with pytest.raises(ValueError, match="finite"):
        blend_predictions(*paths, output, {"graph": 0.4, "catboost": 0.3, "xgboost": 0.3})
    assert output.read_text() == "previous valid submission\n"


def test_valid_predictions_keep_weighted_values_and_sorted_ids(tmp_path):
    paths = write_inputs(tmp_path)
    output = tmp_path / "nested" / "submission.csv"
    result = blend_predictions(*paths, output, {"graph": 0.4, "catboost": 0.3, "xgboost": 0.3})
    assert result["id"].tolist() == [1, 2]
    for target in TARGETS:
        np.testing.assert_allclose(result[target], [1.9, 2.9])
    pd.testing.assert_frame_equal(pd.read_csv(output), result)

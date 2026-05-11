from __future__ import annotations

from pathlib import Path

import pandas as pd

from polymer_prediction.constants import TARGETS


def blend_predictions(
    graph_path: str | Path,
    catboost_path: str | Path,
    xgboost_path: str | Path,
    output_path: str | Path,
    weights: dict[str, float],
) -> pd.DataFrame:
    graph = pd.read_csv(graph_path).sort_values("id").reset_index(drop=True)
    catboost = pd.read_csv(catboost_path).sort_values("id").reset_index(drop=True)
    xgboost = pd.read_csv(xgboost_path).sort_values("id").reset_index(drop=True)

    output = graph[["id"]].copy()
    for target in TARGETS:
        output[target] = (
            weights["graph"] * graph[target]
            + weights["catboost"] * catboost[target]
            + weights["xgboost"] * xgboost[target]
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output

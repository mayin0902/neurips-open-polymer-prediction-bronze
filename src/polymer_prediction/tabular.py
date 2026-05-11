from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polymer_prediction.constants import TARGETS
from polymer_prediction.descriptors import (
    build_mordred_descriptors,
    common_feature_columns,
    load_target_descriptors,
)


def make_regressor(model_name: str, seed: int = 10):
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(verbose=0, random_seed=seed, loss_function="MAE")
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=900,
            learning_rate=0.06,
            max_depth=5,
            reg_lambda=4.0,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=seed,
            objective="reg:squarederror",
        )
    raise ValueError(f"Unsupported model: {model_name}")


def infer_tabular(
    competition_dir: str | Path,
    descriptor_dir: str | Path,
    output_path: str | Path,
    model_name: str,
    seed: int = 10,
) -> pd.DataFrame:
    test = pd.read_csv(Path(competition_dir) / "test.csv")
    test_desc = build_mordred_descriptors(test["SMILES"].tolist())
    train_tables = load_target_descriptors(descriptor_dir)

    submission = pd.DataFrame({"id": test["id"].values})
    for target in TARGETS:
        train_df = train_tables[target].dropna(subset=[target])
        features = common_feature_columns(train_df, test_desc, target)
        model = make_regressor(model_name, seed=seed)
        model.fit(train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0), train_df[target])
        pred = model.predict(test_desc[features].replace([np.inf, -np.inf], np.nan).fillna(0))
        submission[target] = pred

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission

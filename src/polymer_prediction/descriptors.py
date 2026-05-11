from __future__ import annotations

from pathlib import Path

import pandas as pd

from polymer_prediction.constants import DESCRIPTOR_FILES, TARGETS


def load_target_descriptors(descriptor_dir: str | Path) -> dict[str, pd.DataFrame]:
    descriptor_dir = Path(descriptor_dir)
    tables = {}
    for target, filename in DESCRIPTOR_FILES.items():
        table = pd.read_csv(descriptor_dir / filename)
        constant_cols = [col for col in table.columns if table[col].nunique(dropna=False) <= 1]
        table = table.drop(columns=constant_cols).select_dtypes(exclude=["object", "category"])
        tables[target] = table
    return tables


def build_mordred_descriptors(smiles: list[str]) -> pd.DataFrame:
    from mordred import Calculator, descriptors
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    calc = Calculator(descriptors, ignore_3D=True)
    desc = calc.pandas(mols)
    return desc.select_dtypes(exclude=["object", "category"])


def common_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> list[str]:
    train_cols = set(train_df.columns) - {target}
    return sorted(train_cols & set(test_df.columns))


def empty_submission(ids) -> pd.DataFrame:
    output = pd.DataFrame({"id": ids})
    for target in TARGETS:
        output[target] = 0.0
    return output

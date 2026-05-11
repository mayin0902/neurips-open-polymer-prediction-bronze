from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


BAD_POLYMER_PATTERNS = (
    "[R]",
    "[R1]",
    "[R2]",
    "[R3]",
    "[R4]",
    "[R5]",
    "[R']",
    '[R"]',
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "([R])",
    "([R1])",
    "([R2])",
)


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    if any(pattern in smiles for pattern in BAD_POLYMER_PATTERNS):
        return None
    if "][" in smiles and any(token in smiles for token in ("[R", "R]")):
        return None

    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def clean_smiles_frame(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[smiles_col] = cleaned[smiles_col].map(canonicalize_smiles)
    return cleaned[cleaned[smiles_col].notna()].reset_index(drop=True)


def canonicalize_many(smiles: Iterable[str]) -> list[str | None]:
    return [canonicalize_smiles(item) for item in smiles]

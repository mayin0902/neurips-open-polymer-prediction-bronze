# NeurIPS Open Polymer Prediction 2025

![Kaggle](https://img.shields.io/badge/Kaggle-Bronze%20Medal-CD7F32)
![Rank](https://img.shields.io/badge/Rank-217%20%2F%202240-blue)
![Organizer](https://img.shields.io/badge/Organizer-NeurIPS%20%2F%20Kaggle-green)

**Kaggle Bronze Medal solution**, ranked **217 / 2240** in **NeurIPS - Open Polymer Prediction 2025**.

The competition focuses on molecular property prediction for polymers from SMILES strings. The final solution uses robust SMILES canonicalization, target-specific descriptor models, graph neural network features, and weighted blending across complementary branches.

## Result

| Field | Value |
| --- | --- |
| Competition | NeurIPS - Open Polymer Prediction 2025 |
| Organizer | NeurIPS / Kaggle |
| Medal | **Bronze Medal** |
| Rank | **217 / 2240** |
| Awarded | September 16, 2025 |
| Targets | `Tg`, `FFV`, `Tc`, `Density`, `Rg` |

## Method Overview

- Canonicalize SMILES and filter invalid polymer R-group notation before training and inference.
- Build target-specific descriptor tables with RDKit / Mordred features.
- Train CatBoost and XGBoost regressors per target, using only feature columns shared by train and test descriptors.
- Build a graph branch from atom and bond features with molecule-level descriptors.
- Blend GNN, CatBoost, and XGBoost predictions with fixed weights:

```text
0.4 * GNN + 0.3 * CatBoost + 0.3 * XGBoost
```

## Repository Layout

```text
configs/
  default.yaml
scripts/
  infer_tabular.py
  infer_graph.py
  blend.py
src/polymer_prediction/
  constants.py
  smiles.py
  descriptors.py
  tabular.py
  graph.py
  ensemble.py
```

## Quick Start

```bash
python -m pip install -e .
python scripts/infer_tabular.py --config configs/default.yaml --model catboost
python scripts/infer_tabular.py --config configs/default.yaml --model xgboost
python scripts/infer_graph.py --config configs/default.yaml --checkpoint /kaggle/input/polymer-gnn/model.pt
python scripts/blend.py --config configs/default.yaml
```

The default configuration follows Kaggle input paths. For local runs, edit `configs/default.yaml`.

## Notes

Raw competition files, external descriptor datasets, trained checkpoints, and submission artifacts are intentionally excluded. The repository keeps the reproducible solution code and configuration surface.

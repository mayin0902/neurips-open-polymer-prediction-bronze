from __future__ import annotations

from pathlib import Path

import pandas as pd

from polymer_prediction.constants import TARGETS
from polymer_prediction.smiles import clean_smiles_frame


def mol_to_graph(smiles: str):
    import torch
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetMass(),
                int(atom.IsInRing()),
            ]
        )

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = [
            float(bond.GetBondTypeAsDouble()),
            int(bond.GetIsAromatic()),
            int(bond.IsInRing()),
            int(bond.GetStereo()),
        ]
        edges.extend([(start, end), (end, start)])
        edge_features.extend([feature, feature])

    if not edges:
        edges = [(0, 0)]
        edge_features = [[0.0, 0, 0, 0]]

    mol_features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
    ]

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        mol_features=torch.tensor(mol_features, dtype=torch.float),
    )


class PolymerGraphRegressor:
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.0):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv, global_mean_pool

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList([GCNConv(8, hidden_dim)])
                self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
                for _ in range(num_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                self.mol_proj = nn.Sequential(nn.Linear(6, hidden_dim), nn.ReLU())
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, len(TARGETS)),
                )

            def forward(self, batch):
                x = batch.x
                for conv, norm in zip(self.convs, self.norms):
                    x = norm(conv(x, batch.edge_index)).relu()
                graph_vec = global_mean_pool(x, batch.batch)
                mol_vec = self.mol_proj(batch.mol_features.view(graph_vec.size(0), -1))
                return self.head(torch.cat([graph_vec, mol_vec], dim=1))

        self.model = Model()


def infer_graph(
    competition_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    batch_size: int = 128,
) -> pd.DataFrame:
    import torch
    from torch_geometric.loader import DataLoader

    test = pd.read_csv(Path(competition_dir) / "test.csv")
    test = clean_smiles_frame(test)
    graphs = [mol_to_graph(smiles) for smiles in test["SMILES"]]
    keep = [idx for idx, graph in enumerate(graphs) if graph is not None]
    graphs = [graphs[idx] for idx in keep]
    test = test.iloc[keep].reset_index(drop=True)

    wrapper = PolymerGraphRegressor()
    state = torch.load(checkpoint_path, map_location="cpu")
    wrapper.model.load_state_dict(state)
    wrapper.model.eval()

    preds = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            preds.append(wrapper.model(batch).cpu())
    pred = torch.cat(preds).numpy()

    submission = pd.DataFrame({"id": test["id"].values})
    for idx, target in enumerate(TARGETS):
        submission[target] = pred[:, idx]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission

# med_gnn/med_gnn.py
import torch
import torch.nn as nn
from .med_gnn_graph import MedGraph

class MedGNN(nn.Module):
    def __init__(self, graph: MedGraph | None = None, in_dim: int = 1, hidden: int = 16):
        super().__init__()
        self.graph = graph or MedGraph()
        self.A_base = self.graph.base_adj()
        self.disease_idx = self.graph.diseases
        self.n_classes = len(self.disease_idx)

        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, self.n_classes)

    def message_passing(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # Simple two-hop propagation with MLP between hops
        H = A_hat @ X
        H = torch.relu(self.lin1(H))
        H = A_hat @ H
        H = torch.relu(self.lin2(H))
        return H

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor | None = None) -> torch.Tensor:
        A_hat = self.A_base if A_hat is None else A_hat
        H = self.message_passing(X, A_hat)
        disease_H = H[self.disease_idx, :]
        # pool across disease nodes (nodes axis), keep feature dim -> [hidden]
        pooled = disease_H.mean(dim=0)
        # map hidden -> num_classes
        logits = self.readout(pooled) 
        return logits

    def forward_case(self, case: dict) -> torch.Tensor:
        X = self.graph.case_to_X(case)  # [N,1]
        return self.forward(X, self.A_base)

    def forward_patient(self, patient_id: str, current: dict, history: list[dict] | None = None,
                        hist_decay: float = 0.85) -> torch.Tensor:
        X_base = self.graph.case_to_X(current)  # [N,1]
        N = X_base.shape[0]

        A_aug = torch.zeros((N + 1, N + 1), dtype=torch.float32)
        A_aug[:N, :N] = self.A_base

        p_idx = N  # patient node index
        for group in ["symptoms", "labs", "treatments"]:
            for name, val in current.get(group, {}).items():
                if name in self.graph.idx and float(val) != 0.0:
                    j = self.graph.idx[name]
                    A_aug[p_idx, j] = 1.0
                    A_aug[j, p_idx] = 1.0

        if history:
            w = 1.0
            for event in reversed(history):
                w *= hist_decay
                for group in ["symptoms", "labs", "treatments"]:
                    for name, val in event.get(group, {}).items():
                        if name in self.graph.idx and float(val) != 0.0:
                            j = self.graph.idx[name]
                            A_aug[p_idx, j] = max(A_aug[p_idx, j], w)
                            A_aug[j, p_idx] = max(A_aug[j, p_idx], w)

        deg = A_aug.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(deg.clamp(min=1e-6), -0.5))
        A_hat = D_inv_sqrt @ A_aug @ D_inv_sqrt

        X_aug = torch.zeros((N + 1, 1), dtype=torch.float32)
        X_aug[:N, :] = X_base
        X_aug[p_idx, 0] = 1.0

        logits = self.forward(X_aug, A_hat)
        return logits

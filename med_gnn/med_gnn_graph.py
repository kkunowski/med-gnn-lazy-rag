import torch

NODE_TABLE = {
    "symptoms": ["fatigue", "cough", "chest_pain"],
    "diseases": ["pneumonia", "anemia", "myocardial_infarction"],
    "labs": ["hemoglobin", "CRP", "troponin"],
    "treatments": ["antibiotics", "iron_supplement", "aspirin"],
}

DEFAULT_WEIGHTS = {
    ("symptoms", "diseases"): 1.0,
    ("labs", "diseases"): 0.7,
    ("treatments", "diseases"): 0.4,
}

class MedGraph:
    def __init__(self, node_table: dict = NODE_TABLE, weights: dict = DEFAULT_WEIGHTS):
        self.node_table = node_table
        self.weights = weights
        self.groups = list(self.node_table.keys())
        self.all_nodes = [n for g in self.groups for n in self.node_table[g]]
        self.idx = {n: i for i, n in enumerate(self.all_nodes)}
        self.N = len(self.all_nodes)
        self.group_indices = {g: [self.idx[x] for x in self.node_table[g]] for g in self.groups}
        self.diseases = self.group_indices["diseases"]

    def base_adj(self) -> torch.Tensor:
        A = torch.zeros((self.N, self.N), dtype=torch.float32)
        for (src_group, dst_group), w in self.weights.items():
            for s in self.node_table[src_group]:
                for d in self.node_table[dst_group]:
                    i, j = self.idx[s], self.idx[d]
                    A[i, j] = max(A[i, j], w)
                    A[j, i] = max(A[j, i], w)
        A += torch.eye(self.N, dtype=torch.float32)
        deg = A.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(deg.clamp(min=1e-6), -0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def case_to_X(self, case: dict) -> torch.Tensor:
        X = torch.zeros((self.N, 1), dtype=torch.float32)
        for group in ["symptoms", "labs", "treatments"]:
            if group in case:
                for name, val in case[group].items():
                    if name in self.idx:
                        X[self.idx[name]] = float(val)
        return X

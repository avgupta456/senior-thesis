import torch
from torch_geometric.utils import degree

from src.explainers.explainer import Explainer
from src.utils import get_neighbors, sigmoid


class DegreeExplainer(Explainer):
    def __init__(self, pred_model, x, edge_index):
        super().__init__(pred_model, x, edge_index)

    def explain_edge(self, node_idx_1, node_idx_2, target):
        # TODO: Can use k_hop_subgraph here, not urgent
        neighbors = get_neighbors(self.edge_index, node_idx_1, node_idx_2)
        degrees = degree(self.edge_index[0], num_nodes=self.x.shape[0])
        mean_degree = torch.mean(degrees[neighbors])

        # minimize degree for positive edge (target=1), maximize otherwise
        mult = 1 if target == 1 else -1

        output = {}
        for neighbor in neighbors:
            logit = mult * (mean_degree - degrees[neighbor])
            output[neighbor] = sigmoid(logit)
        return output

import torch
from torch_geometric.utils import get_num_hops

from src.explainers.explainer import Explainer
from src.utils import get_neighbors_single_node, sigmoid


class EmbeddingExplainer(Explainer):
    def __init__(self, pred_model, x, edge_index):
        super().__init__(pred_model, x, edge_index)

        self.num_hops = get_num_hops(pred_model)

    def explain_edge(self, node_idx_1, node_idx_2, target):
        # TODO: Can use k_hop_subgraph here, not urgent
        x, edge_index = self.x, self.edge_index

        neighbors_1 = get_neighbors_single_node(edge_index, node_idx_1)
        if len(neighbors_1) > 1:
            label_index = torch.tensor([[node_idx_2 for _ in neighbors_1], neighbors_1])
            label_index = label_index.to(x.device)
            node_idx_1_sim = self.pred_model(x, edge_index, label_index)[1]
        else:
            node_idx_1_sim = torch.zeros(1)

        neighbors_2 = get_neighbors_single_node(edge_index, node_idx_2)
        if len(neighbors_2) > 1:
            label_index = torch.tensor([[node_idx_1 for _ in neighbors_2], neighbors_2])
            label_index = label_index.to(x.device)
            node_idx_2_sim = self.pred_model(x, edge_index, label_index)[1]
        else:
            node_idx_2_sim = torch.zeros(1)

        # maximize embedding similarity for positive edge (target=1), minimize otherwise
        mult = 1 if target == 1 else -1

        output = {}
        for i, neighbor in enumerate(neighbors_1):
            output[neighbor] = sigmoid(mult * node_idx_1_sim[i])
        for i, neighbor in enumerate(neighbors_2):
            new = sigmoid(mult * node_idx_2_sim[i])
            if neighbor in output:
                old = output[neighbor]
                output[neighbor] = max(old, new)
            else:
                output[neighbor] = new
        return output

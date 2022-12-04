from collections import defaultdict

import torch
from torch_geometric.utils import get_num_hops

from src.explainers.explainer import Explainer
from src.utils.neighbors import get_neighbors


class EmbeddingExplainer(Explainer):
    # Output is model probability of the edge to neighbor existing
    # Previously model probability of neighbor to other endpoint
    # But this fails for heterogeneous graphs
    # (A <-> B <-> C, but A <-> C does not exist)

    def __init__(self, pred_model):
        super().__init__(pred_model)

        self.num_hops = get_num_hops(pred_model)

    def explain_edge(self, data, node_idx_1, node_1_type, node_idx_2, node_2_type):
        output = defaultdict(int)
        z_dict = self.pred_model.encode(data.x_dict, data.edge_index_dict)

        # Node 1 neighbors
        neighbors_1 = get_neighbors(data, node_idx_1, node_1_type, 0, "")
        for k, v in neighbors_1.items():
            if len(v) == 0:
                continue

            n2_z = z_dict[node_2_type][node_idx_2]
            for neighbor in v:
                neighbor_z = z_dict[k][neighbor]
                sim = torch.cosine_similarity(n2_z, neighbor_z, dim=0)
                output[(k, neighbor)] = max(output[(k, neighbor)], sim)

        # Node 2 neighbors
        neighbors_2 = get_neighbors(data, node_idx_2, node_2_type, 0, "")
        for k, v in neighbors_2.items():
            if len(v) == 0:
                continue

            n1_z = z_dict[node_1_type][node_idx_1]
            for neighbor in v:
                neighbor_z = z_dict[k][neighbor]
                sim = torch.cosine_similarity(n1_z, neighbor_z, dim=0)
                output[(k, neighbor)] = max(output[(k, neighbor)], sim)

        return output

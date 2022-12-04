from collections import defaultdict

import torch
from torch_geometric.utils import get_num_hops

from src.explainers.explainer import Explainer
from src.utils.neighbors import get_neighbors
from src.utils.utils import device, sigmoid


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
        for n_idx, n_type in [(node_idx_1, node_1_type), (node_idx_2, node_2_type)]:
            neighbors_1 = get_neighbors(data, n_idx, n_type, 0, "")
            for k, v in neighbors_1.items():
                if len(v) == 0:
                    continue

                if (k, "to", n_type) in data.edge_index_dict:
                    edge_label_index = [v, [n_idx] * len(v)]
                    edge_key = (k, "to", n_type)
                elif (n_type, "to", k) in data.edge_index_dict:
                    edge_label_index = [[n_idx] * len(v), v]
                    edge_key = (n_type, "to", k)
                else:
                    continue

                edge_label_index = torch.tensor(edge_label_index).to(device)
                sim = self.pred_model.decode(
                    z_dict[edge_key[0]], z_dict[edge_key[2]], edge_label_index
                )

                for i, neighbor in enumerate(v):
                    value = sigmoid(sim[i]).item()
                    output[(k, neighbor)] = max(output[(k, neighbor)], value)

        return output

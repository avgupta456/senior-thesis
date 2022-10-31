import statistics

import numpy as np
from torch_geometric.utils import get_num_hops

from src.explainers.explainer import Explainer
from src.utils import edge_centered_subgraph, get_neighbors, mask_nodes


class SubgraphX(Explainer):
    def __init__(self, pred_model, x, edge_index, T=10):
        super().__init__(pred_model, x, edge_index)

        self.num_hops = get_num_hops(pred_model)
        self.T = T

    def explain_edge(self, node_idx_1, node_idx_2):
        # Only operate on a k-hop subgraph around `node_idx_1` and `node_idx_2.
        (x, edge_index, mapping, subset, _) = edge_centered_subgraph(
            node_idx_1, node_idx_2, self.x, self.edge_index, self.num_hops
        )
        edge_label_index = mapping.unsqueeze(1)

        neighbors = get_neighbors(edge_index, edge_label_index[0], edge_label_index[1])

        output = {}
        for neighbor in neighbors:
            pred_diffs = []
            for _ in range(self.T):
                S_filter = np.ones(x.shape[0], dtype=bool)
                S_filter[neighbors] = np.random.random(neighbors.shape[0]) > 0.5
                S_filter[neighbor] = False

                temp_x = mask_nodes(x, S_filter)
                old_pred = self.pred_model(temp_x, edge_index, edge_label_index)[1]

                temp_x[neighbor] = x[neighbor]
                new_pred = self.pred_model(temp_x, edge_index, edge_label_index)[1]

                pred_diff = new_pred - old_pred
                pred_diffs.append(pred_diff.item())

            diff_avg = sum(pred_diffs) / len(pred_diffs)
            diff_std = statistics.stdev(pred_diffs) / np.sqrt(self.T)
            logit = np.clip(diff_avg / diff_std, -10, 10)
            output[subset[neighbor].item()] = 1 / (1 + np.exp(-logit))

        return output

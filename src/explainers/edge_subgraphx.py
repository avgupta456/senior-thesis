import statistics

import numpy as np
import torch
from torch_geometric.utils import get_num_hops

from src.explainers.explainer import Explainer
from src.utils import edge_centered_subgraph, get_neighbors, sigmoid


class EdgeSubgraphX(Explainer):
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
            # Only optimizes the edges from neighbors to node_1/node_2, other direction not needed for prediction
            edge_mask_1 = edge_index[1] == edge_label_index[0]
            edge_mask_2 = edge_index[1] == edge_label_index[1]
            sub_edge_mask = edge_mask_1 | edge_mask_2

            pred_diffs = []
            for _ in range(self.T):
                rand_filter = np.random.random(sub_edge_mask.shape[0]) > 0.5
                S_filter = torch.ones(edge_index.shape[1], dtype=bool)
                S_filter[sub_edge_mask & rand_filter] = False

                # Used to be (sub_edge_mask & (edge_index[0] == neighbor))
                # Current method removes all edges from neighbor to other neighbors
                S_filter[(edge_index[0] == neighbor)] = False
                temp_edge_index = edge_index[:, S_filter]
                old_pred = self.pred_model(x, temp_edge_index, edge_label_index)[1]

                S_filter[(edge_index[0] == neighbor)] = True
                temp_edge_index = edge_index[:, S_filter]
                new_pred = self.pred_model(x, temp_edge_index, edge_label_index)[1]

                pred_diff = new_pred - old_pred
                pred_diffs.append(pred_diff.item())

            diff_avg = sum(pred_diffs) / len(pred_diffs)
            diff_std = statistics.stdev(pred_diffs) / np.sqrt(self.T)
            logit = diff_avg / diff_std / 10
            output[subset[neighbor].item()] = sigmoid(logit)

        return output

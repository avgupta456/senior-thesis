import statistics

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import get_num_hops

from src.explainers.explainer import Explainer
from src.utils import edge_centered_subgraph, get_neighbors, sigmoid


class EdgeSubgraphX(Explainer):
    def __init__(self, pred_model, x, edge_index, T=10):
        super().__init__(pred_model, x, edge_index)

        self.num_hops = get_num_hops(pred_model)
        self.T = T

    def explain_edge(self, node_idx_1, node_idx_2, target):
        # Only operate on a k-hop subgraph around `node_idx_1` and `node_idx_2.
        (x, edge_index, mapping, subset, _) = edge_centered_subgraph(
            node_idx_1, node_idx_2, self.x, self.edge_index, self.num_hops
        )
        edge_label_index = mapping.unsqueeze(1)

        # Only optimizes the edges from neighbors to node_1/node_2
        # Other direction not needed for prediction
        edge_mask_1 = edge_index[1] == edge_label_index[0]
        edge_mask_2 = edge_index[1] == edge_label_index[1]
        sub_edge_mask = edge_mask_1 | edge_mask_2

        neighbors = get_neighbors(edge_index, edge_label_index[0], edge_label_index[1])

        data_arr = []
        for neighbor in neighbors:
            pred_diffs = []
            for _ in range(self.T):
                S_filter = torch.zeros(edge_index.shape[1], dtype=bool)
                S_filter[sub_edge_mask] = True
                rand_filter = np.random.random(sub_edge_mask.shape[0]) > 0.5
                S_filter[(sub_edge_mask) & rand_filter] = False

                S_filter[(edge_index[0] == neighbor)] = False
                temp_data_1 = Data(
                    x, edge_index[:, S_filter], edge_label_index=edge_label_index
                )

                S_filter[(edge_index[0] == neighbor)] = True
                temp_data_2 = Data(
                    x, edge_index[:, S_filter], edge_label_index=edge_label_index
                )

                data_arr.extend([temp_data_1, temp_data_2])

        preds = []
        data_loader = DataLoader(data_arr, batch_size=128, shuffle=False)
        for batch in data_loader:
            z = self.pred_model.encode(batch.x, batch.edge_index)
            pred = self.pred_model.decode(z, batch.edge_label_index)
            pred = pred.squeeze().cpu().detach().tolist()
            preds.extend(pred)

        # maximize logit diff for positive edge (target=1), minimize otherwise
        mult = 1 if target == 1 else -1

        output = {}
        for i, neighbor in enumerate(neighbors):
            _preds = preds[2 * self.T * i : 2 * self.T * (i + 1)]
            pred_diffs = [_preds[2 * j + 1] - _preds[2 * j] for j in range(self.T)]
            diff_avg = sum(pred_diffs) / len(pred_diffs)
            diff_std = statistics.stdev(pred_diffs) / np.sqrt(self.T)
            logit = mult * diff_avg / diff_std / 10
            output[subset[neighbor].item()] = sigmoid(logit)

        return output

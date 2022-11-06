import statistics

import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import get_num_hops

from src.explainers.explainer import Explainer
from src.utils import edge_centered_subgraph, get_neighbors, mask_nodes, sigmoid


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

        data_arr = []
        for neighbor in neighbors:
            pred_diffs = []
            for _ in range(self.T):
                S_filter = np.ones(x.shape[0], dtype=bool)
                S_filter[neighbors] = np.random.random(neighbors.shape[0]) > 0.5
                S_filter[neighbor] = False

                temp_x = mask_nodes(x, S_filter)
                temp_data_1 = Data(
                    temp_x, edge_index, edge_label_index=edge_label_index
                )

                temp_x[neighbor] = x[neighbor]
                temp_data_2 = Data(
                    temp_x, edge_index, edge_label_index=edge_label_index
                )

                data_arr.extend([temp_data_1, temp_data_2])

        preds = []
        data_loader = DataLoader(data_arr, batch_size=128, shuffle=False)
        for batch in data_loader:
            z = self.pred_model.encode(batch.x, batch.edge_index)
            pred = self.pred_model.decode(z, batch.edge_label_index)
            pred = pred.squeeze().cpu().detach().tolist()
            preds.extend(pred)

        output = {}
        for i, neighbor in enumerate(neighbors):
            _preds = preds[2 * self.T * i : 2 * self.T * (i + 1)]
            pred_diffs = [_preds[2 * j + 1] - _preds[2 * j] for j in range(self.T)]
            diff_avg = sum(pred_diffs) / len(pred_diffs)
            diff_std = statistics.stdev(pred_diffs) / np.sqrt(self.T)
            logit = diff_avg / diff_std / 10
            output[subset[neighbor].item()] = sigmoid(logit)

        return output

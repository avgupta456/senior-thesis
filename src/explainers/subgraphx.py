import statistics

import numpy as np
import torch

from src.explainers.explainer import Explainer
from src.utils.neighbors import get_neighbors
from src.utils.utils import device, sigmoid


class SubgraphX(Explainer):
    def __init__(self, pred_model, T=10):
        super().__init__(pred_model)
        self.T = T

    def explain_edge(self, data, node_idx_1, node_1_type, node_idx_2, node_2_type):
        inputs = (data, node_idx_1, node_1_type, node_idx_2, node_2_type)
        neighbors = get_neighbors(*inputs)

        flat_neighbors = []
        for node_type in neighbors:
            flat_neighbors += [(node_type, n) for n in neighbors[node_type]]

        output = {}
        _label_index = torch.tensor([[node_idx_1], [node_idx_2]]).to(device)
        key = (node_1_type, "to", node_2_type)
        for node_type in neighbors:
            for neighbor in neighbors[node_type]:
                curr = (node_type, neighbor)
                pred_diffs = []
                for _ in range(self.T):
                    # nodes to remove, does not include the current node
                    r_nodes = [x for x in flat_neighbors if np.random.rand() < 0.5]
                    r_nodes = [x for x in r_nodes if x != curr]

                    # set r_nodes to 0
                    with_data = data.clone()
                    for node in r_nodes:
                        with_data.x_dict[node[0]][node[1]] = 0

                    with_pred = self.pred_model(
                        with_data.x_dict, with_data.edge_index_dict, _label_index, key
                    )[1].item()

                    # set curr to 0
                    with_data.x_dict[curr[0]][curr[1]] = 0
                    without_pred = self.pred_model(
                        with_data.x_dict,
                        with_data.edge_index_dict,
                        _label_index,
                        key,
                    )[1].item()

                    pred_diffs.append(with_pred - without_pred)

                diff_avg = sum(pred_diffs) / len(pred_diffs)
                diff_std = statistics.stdev(pred_diffs) / np.sqrt(self.T)
                logit = diff_avg / diff_std / 10
                output[(node_type, neighbor)] = sigmoid(logit)

        return output

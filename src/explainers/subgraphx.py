import statistics

import numpy as np
import torch

from src.explainers.explainer import Explainer


class SubgraphX(Explainer):
    def __init__(self, pred_model, x, edge_index, T=10):
        super().__init__(pred_model, x, edge_index)

        self.T = T

    def explain_edge(self, node_idx_1, node_idx_2):
        node_1_neighbors = set(
            self.edge_index[:, (self.edge_index[0] == node_idx_1)][1].cpu().numpy()
        )
        node_2_neighbors = set(
            self.edge_index[:, (self.edge_index[0] == node_idx_2)][1].cpu().numpy()
        )
        neighbors = np.array(list(node_1_neighbors.union(node_2_neighbors)))

        output = {}
        for neighbor in neighbors:
            pred_diffs = []
            for _ in range(self.T):
                temp_x = self.x.clone()
                fake_x = temp_x.clone()
                fake_x[neighbors] = 0

                S_filter = neighbors[np.random.random(neighbors.shape[0]) > 0.5]
                fake_x[S_filter] = temp_x[S_filter]
                fake_x[neighbor] = 0
                old_z = self.pred_model.encode(fake_x, self.edge_index)
                old_pred = self.pred_model.decode(
                    old_z, torch.tensor([[node_idx_1], [node_idx_2]])
                )

                fake_x[neighbor] = temp_x[neighbor]
                new_z = self.pred_model.encode(fake_x, self.edge_index)
                new_pred = self.pred_model.decode(
                    new_z, torch.tensor([[node_idx_1], [node_idx_2]])
                )

                pred_diff = new_pred - old_pred
                pred_diffs.append(pred_diff.item())

            diff_avg, diff_std = sum(pred_diffs) / len(pred_diffs), statistics.stdev(
                pred_diffs
            ) / np.sqrt(self.T)
            output[neighbor] = 1 / (1 + np.exp(-(diff_avg / diff_std)))

        return output

import numpy as np

from explainers.explainer import Explainer

class RandomExplainer(Explainer):
    def __init__(self, pred_model, x, edge_index):
        super().__init__(pred_model, x, edge_index)

    def explain_edge(self, node_idx_1, node_idx_2):
        node_1_neighbors = set(self.edge_index[:, (self.edge_index[0] == node_idx_1)][1].cpu().numpy())
        node_2_neighbors = set(self.edge_index[:, (self.edge_index[0] == node_idx_2)][1].cpu().numpy())
        neighbors = np.array(list(node_1_neighbors.union(node_2_neighbors)))

        output = {}
        for neighbor in neighbors:
            output[neighbor] = np.random.rand()
        return output
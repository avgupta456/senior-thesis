import numpy as np

from src.explainers.explainer import Explainer
from src.utils import get_neighbors


class RandomExplainer(Explainer):
    def __init__(self, pred_model, x, edge_index):
        super().__init__(pred_model, x, edge_index)

    def explain_edge(self, node_idx_1, node_idx_2, target):
        neighbors = get_neighbors(self.edge_index, node_idx_1, node_idx_2)

        output = {}
        for neighbor in neighbors:
            output[neighbor] = np.random.rand()
        return output

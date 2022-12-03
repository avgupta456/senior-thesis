import numpy as np

from src.explainers.explainer import Explainer
from src.utils.neighbors import get_neighbors


class RandomExplainer(Explainer):
    def __init__(self, pred_model):
        super().__init__(pred_model)

    def explain_edge(
        self, data, node_idx_1, node_1_type, node_idx_2, node_2_type, target
    ):
        neighbors = get_neighbors(
            data, node_idx_1, node_1_type, node_idx_2, node_2_type
        )

        output = {}
        for node_type in neighbors:
            for neighbor in neighbors[node_type]:
                output[(node_type, neighbor)] = np.random.rand()
        return output

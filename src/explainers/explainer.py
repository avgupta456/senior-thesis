class Explainer:
    def __init__(self, pred_model):
        self.pred_model = pred_model

    def explain_edge(self, data, node_idx_1, node_1_type, node_idx_2, node_2_type):
        raise NotImplementedError

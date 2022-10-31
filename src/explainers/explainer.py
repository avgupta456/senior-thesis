class Explainer:
    def __init__(self, pred_model, x, edge_index):
        self.pred_model = pred_model
        self.x = x
        self.edge_index = edge_index

    def explain_edge(self, node_idx_1, node_idx_2):
        raise NotImplementedError

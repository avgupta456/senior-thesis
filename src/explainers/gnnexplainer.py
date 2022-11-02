from math import sqrt

import torch
from torch_geometric.nn import GNNExplainer as PyG_GNNExplainer
from torch_geometric.nn.models.explainer import set_masks

from src.explainers.explainer import Explainer
from src.utils import edge_centered_subgraph

EPS = 1e-15


class _GNNExplainer(PyG_GNNExplainer):
    coeffs = {
        "edge_size": 0.10,
        "edge_reduction": "sum",
        "edge_ent": -1.0,
    }

    def _initialize_masks(self, x, edge_index, sub_edge_mask=None):
        (N, F), E = x.size(), edge_index.size(1)
        self.node_feat_mask = torch.nn.Parameter(100 * torch.ones(1, F))

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        if sub_edge_mask is None:
            self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        else:
            E_1, mask = sub_edge_mask.sum(), 100 * torch.ones(E)
            mask[sub_edge_mask] = torch.randn(E_1) * std
            self.edge_mask = torch.nn.Parameter(mask)

    def _loss(self, log_logits, prediction, node_idx=None):
        error_loss = -log_logits[prediction]

        m = self.edge_mask[self.sub_edge_mask].sigmoid()
        edge_reduce = getattr(torch, self.coeffs["edge_reduction"])
        edge_size_loss = edge_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        edge_ent_loss = ent.mean()

        loss = (
            error_loss
            + self.coeffs["edge_size"] * edge_size_loss
            + self.coeffs["edge_ent"] * edge_ent_loss
        )

        return loss

    def explain_edge(self, node_idx_1, node_idx_2, x, edge_index):
        self.model.eval()
        self._clear_masks()

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx_1` and `node_idx_2.
        (
            x,
            edge_index,
            mapping,
            _,
            hard_edge_mask,
        ) = edge_centered_subgraph(node_idx_1, node_idx_2, x, edge_index, self.num_hops)

        # Only optimizes the edges from neighbors to node_1/node_2, other direction not needed for prediction
        self.sub_edge_mask = (edge_index[1] == mapping[0]) | (
            edge_index[1] == mapping[1]
        )
        edge_label_index = mapping.unsqueeze(1)

        # Get the initial prediction
        prediction = self.get_initial_prediction(
            x, edge_index, edge_label_index=edge_label_index
        )

        self._initialize_masks(x, edge_index, self.sub_edge_mask)
        self.to(x.device)

        set_masks(self.model, self.edge_mask, edge_index, apply_sigmoid=True)
        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(
                x=x, edge_index=edge_index, edge_label_index=edge_label_index
            )
            loss = self.get_loss(out, prediction, mapping).mean()
            loss.backward()
            optimizer.step()

        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self._clear_masks()

        return edge_mask


class GNNExplainer(Explainer):
    def __init__(self, pred_model, x, edge_index, epochs=200, lr=1e-2):
        super().__init__(pred_model, x, edge_index)
        self.explainer = _GNNExplainer(pred_model, epochs=epochs, lr=lr)

    def explain_edge(self, node_idx_1, node_idx_2):
        edge_mask = self.explainer.explain_edge(
            node_idx_1, node_idx_2, self.x, self.edge_index
        )

        output = {}
        edge_filter = (self.edge_index[1] == node_idx_1) | (
            self.edge_index[1] == node_idx_2
        )
        temp_edge_index = self.edge_index[0, edge_filter].cpu().numpy()
        temp_edge_mask = edge_mask[edge_filter].cpu().numpy()
        for node_idx, weight in zip(temp_edge_index, temp_edge_mask):
            output[node_idx] = (output.get(node_idx, weight) + weight) / 2

        return output

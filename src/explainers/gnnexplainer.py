from math import sqrt

import torch
from torch_geometric.nn import MessagePassing

from src.explainers.explainer import Explainer

EPS = 1e-15


def set_masks(model, mask, edge_index, apply_sigmoid=True):
    loop_mask = edge_index[0] != edge_index[1]

    # Loop over layers and set masks on MessagePassing layers:
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.explain = True
            module._edge_mask = mask
            module._loop_mask = loop_mask
            module._apply_sigmoid = apply_sigmoid


def set_hetero_masks(model, mask_dict, edge_index_dict, apply_sigmoid=True):
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleDict):
            for edge_type in mask_dict.keys():
                str_edge_type = "__".join(edge_type)
                if str_edge_type in module:
                    set_masks(
                        module[str_edge_type],
                        mask_dict[edge_type],
                        edge_index_dict[edge_type],
                        apply_sigmoid=apply_sigmoid,
                    )


def clear_masks(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.explain = False
            module._edge_mask = None
            module._loop_mask = None
            module._apply_sigmoid = True
    return module


class _GNNExplainer:
    # overwritten, dataset specific
    coeffs = {
        "edge_size": None,
        "edge_ent": None,
    }

    def __init__(self, model, epochs, lr, **kwargs):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.coeffs.update(kwargs)
        self._clear_masks()

    def _initialize_masks(self, N, edge_index_dict, sub_edge_mask):
        self.edge_mask = {}
        std = sqrt(2 / N)
        for k, v in edge_index_dict.items():
            E = v.size(1)
            E_1, mask = sub_edge_mask[k].sum(), 100 * torch.ones(E)
            mask[sub_edge_mask[k]] = torch.randn(E_1) * std
            self.edge_mask[k] = torch.nn.Parameter(mask)

    def _clear_masks(self):
        clear_masks(self.model)
        self.edge_mask = None

    def get_loss(self, log_logits, prediction):
        error_loss = -log_logits[prediction]

        m = []
        for k, v in self.edge_mask.items():
            m.append(v[self.sub_edge_mask[k]])
        m = torch.cat(m).sigmoid()

        edge_size_loss = (torch.mean(m) - 0.5) ** 2
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        edge_ent_loss = ent.mean()

        loss = (
            error_loss
            + self.coeffs["edge_size"] * edge_size_loss
            + self.coeffs["edge_ent"] * edge_ent_loss
        )

        return loss

    def explain_edge(self, data, node_idx_1, node_1_type, node_idx_2, node_2_type):
        self.model.eval()
        self._clear_masks()

        sub_edge_mask = {
            k: torch.zeros(v.size(1), dtype=torch.bool)
            for k, v in data.edge_index_dict.items()
        }

        for k, v in data.edge_index_dict.items():
            if k[2] == node_1_type:
                sub_edge_mask[k][v[1] == node_idx_1] = True
            if k[2] == node_2_type:
                sub_edge_mask[k][v[1] == node_idx_2] = True

        self.sub_edge_mask = sub_edge_mask

        N = sum([sum(v) for v in sub_edge_mask.values()])
        self._initialize_masks(N, data.edge_index_dict, self.sub_edge_mask)

        set_hetero_masks(
            self.model, self.edge_mask, data.edge_index_dict, apply_sigmoid=True
        )

        optimizer = torch.optim.Adam(params=self.edge_mask.values(), lr=self.lr)

        edge_label_index = torch.tensor([[node_idx_1], [node_idx_2]])
        key = (node_1_type, "to", node_2_type)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(data.x_dict, data.edge_index_dict, edge_label_index, key)
            loss = self.get_loss(out, 1).mean()
            loss.backward()
            optimizer.step()

        out = self.edge_mask, self.sub_edge_mask

        self._clear_masks()

        return out


class GNNExplainer(Explainer):
    def __init__(self, pred_model, epochs, lr, **kwargs):
        super().__init__(pred_model)
        self.explainer = _GNNExplainer(pred_model, epochs=epochs, lr=lr, **kwargs)

    def explain_edge(self, data, node_idx_1, node_1_type, node_idx_2, node_2_type):
        edge_mask, sub_edge_mask = self.explainer.explain_edge(
            data, node_idx_1, node_1_type, node_idx_2, node_2_type
        )

        output = {}
        for k, v in edge_mask.items():
            edge_index = data.edge_index_dict[k]
            for n, val in zip(edge_index[0, sub_edge_mask[k]], v[sub_edge_mask[k]]):
                output[(k[0], n.item())] = val.sigmoid().item()

        return output

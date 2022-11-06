import numpy as np
import torch
from numpy import ndarray
from torch_geometric.utils import k_hop_subgraph


def sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    return 1 / (1 + np.exp(-x))


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return tensor_to_numpy(x)
    return x


def edge_centered_subgraph(node_idx_1, node_idx_2, x, edge_index, num_hops):
    num_nodes = x.size(0)

    subset_1, _, _, edge_mask_1 = k_hop_subgraph(
        node_idx_1, num_hops, edge_index, num_nodes=num_nodes
    )
    subset_2, _, _, edge_mask_2 = k_hop_subgraph(
        node_idx_2, num_hops, edge_index, num_nodes=num_nodes
    )

    # Combines two node-centered subgraphs
    temp_node_idx = edge_index[0].new_full((num_nodes,), -1)  # full size
    edge_mask = edge_mask_1 | edge_mask_2
    edge_index = edge_index[:, edge_mask]  # filters out edges
    subset = torch.cat((subset_1, subset_2)).unique()
    temp_node_idx[subset] = torch.arange(subset.size(0), device=edge_index.device)
    edge_index = temp_node_idx[edge_index]  # maps edge_index to [0, n]
    x = x[subset]  # filters out nodes
    mapping = torch.tensor(
        [
            (subset == node_idx_1).nonzero().item(),
            (subset == node_idx_2).nonzero().item(),
        ]
    )

    return x, edge_index, mapping, subset, edge_mask


def get_neighbors_single_node(edge_index, node_idx: int) -> ndarray:
    return edge_index[:, (edge_index[0] == node_idx)][1].cpu().numpy()


def get_neighbors(edge_index, node_idx_1: int, node_idx_2: int) -> ndarray:
    node_1_neighbors = edge_index[:, (edge_index[0] == node_idx_1)][1].cpu().numpy()
    node_2_neighbors = edge_index[:, (edge_index[0] == node_idx_2)][1].cpu().numpy()
    neighbors = np.array(list(set(node_1_neighbors).union(set(node_2_neighbors))))
    return neighbors


def mask_nodes(x, mask):
    new_x = x.clone()
    new_x[~mask] = 0
    return new_x

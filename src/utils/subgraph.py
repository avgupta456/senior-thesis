import torch
from torch_geometric.data import HeteroData

from src.utils.utils import device


def mask_nodes(x, mask):
    new_x = x.clone()
    return new_x[mask]


def mask_edges(edge_index, src_mask, dst_mask):
    new_edge_index = edge_index.clone()
    new_edge_index[0, ~src_mask[edge_index[0]]] = -1
    new_edge_index[1, ~dst_mask[edge_index[1]]] = -1
    new_edge_index = new_edge_index[
        :, (new_edge_index[0] >= 0) & (new_edge_index[1] >= 0)
    ]

    # map from old to new indices
    new_src = torch.zeros(src_mask.size(0), dtype=torch.long) - 1
    new_src[src_mask] = torch.arange(src_mask.sum())
    new_dst = torch.zeros(dst_mask.size(0), dtype=torch.long) - 1
    new_dst[dst_mask] = torch.arange(dst_mask.sum())
    new_edge_index[0] = new_src[new_edge_index[0]]
    new_edge_index[1] = new_dst[new_edge_index[1]]

    return new_edge_index


def apply_node_mask(data, node_mask):
    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict

    x_dict = {k: mask_nodes(v, node_mask[k]) for k, v in x_dict.items()}
    edge_index_dict = {
        k: mask_edges(v, node_mask[k[0]], node_mask[k[2]])
        for k, v in edge_index_dict.items()
    }

    new_data = HeteroData()
    new_data.x_dict = x_dict
    new_data.edge_index_dict = edge_index_dict
    new_data = new_data.to(device)

    return new_data


def apply_edge_mask(data, edge_mask):
    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict
    edge_index_dict = {k: v[:, edge_mask[k]] for k, v in edge_index_dict.items()}

    new_data = HeteroData()
    new_data.x_dict = x_dict
    new_data.edge_index_dict = edge_index_dict
    new_data = new_data.to(device)

    return new_data


def edge_centered_subgraph(
    node_idx_1, node_1_type, node_idx_2, node_2_type, data, num_hops
):
    x_dict = data.x_dict

    x_mask = {k: torch.zeros(v.size(0), dtype=torch.bool) for k, v in x_dict.items()}
    x_mask[node_1_type][node_idx_1] = True
    x_mask[node_2_type][node_idx_2] = True

    for _ in range(num_hops):
        new_masks = {k: v for k, v in x_mask.items()}
        for k, v in data.edge_index_dict.items():
            if k[1] == "rev_to":
                continue

            curr_src = x_mask[k[0]].nonzero().view(-1)
            curr_dst = x_mask[k[2]].nonzero().view(-1)

            new_dst = v[1, torch.isin(v[0], curr_src)]
            new_src = v[0, torch.isin(v[1], curr_dst)]

            new_src_mask = torch.zeros(x_dict[k[0]].size(0), dtype=torch.bool)
            new_src_mask[new_src] = True

            new_dst_mask = torch.zeros(x_dict[k[2]].size(0), dtype=torch.bool)
            new_dst_mask[new_dst] = True

            new_masks[k[0]] = new_masks[k[0]] | new_src_mask
            new_masks[k[2]] = new_masks[k[2]] | new_dst_mask

        x_mask = new_masks

    # find place in masked graph of node_idx_1
    new_node_idx_1 = x_mask[node_1_type].nonzero().view(-1).tolist().index(node_idx_1)
    new_node_idx_2 = x_mask[node_2_type].nonzero().view(-1).tolist().index(node_idx_2)

    return apply_node_mask(data, x_mask), new_node_idx_1, new_node_idx_2


def remove_edge_connections(
    data, node_idx_1, node_1_type, node_idx_2, node_2_type, edges
):
    # Ex: edges = [("actor", 2772), ("director", 1331), ...]

    edge_mask = {
        k: torch.ones(v.size(1), dtype=torch.bool)
        for k, v in data.edge_index_dict.items()
    }

    for node_type, node_idx in edges:
        for k, v in data.edge_index_dict.items():
            if k[0] == node_type and k[2] == node_1_type:
                edge_mask[k][(v[0] == node_idx) & (v[1] == node_idx_1)] = False
            if k[0] == node_type and k[2] == node_2_type:
                edge_mask[k][(v[0] == node_idx) & (v[1] == node_idx_2)] = False
            if k[0] == node_1_type and k[2] == node_type:
                edge_mask[k][(v[0] == node_idx_1) & (v[1] == node_idx)] = False
            if k[0] == node_2_type and k[2] == node_type:
                edge_mask[k][(v[0] == node_idx_2) & (v[1] == node_idx)] = False

    return apply_edge_mask(data, edge_mask)

from collections import defaultdict


def get_neighbors(data, node_idx_1, node_1_type, node_idx_2, node_2_type):
    neighbors = defaultdict(list)
    for k, v in data.edge_index_dict.items():
        if k[0] == node_1_type:
            neighbors[k[2]].extend(list(v[1, v[0] == node_idx_1].numpy()))

        if k[0] == node_2_type:
            neighbors[k[2]].extend(list(v[1, v[0] == node_idx_2].numpy()))

        if k[2] == node_1_type:
            neighbors[k[0]].extend(list(v[0, v[1] == node_idx_1].numpy()))

        if k[2] == node_2_type:
            neighbors[k[0]].extend(list(v[0, v[1] == node_idx_2].numpy()))

    neighbors = {k: list(set(v)) for k, v in neighbors.items()}
    return neighbors

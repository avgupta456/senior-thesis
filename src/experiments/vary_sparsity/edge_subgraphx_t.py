import torch

from src.dataset import dataset, device, test_data
from src.pred import Net
from src.experiments.vary_sparsity.base import (
    run_experiment,
    sample_edge_subgraphx as _sample_edge_subgraphx,
)


def sample_edge_subgraph(T):
    def inner_sample_edge_subgraph(model, x, edge_index, node_idx_1, node_idx_2):
        return _sample_edge_subgraphx(model, x, edge_index, node_idx_1, node_idx_2, T=T)

    return inner_sample_edge_subgraph


if __name__ == "__main__":
    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))
    run_experiment(
        model,
        test_data.x,
        test_data.edge_index,
        test_data.edge_label_index[:, 10:11],
        [sample_edge_subgraph(T) for T in [2, 5, 10, 20]],
        ["EdgeSubgraphX T={}".format(T) for T in [2, 5, 10, 20]],
        show_plots=True,
    )

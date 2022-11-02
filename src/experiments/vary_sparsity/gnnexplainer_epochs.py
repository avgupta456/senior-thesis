import torch

from src.dataset import dataset, device, test_data
from src.pred import Net
from src.experiments.vary_sparsity.base import (
    run_experiment,
    sample_gnnexplainer as _sample_gnnexplainer,
)


def sample_gnnexplainer(epochs):
    def inner_sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2):
        return _sample_gnnexplainer(
            model, x, edge_index, node_idx_1, node_idx_2, epochs=epochs
        )

    return inner_sample_gnnexplainer


if __name__ == "__main__":
    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))
    run_experiment(
        model,
        test_data.x,
        test_data.edge_index,
        test_data.edge_label_index[:, 10:11],
        [sample_gnnexplainer(epochs) for epochs in [20, 50, 100, 200]],
        ["GNNExplainer epochs={}".format(epochs) for epochs in [20, 50, 100, 200]],
    )

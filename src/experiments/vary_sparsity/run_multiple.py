import json

import torch

from src.dataset import dataset, device, test_data
from src.experiments.vary_sparsity.base import run_experiment
from src.explainers.main import (
    sample_degree,
    sample_edge_subgraphx,
    sample_embedding,
    sample_gnnexplainer,
    sample_random,
    sample_subgraphx,
)
from src.pred import Net


def get_data(start, stop):
    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))
    all_data = run_experiment(
        model,
        test_data.x,
        test_data.edge_index,
        test_data.edge_label_index[:, start : stop + 1],
        test_data.edge_label[start : stop + 1],
        [
            sample_gnnexplainer,
            sample_subgraphx,
            sample_edge_subgraphx,
            sample_embedding,
            sample_degree,
            sample_random,
        ],
        ["GNNExplainer", "SubgraphX", "EdgeSubgraphX", "Embedding", "Degree", "Random"],
    )
    return all_data


if __name__ == "__main__":
    start, stop = 0, 600
    all_data = get_data(start, stop)
    with open(f"./results/vary_sparsity/data_{start}_{stop}.json", "w") as f:
        json.dump(all_data, f)

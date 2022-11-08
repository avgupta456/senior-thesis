import json

import torch


def get_data():
    from src.dataset import dataset, device, test_data
    from src.experiments.vary_sparsity.base import (
        run_experiment,
        sample_degree,
        sample_edge_subgraphx,
        sample_embedding,
        sample_gnnexplainer,
        sample_random,
        sample_subgraphx,
    )
    from src.pred import Net

    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))
    all_data = run_experiment(
        model,
        test_data.x,
        test_data.edge_index,
        test_data.edge_label_index[:, 0:600],
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
    all_data = get_data()
    with open("./results/vary_sparsity/data_600.json", "w") as f:
        json.dump(all_data, f)

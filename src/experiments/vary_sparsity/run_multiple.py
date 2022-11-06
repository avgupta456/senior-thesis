import json
from collections import defaultdict

import matplotlib.pyplot as plt
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
        test_data.edge_label_index[:, 50:200],
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


def plot_data(all_data):
    sampler_names = [
        "GNNExplainer",
        "SubgraphX",
        "EdgeSubgraphX",
        "Embedding",
        "Degree",
        "Random",
    ]

    count = 0
    x_samples = 1000
    merged_necessary_results = defaultdict(lambda: [0 for _ in range(x_samples)])
    merged_sufficient_results = defaultdict(lambda: [0 for _ in range(x_samples)])
    merged_results = defaultdict(lambda: [0 for _ in range(x_samples)])
    for i, data in all_data.items():
        n = len(data["sufficient"][sampler_names[0]])
        # Filters out examples where the two nodes individually predict an edge
        if data["sufficient"]["Random"][0] > 0.5:
            continue
        # Or where there are fewer than 10 neighbors
        if n < 10:
            continue

        count += 1
        for name in sampler_names:
            for j in range(x_samples):
                best_x = int(j / x_samples * n)
                merged_sufficient_results[name][j] += data["sufficient"][name][best_x]
                merged_necessary_results[name][j] += data["necessary"][name][best_x]
                merged_results[name][j] += data["characterization"][name][best_x]

    cutoff = int(0.8 * x_samples)
    for result, result_label in zip(
        [merged_sufficient_results, merged_necessary_results, merged_results],
        ["(1 - Fidelity-)", "Fidelity+", "Characterization"],
    ):
        fig, ax = plt.subplots()
        for sampler_name in sampler_names:
            ax.plot(
                [x / x_samples for x in range(cutoff)],
                [x / count for x in result[sampler_name][:cutoff]],
                label=sampler_name,
            )
        ax.set_xlabel("Sparsity")
        ax.set_ylabel(result_label)
        ax.set_title(f"{result_label} vs Sparsity (n={count})")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    create_data = True

    if create_data:
        all_data = get_data()
        with open("./results/vary_sparsity/data.json", "w") as f:
            json.dump(all_data, f)
    else:
        with open("./results/vary_sparsity/data.json", "r") as f:
            all_data = json.load(f)
        plot_data(all_data)

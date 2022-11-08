import json
from collections import defaultdict

import matplotlib.pyplot as plt


def plot_data(all_data, filter=None):
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

        if filter == "small" and n > 50:
            continue
        if filter == "medium" and (n < 50 or n > 200):
            continue
        if filter == "large" and n < 200:
            continue

        count += 1
        for name in sampler_names:
            for j in range(x_samples):
                best_x = int(j / x_samples * n)
                merged_sufficient_results[name][j] += data["sufficient"][name][best_x]
                merged_necessary_results[name][j] += data["necessary"][name][best_x]
                merged_results[name][j] += data["characterization"][name][best_x]

    cutoff = int(x_samples)
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
    with open("./results/vary_sparsity/data_600.json", "r") as f:
        all_data = json.load(f)
    plot_data(all_data, "large")

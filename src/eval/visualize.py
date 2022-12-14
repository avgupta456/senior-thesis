import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.eval.utils import process_data


def plot_continuous_sparsity(
    dataset_name, processed_data, sampler_names, sampler_display_names, colors
):
    count = len(processed_data)
    x_samples = 1000
    merged_necessary_results = defaultdict(lambda: [0 for _ in range(x_samples)])
    merged_sufficient_results = defaultdict(lambda: [0 for _ in range(x_samples)])
    merged_results = defaultdict(lambda: [0 for _ in range(x_samples)])
    for _, data in processed_data.items():
        n = len(data["Random"])
        for name in sampler_names:
            for j in range(x_samples):
                best_x = int(j / x_samples * n)
                fid_minus = data[name][best_x]["fid_minus"]
                fid_plus = data[name][best_x]["fid_plus"]
                charact = data[name][best_x]["charact"]
                merged_sufficient_results[name][j] += 1 - fid_minus
                merged_necessary_results[name][j] += fid_plus
                merged_results[name][j] += charact

    cutoff = int(x_samples)
    for result, result_label in zip(
        [merged_sufficient_results, merged_necessary_results, merged_results],
        ["(1 - Fidelity-)", "Fidelity+", "Characterization"],
    ):
        fig, ax = plt.subplots()
        fig.tight_layout(pad=3.0)
        for sampler_name, sampler_display_name, color in zip(
            sampler_names, sampler_display_names, colors
        ):
            ax.plot(
                [x / x_samples for x in range(cutoff)],
                [x / count for x in result[sampler_name][:cutoff]],
                label=sampler_display_name,
                color=color,
            )
        ax.set_xlabel("Sparsity")
        ax.set_ylabel(result_label)
        ax.set_title(f"{dataset_name}: {result_label} vs Sparsity (n={count})")
        ax.legend()
        plt.show()


def plot_topk_sparsity(
    dataset_name, processed_data, sampler_names, sampler_display_names, colors, k_arr
):
    count = len(processed_data)
    necessary_results = defaultdict(lambda: [[] for _ in k_arr])
    sufficient_results = defaultdict(lambda: [[] for _ in k_arr])
    results = defaultdict(lambda: [[] for _ in k_arr])
    for _, data in processed_data.items():
        n = len(data["Random"])
        for name in sampler_names:
            for i, k in enumerate(k_arr):
                if k + 1 >= n:
                    continue
                fid_minus = data[name][k + 1]["fid_minus"]
                fid_plus = data[name][k + 1]["fid_plus"]
                charact = data[name][k + 1]["charact"]
                sufficient_results[name][i].append(1 - fid_minus)
                necessary_results[name][i].append(fid_plus)
                results[name][i].append(charact)

    for result, result_label in zip(
        [sufficient_results, necessary_results, results],
        ["(1 - Fidelity-)", "Fidelity+", "Characterization"],
    ):
        # Bar Plot with error bars
        fig, ax = plt.subplots()
        fig.tight_layout(pad=3.0)
        labels = [f"{k}" for k in k_arr]
        x = range(len(labels))
        width = 0.1
        for i, (sampler_name, sampler_display_name, color) in enumerate(
            zip(sampler_names, sampler_display_names, colors)
        ):
            ax.bar(
                [a + i * width for a in x],
                [np.mean(result[sampler_name][i]) for i in range(len(k_arr))],
                width=width,
                label=sampler_display_name,
                color=color,
            )
            # add error bar
            ax.errorbar(
                [a + i * width for a in x],
                [np.mean(result[sampler_name][i]) for i in range(len(k_arr))],
                yerr=[
                    np.std(result[sampler_name][i])
                    / np.sqrt(len(result[sampler_name][i]))
                    for i in range(len(k_arr))
                ],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
        ax.set_xticks([a + (len(colors) / 2 - 0.5) * width for a in x])
        ax.set_xticklabels(labels)
        ax.set_xlabel("Top K")
        ax.set_ylabel(result_label)
        ax.set_title(f"{dataset_name}: {result_label} vs Top K (n={count})")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python visualize.py <dataset> <start> <stop> <group>")

    dataset = sys.argv[1]
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    group = "" if len(sys.argv) < 5 else sys.argv[4]

    with open(f"./results/raw_data/data_{dataset}_{start}_{stop}.json", "r") as f:
        all_data = json.load(f)

    sampler_names = [
        "EdgeGNNExplainer",
        "EdgeSubgraphX",
        "GNNExplainer",
        "SubgraphX",
        "Embedding",
        "Random",
    ]

    sampler_display_names = [
        "Edge GNNExplainer",
        "Edge SubgraphX",
        "GNNExplainer",
        "SubgraphX",
        "Embedding Baseline",
        "Random Baseline",
    ]

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    if group == "subgraphx":
        indices = [1, 3, 5]
    elif group == "gnnexplainer":
        indices = [0, 2, 5]
    elif group == "original":
        indices = [2, 3, 4, 5]
    elif group == "final":
        indices = [0, 1, 4, 5]
    else:
        indices = range(len(sampler_names))

    sampler_names = [sampler_names[i] for i in indices]
    sampler_display_names = [sampler_display_names[i] for i in indices]
    colors = [colors[i] for i in indices]

    dataset_name = {
        "facebook": "Facebook",
        "imdb": "IMDB",
        "lastfm": "LastFM",
    }[dataset]
    processed_data = process_data(all_data, sampler_names)
    plot_continuous_sparsity(
        dataset_name,
        processed_data,
        sampler_names,
        sampler_display_names,
        colors,
    )
    plot_topk_sparsity(
        dataset_name,
        processed_data,
        sampler_names,
        sampler_display_names,
        colors,
        [1, 5, 10],
    )

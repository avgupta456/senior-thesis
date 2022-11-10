import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.metrics.fidelity import charact_prob, fid_minus_prob, fid_plus_prob
from src.utils import sigmoid


def filter_data(all_data, filter=None):
    filtered_data = {}
    for i, data in all_data.items():
        # Only handles positive edge explanations currently, > 285 --> negative
        if int(i) > 285:
            continue

        # Ensure empty prediction is close to 0 (negative), full prediction is close to 1 (positive)
        _, initial_pred, expl_pred, _ = data["Random"][0]
        if not (sigmoid(expl_pred) < 1 / 3 and sigmoid(initial_pred) > 2 / 3):
            continue

        # And there are at least 10 nodes in the joint neighborhood
        n = len(data["Random"])
        if n < 10:
            continue

        # Additional filters
        if filter == "small" and n > 50:
            continue
        if filter == "medium" and (n < 50 or n > 200):
            continue
        if filter == "large" and n < 200:
            continue

        filtered_data[i] = data
    return filtered_data


def process_data(filtered_data, sampler_names):
    processed_data = {}
    for i, data in filtered_data.items():
        processed_data[i] = {}
        n = len(data["Random"])
        for name in sampler_names:
            processed_data[i][name] = []
            for j in range(n):
                _, initial_pred, expl_pred, remove_pred = data[name][j]
                sig_initial_pred = sigmoid(initial_pred)
                sig_expl_pred = sigmoid(expl_pred)
                sig_remove_pred = sigmoid(remove_pred)
                fid_minus = fid_minus_prob(sig_initial_pred, sig_expl_pred)
                fid_plus = fid_plus_prob(sig_initial_pred, sig_remove_pred)
                charact = charact_prob(
                    sig_initial_pred, sig_expl_pred, sig_remove_pred, 0.2, 0.8
                )
                fid_minus = min(max(fid_minus.item(), 0), 1)
                fid_plus = min(max(fid_plus.item(), 0), 1)
                charact = min(max(charact.item(), 0), 1)
                processed_data[i][name].append(
                    {"fid_minus": fid_minus, "fid_plus": fid_plus, "charact": charact}
                )
    return processed_data


def plot_continuous_sparsity(processed_data, sampler_names):
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


def plot_topk_sparsity(processed_data, sampler_names, k_arr):
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
        labels = [f"{k}" for k in k_arr]
        x = range(len(labels))
        width = 0.1
        for i, sampler_name in enumerate(sampler_names):
            ax.bar(
                [a + i * width for a in x],
                [np.mean(result[sampler_name][i]) for i in range(len(k_arr))],
                width=width,
                label=sampler_name,
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
        ax.set_xticks([a + (len(x) / 2 + 1) * width for a in x])
        ax.set_xticklabels(labels)
        ax.set_xlabel("Top K")
        ax.set_ylabel(result_label)
        ax.set_title(f"{result_label} vs Top K (n={count})")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    with open("./results/vary_sparsity/data_600.json", "r") as f:
        all_data = json.load(f)

    filtered_data = filter_data(all_data, filter="large")

    sampler_names = [
        "GNNExplainer",
        "SubgraphX",
        "EdgeSubgraphX",
        "Embedding",
        "Degree",
        "Random",
    ]

    processed_data = process_data(filtered_data, sampler_names)

    plot_continuous_sparsity(processed_data, sampler_names)
    plot_topk_sparsity(processed_data, sampler_names, [5, 10, 20])

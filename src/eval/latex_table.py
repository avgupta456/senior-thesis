import json
import sys

import numpy as np

from src.eval.utils import process_data


def create_latex_table(datasets, dataset_names, sampler_names, k_arr):
    dataset_mean = {}
    dataset_std = {}
    for dataset, dataset_name in zip(datasets, dataset_names):
        count = len(dataset)
        charact = {name: [[] for _ in k_arr] for name in sampler_names}
        for _, data in dataset.items():
            n = len(data["Random"])
            for name in sampler_names:
                for i, k in enumerate(k_arr):
                    if k + 1 >= n:
                        continue
                    charact[name][i].append(data[name][k + 1]["charact"])

        mean = {
            name: [sum(charact[name][i]) / count for i in range(len(k_arr))]
            for name in sampler_names
        }
        std = {
            name: [np.std(charact[name][i]) / np.sqrt(count) for i in range(len(k_arr))]
            for name in sampler_names
        }

        dataset_mean[dataset_name] = mean
        dataset_std[dataset_name] = std

    print(
        """
\\begin{table}[H]
    \\centering
        \\begin{tabular}{|c|ccc|ccc|ccc|}
        \\hline
        & \\multicolumn{3}{|c|}{Facebook (n="""
        + str(len(datasets[0]))
        + """)} & \\multicolumn{3}{|c|}{IMDB (n="""
        + str(len(datasets[1]))
        + """)} & \\multicolumn{3}{|c|}{LastFM (n="""
        + str(len(datasets[2]))
        + """)} \\\\
        &  K=1 & 5 & 10 & 1 & 5 & 10 & 1 & 5 & 10 \\\\ \\hline
    """
    )
    for sampler_name in sampler_names:
        display_name = sampler_name
        if sampler_name == "EdgeGNNExplainer":
            display_name = "Edge GNNExplainer"
        elif sampler_name == "EdgeSubgraphX":
            display_name = "Edge SubgraphX"
        print(f"\t{display_name} & ", end="")
        for i, dataset_name in enumerate(dataset_names):
            for j, k in enumerate(k_arr):
                is_best_sampler = dataset_mean[dataset_name][sampler_name][j] == max(
                    dataset_mean[dataset_name][name][j] for name in sampler_names
                )
                if is_best_sampler:
                    print("\\textbf{", end="")
                print(
                    f"{dataset_mean[dataset_name][sampler_name][j]:.2f} ",
                    end="",
                )
                if is_best_sampler:
                    print("}", end="")
                if i != len(dataset_names) - 1 or j != len(k_arr) - 1:
                    print("& ", end="")
        print("\\\\ \\hline")
    print(
        """
        \\hline
    \\end{tabular}
    \\caption{Mean characterization score for Top-K explanations. Best methods bolded.}
    \\label{tab:charact}
\\end{table}
"""
    )


if __name__ == "__main__":
    group = "" if len(sys.argv) < 2 else sys.argv[1]

    with open("./results/data/data_facebook_0_300.json", "r") as f:
        facebook_data = json.load(f)

    with open("./results/data/data_imdb_0_10000.json", "r") as f:
        imdb_data = json.load(f)

    with open("./results/data/data_lastfm_0_1000.json", "r") as f:
        lastfm_data = json.load(f)

    sampler_names = [
        "EdgeGNNExplainer",
        "EdgeSubgraphX",
        "GNNExplainer",
        "SubgraphX",
        "Embedding",
        "Random",
    ]

    if group == "subgraphx":
        indices = [5, 3, 1]
    elif group == "gnnexplainer":
        indices = [5, 2, 0]
    elif group == "original":
        indices = [5, 4, 2, 3]
    elif group == "final":
        indices = [5, 4, 1, 0]
    else:
        indices = range(len(sampler_names))

    sampler_names = [sampler_names[i] for i in indices]

    print("Loading Facebook")
    facebook_data = process_data(facebook_data, sampler_names)
    print("Loading IMDB")
    imdb_data = process_data(imdb_data, sampler_names)
    print("Loading LastFM")
    lastfm_data = process_data(lastfm_data, sampler_names)

    create_latex_table(
        [facebook_data, imdb_data, lastfm_data],
        ["Facebook", "IMDB", "LastFM"],
        sampler_names,
        [1, 5, 10],
    )

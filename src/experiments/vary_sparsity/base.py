from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import dataset, device, test_data
from src.explainers.main import (
    sample_degree,
    sample_edge_subgraphx,
    sample_embedding,
    sample_gnnexplainer,
    sample_random,
    sample_subgraphx,
)
from src.metrics.fidelity import charact_prob, fid_minus_prob, fid_plus_prob
from src.pred import Net
from src.utils import get_neighbors, sigmoid


def run_experiment(
    model, x, edge_index, edge_label_index, samplers, sampler_names, show_plots=False
):
    all_results = {}
    for i in range(edge_label_index.shape[1]):
        node_idx_1 = edge_label_index[0, i].item()
        node_idx_2 = edge_label_index[1, i].item()
        n_neighbors = get_neighbors(edge_index, node_idx_1, node_idx_2).shape[0]
        print(i, node_idx_1, node_idx_2, n_neighbors)

        curr_edge_label_index = torch.tensor([[node_idx_1], [node_idx_2]]).to(device)
        initial_pred = model(x, edge_index, curr_edge_label_index)[1]

        sampler_outputs = []
        for sampler, sampler_name in zip(samplers, sampler_names):
            output = sampler(model, x, edge_index, node_idx_1, node_idx_2)
            output = sorted(output.items(), key=lambda x: -x[1])
            sampler_outputs.append(output)

        sub_edge_mask = (edge_index[1] == node_idx_1) | (edge_index[1] == node_idx_2)

        sufficient_results = defaultdict(list)
        necessary_results = defaultdict(list)
        results = defaultdict(list)
        for k in range(n_neighbors + 1):
            for output, sampler_name in zip(sampler_outputs, sampler_names):
                keep_edges = torch.isin(edge_index[0], torch.tensor(output[:k]))

                S_filter = np.ones(edge_index.shape[1], dtype=bool)
                S_filter[sub_edge_mask] = 0
                S_filter[sub_edge_mask & keep_edges] = 1

                expl_pred = model(x, edge_index[:, S_filter], curr_edge_label_index)[1]
                fid_minus = fid_minus_prob(sigmoid(initial_pred), sigmoid(expl_pred))
                sufficient_results[sampler_name].append(1 - fid_minus.item())

                S_filter = np.ones(edge_index.shape[1], dtype=bool)
                S_filter[sub_edge_mask] = 1
                S_filter[sub_edge_mask & keep_edges] = 0

                remove_pred = model(x, edge_index[:, S_filter], curr_edge_label_index)[
                    1
                ]
                fid_plus = fid_plus_prob(sigmoid(initial_pred), sigmoid(remove_pred))
                necessary_results[sampler_name].append(fid_plus.item())

                charact = charact_prob(
                    sigmoid(initial_pred), sigmoid(expl_pred), sigmoid(remove_pred)
                )
                results[sampler_name].append(charact.item())

        if show_plots:
            for result, result_label in zip(
                [sufficient_results, necessary_results, results],
                ["Sufficient", "Necessary", "Characterization"],
            ):
                fig, ax = plt.subplots()
                for sampler_name in sampler_names:
                    ax.plot(result[sampler_name], label=sampler_name)
                ax.set_xlabel("Number of nodes")
                ax.set_ylabel("Prediction")
                ax.set_title(f"{result_label} vs Sparsity")
                ax.legend()
                plt.show()

        all_results[i] = {
            "sufficient": sufficient_results,
            "necessary": necessary_results,
            "characterization": results,
        }

    return all_results


if __name__ == "__main__":
    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))
    run_experiment(
        model,
        test_data.x,
        test_data.edge_index,
        test_data.edge_label_index[:, 12:13],
        [
            sample_gnnexplainer,
            sample_subgraphx,
            sample_edge_subgraphx,
            sample_embedding,
            sample_degree,
            sample_random,
        ],
        ["GNNExplainer", "SubgraphX", "EdgeSubgraphX", "Embedding", "Degree", "Random"],
        show_plots=True,
    )

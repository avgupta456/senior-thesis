from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import dataset, device, test_data
from src.explainers.random import RandomExplainer
from src.explainers.subgraphx import SubgraphX
from src.explainers.edge_subgraphx import EdgeSubgraphX
from src.explainers.gnnexplainer import GNNExplainer
from src.pred import Net
from src.utils import get_neighbors, sigmoid
from src.metrics.fidelity import fid_plus_prob, fid_minus_prob, charact_prob


def sample_gnnexplainer(
    model, x, edge_index, node_idx_1, node_idx_2, epochs=100, lr=0.01
):
    # GNNExplainer, 100 queries per explanation
    gnnexplainer = GNNExplainer(model, x, edge_index, epochs=epochs, lr=lr)
    return gnnexplainer.explain_edge(node_idx_1, node_idx_2)


def sample_subgraphx(model, x, edge_index, node_idx_1, node_idx_2, T=5):
    # SubgraphX, 5 * size(neighborhood) queries per explanation
    subgraphx = SubgraphX(model, x, edge_index, T=T)
    return subgraphx.explain_edge(node_idx_1, node_idx_2)


def sample_edge_subgraphx(model, x, edge_index, node_idx_1, node_idx_2, T=5):
    # EdgeSubgraphX, 5 * size(neighborhood) queries per explanation
    edge_subgraphx = EdgeSubgraphX(model, x, edge_index, T=T)
    return edge_subgraphx.explain_edge(node_idx_1, node_idx_2)


def sample_random(model, x, edge_index, node_idx_1, node_idx_2):
    random_explainer = RandomExplainer(model, x, edge_index)
    return random_explainer.explain_edge(node_idx_1, node_idx_2)


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
        test_data.edge_label_index[:, 10:11],
        [sample_gnnexplainer, sample_subgraphx, sample_edge_subgraphx, sample_random],
        ["GNNExplainer", "SubgraphX", "EdgeSubgraphX", "Random"],
    )

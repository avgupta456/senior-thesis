from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import dataset, device, test_data
from src.explainers.edge_subgraphx import EdgeSubgraphX
from src.explainers.gnnexplainer import GNNExplainer
from src.explainers.random import RandomExplainer
from src.explainers.subgraphx import SubgraphX
from src.metrics.fidelity import charact_prob, fid_minus_prob, fid_plus_prob
from src.pred import Net
from src.utils import get_neighbors, mask_nodes, sigmoid


def sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2):
    # GNNExplainer, 200 queries per explanation
    gnnexplainer = GNNExplainer(model, x, edge_index, epochs=200, lr=0.1)
    return gnnexplainer.explain_edge(node_idx_1, node_idx_2)


def sample_subgraphx(model, x, edge_index, node_idx_1, node_idx_2):
    # SubgraphX, T * size(neighborhood) queries per explanation
    subgraphx = SubgraphX(model, x, edge_index, T=10)
    return subgraphx.explain_edge(node_idx_1, node_idx_2)


def sample_edge_subgraphx(model, x, edge_index, node_idx_1, node_idx_2):
    # EdgeSubgraphX, T * size(neighborhood) queries per explanation
    edge_subgraphx = EdgeSubgraphX(model, x, edge_index, T=10)
    return edge_subgraphx.explain_edge(node_idx_1, node_idx_2)


def sample_random(model, x, edge_index, node_idx_1, node_idx_2):
    random_explainer = RandomExplainer(model, x, edge_index)
    return random_explainer.explain_edge(node_idx_1, node_idx_2)


if __name__ == "__main__":
    # Load the dataset and model
    x, edge_index = test_data.x, test_data.edge_index
    index = 10
    node_idx_1 = test_data.edge_label_index[0, index].item()  # 24
    node_idx_2 = test_data.edge_label_index[1, index].item()  # 187

    print("Node 1:", node_idx_1)
    print("Node 2:", node_idx_2)
    print("N Neighbors:", get_neighbors(edge_index, node_idx_1, node_idx_2).shape[0])

    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))

    neighbors = get_neighbors(edge_index, node_idx_1, node_idx_2)
    edge_label_index = torch.tensor([[node_idx_1], [node_idx_2]])

    initial_pred = model(x, edge_index, edge_label_index)[1]

    gnnexplainer_output = sample_gnnexplainer(
        model, x, edge_index, node_idx_1, node_idx_2
    )
    gnnexplainer_output = sorted(gnnexplainer_output.items(), key=lambda x: -x[1])

    subgraphx_output = sample_subgraphx(model, x, edge_index, node_idx_1, node_idx_2)
    subgraphx_output = sorted(subgraphx_output.items(), key=lambda x: -x[1])

    edge_subgraphx_output = sample_edge_subgraphx(
        model, x, edge_index, node_idx_1, node_idx_2
    )
    edge_subgraphx_output = sorted(edge_subgraphx_output.items(), key=lambda x: -x[1])

    random_output = sample_random(model, x, edge_index, node_idx_1, node_idx_2)
    random_output = sorted(random_output.items(), key=lambda x: -x[1])

    sufficient_results = defaultdict(list)
    necessary_results = defaultdict(list)
    results = defaultdict(list)
    for k in range(neighbors.shape[0] + 1):
        for label, output in zip(
            ["gnnexplainer", "subgraphx", "edge_subgraphx", "random"],
            [
                gnnexplainer_output,
                subgraphx_output,
                edge_subgraphx_output,
                random_output,
            ],
        ):
            keep_nodes = [x for x, _ in output[:k]]

            S_filter = np.ones(x.shape[0], dtype=bool)
            S_filter[neighbors] = 0
            S_filter[keep_nodes] = 1

            # Include this line to remove the features of the edge nodes
            # S_filter[[node_idx_1, node_idx_2]] = 0

            temp_x = mask_nodes(x, S_filter)
            expl_pred = model(temp_x, edge_index, edge_label_index)[1]
            fid_minus = fid_minus_prob(sigmoid(initial_pred), sigmoid(expl_pred))
            sufficient_results[label].append(1 - fid_minus.item())

            S_filter = np.ones(x.shape[0], dtype=bool)
            S_filter[neighbors] = 1
            S_filter[keep_nodes] = 0

            temp_x = mask_nodes(x, S_filter)
            remove_pred = model(temp_x, edge_index, edge_label_index)[1]
            fid_plus = fid_plus_prob(sigmoid(initial_pred), sigmoid(remove_pred))
            necessary_results[label].append(fid_plus.item())

            charact = charact_prob(
                sigmoid(initial_pred), sigmoid(expl_pred), sigmoid(remove_pred)
            )
            results[label].append(charact.item())

    for result, result_label in zip(
        [sufficient_results, necessary_results, results],
        ["Sufficient", "Necessary", "Characterization"],
    ):
        fig, ax = plt.subplots()
        for key, label in zip(
            ["gnnexplainer", "subgraphx", "edge_subgraphx", "random"],
            ["GNNExplainer", "SubgraphX", "EdgeSubgraphX", "Random"],
        ):
            ax.plot(result[key], label=label)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{result_label} vs Sparsity")
        ax.legend()
        plt.show()
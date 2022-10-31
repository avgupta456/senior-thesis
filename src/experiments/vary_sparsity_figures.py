from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import dataset, device, test_data
from src.explainers.random import RandomExplainer
from src.explainers.subgraphx import SubgraphX
from src.explainers.gnnexplainer import GNNExplainer
from src.pred import Net
from src.utils import get_neighbors, mask_nodes


def sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2):
    # GNNExplainer, 200 queries per explanation
    gnnexplainer = GNNExplainer(model, x, edge_index, epochs=200, lr=0.1)
    return gnnexplainer.explain_edge(node_idx_1, node_idx_2)


def sample_subgraphx(model, x, edge_index, node_idx_1, node_idx_2):
    # SubgraphX, T * size(neighborhood) queries per explanation
    subgraphx = SubgraphX(model, x, edge_index, T=10)
    return subgraphx.explain_edge(node_idx_1, node_idx_2)


def sample_random(model, x, edge_index, node_idx_1, node_idx_2):
    random_explainer = RandomExplainer(model, x, edge_index)
    return random_explainer.explain_edge(node_idx_1, node_idx_2)


def sparsity_experiment(model, x, edge_index, node_idx_1, node_idx_2):
    gnnexplainer_output = sample_gnnexplainer(
        model, x, edge_index, node_idx_1, node_idx_2
    )
    gnnexplainer_output = sorted(gnnexplainer_output.items(), key=lambda x: -x[1])

    subgraphx_output = sample_subgraphx(model, x, edge_index, node_idx_1, node_idx_2)
    subgraphx_output = sorted(subgraphx_output.items(), key=lambda x: -x[1])

    random_output = sample_random(model, x, edge_index, node_idx_1, node_idx_2)
    random_output = sorted(random_output.items(), key=lambda x: -x[1])

    neighbors = get_neighbors(edge_index, node_idx_1, node_idx_2)
    edge_label_index = torch.tensor([[node_idx_1], [node_idx_2]])

    sufficient_results = defaultdict(list)
    necessary_results = defaultdict(list)
    for k in range(neighbors.shape[0] + 1):
        for label, output in zip(
            ["gnnexplainer", "subgraphx", "random"],
            [gnnexplainer_output, subgraphx_output, random_output],
        ):
            keep_nodes = [x for x, _ in output[:k]]

            S_filter = np.ones(x.shape[0], dtype=bool)
            S_filter[neighbors] = 0
            S_filter[keep_nodes] = 1
            # S_filter[[node_idx_1, node_idx_2]] = 0

            temp_x = mask_nodes(x, S_filter)
            pred = model(temp_x, edge_index, edge_label_index)[1]
            sufficient_results[label].append(pred.item())

            S_filter = np.ones(x.shape[0], dtype=bool)
            S_filter[neighbors] = 1
            S_filter[keep_nodes] = 0

            temp_x = mask_nodes(x, S_filter)
            pred = model(temp_x, edge_index, edge_label_index)[1]
            necessary_results[label].append(pred.item())

    # Sufficient Predictions
    fig, ax = plt.subplots()

    ax.plot(sufficient_results["gnnexplainer"], label="GNNExplainer")
    ax.plot(sufficient_results["subgraphx"], label="SubgraphX")
    ax.plot(sufficient_results["random"], label="Random")

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Prediction")
    ax.set_title("Sufficient Predictions: Expl Pred vs. Num Nodes")

    ax.legend()

    fig.savefig("./figures/sufficient/{}_{}.png".format(node_idx_1, node_idx_2))
    plt.close()

    # Necessary Predictions
    fig, ax = plt.subplots()

    ax.plot(necessary_results["gnnexplainer"], label="GNNExplainer")
    ax.plot(necessary_results["subgraphx"], label="SubgraphX")
    ax.plot(necessary_results["random"], label="Random")

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Prediction")
    ax.set_title("Necessary Predictions: Removed Pred vs. Num Nodes")

    ax.legend()

    fig.savefig("./figures/necessary/{}_{}.png".format(node_idx_1, node_idx_2))
    plt.close()


if __name__ == "__main__":
    # Load the dataset and model
    x, edge_index = test_data.x, test_data.edge_index

    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))

    for i in range(test_data.edge_label_index.shape[1]):
        node_idx_1 = test_data.edge_label_index[0][i].item()
        node_idx_2 = test_data.edge_label_index[1][i].item()
        n_neighbors = get_neighbors(edge_index, node_idx_1, node_idx_2).shape[0]
        print(i, node_idx_1, node_idx_2, n_neighbors)

        sparsity_experiment(model, x, edge_index, node_idx_1, node_idx_2)

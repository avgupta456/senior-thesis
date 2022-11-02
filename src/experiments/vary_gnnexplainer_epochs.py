from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import dataset, device, test_data
from src.explainers.gnnexplainer import GNNExplainer
from src.pred import Net
from src.utils import get_neighbors, sigmoid
from src.metrics.fidelity import fid_plus_prob, fid_minus_prob, charact_prob


def sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2, epochs):
    # GNNExplainer, 200 queries per explanation
    gnnexplainer = GNNExplainer(model, x, edge_index, epochs=epochs, lr=0.01)
    return gnnexplainer.explain_edge(node_idx_1, node_idx_2)


if __name__ == "__main__":
    # Load the dataset and model
    x, edge_index = test_data.x, test_data.edge_index
    index = 12
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

    output_20 = sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2, 20)
    output_20 = sorted(output_20.items(), key=lambda x: -x[1])

    output_50 = sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2, 50)
    output_50 = sorted(output_50.items(), key=lambda x: -x[1])

    output_100 = sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2, 100)
    output_100 = sorted(output_100.items(), key=lambda x: -x[1])

    output_200 = sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2, 200)
    output_200 = sorted(output_200.items(), key=lambda x: -x[1])

    sufficient_results = defaultdict(list)
    necessary_results = defaultdict(list)
    results = defaultdict(list)

    edge_mask_1 = edge_index[1] == edge_label_index[0]
    edge_mask_2 = edge_index[1] == edge_label_index[1]
    sub_edge_mask = edge_mask_1 | edge_mask_2

    for k in range(neighbors.shape[0] + 1):
        for label, output in zip(
            ["20", "50", "100", "200"], [output_20, output_50, output_100, output_200]
        ):
            keep_edges = torch.isin(edge_index[0], torch.tensor(output[:k]))

            S_filter = np.ones(edge_index.shape[1], dtype=bool)
            S_filter[sub_edge_mask] = 0
            S_filter[sub_edge_mask & keep_edges] = 1

            expl_pred = model(x, edge_index[:, S_filter], edge_label_index)[1]
            fid_minus = fid_minus_prob(sigmoid(initial_pred), sigmoid(expl_pred))
            sufficient_results[label].append(1 - fid_minus.item())

            S_filter = np.ones(edge_index.shape[1], dtype=bool)
            S_filter[sub_edge_mask] = 1
            S_filter[sub_edge_mask & keep_edges] = 0

            remove_pred = model(x, edge_index[:, S_filter], edge_label_index)[1]
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
        for key, label in zip(["20", "50", "100", "200"], ["20", "50", "100", "200"]):
            ax.plot(result[key], label=label)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{result_label} vs Sparsity")
        ax.legend()
        plt.show()

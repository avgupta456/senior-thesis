from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import dataset, device, test_data
from src.explainers.main import (  # noqa: F401
    sample_degree,
    sample_edge_subgraphx,
    sample_embedding,
    sample_gnnexplainer,
    sample_random,
    sample_subgraphx,
)
from src.pred import Net
from src.utils import get_neighbors, sigmoid


def run_experiment(
    model,
    x,
    edge_index,
    edge_label_index,
    edge_label,
    samplers,
    sampler_names,
    show_plots=False,
):
    all_results = {}
    for i in range(edge_label_index.shape[1]):
        # TODO: Can use k_hop_subgraph here, not urgent
        node_idx_1 = edge_label_index[0, i].item()
        node_idx_2 = edge_label_index[1, i].item()
        target = int(edge_label[i].item())  # 0 for negative, 1 for positive
        n_neighbors = get_neighbors(edge_index, node_idx_1, node_idx_2).shape[0]
        print(i, node_idx_1, node_idx_2, target, n_neighbors)

        _label_index = torch.tensor([[node_idx_1], [node_idx_2]]).to(device)
        initial_pred = model(x, edge_index, _label_index)[1]

        sampler_outputs = []
        for sampler, sampler_name in zip(samplers, sampler_names):
            output = sampler(model, x, edge_index, node_idx_1, node_idx_2, target)
            output = sorted(output.items(), key=lambda x: -x[1])
            sampler_outputs.append(output)

        sub_edge_mask = (edge_index[1] == node_idx_1) | (edge_index[1] == node_idx_2)

        results = defaultdict(list)
        for k in range(n_neighbors + 1):
            for output, sampler_name in zip(sampler_outputs, sampler_names):
                keep_edges = torch.isin(edge_index[0], torch.tensor(output[:k]))

                S_filter = np.ones(edge_index.shape[1], dtype=bool)
                S_filter[sub_edge_mask] = 0
                S_filter[sub_edge_mask & keep_edges] = 1

                expl_pred = model(x, edge_index[:, S_filter], _label_index)[1]

                S_filter = np.ones(edge_index.shape[1], dtype=bool)
                S_filter[sub_edge_mask] = 1
                S_filter[sub_edge_mask & keep_edges] = 0

                remove_pred = model(x, edge_index[:, S_filter], _label_index)[1]

                new_node = None if k == 0 else int(output[k - 1][0])
                results[sampler_name].append(
                    [
                        new_node,
                        float(initial_pred.item()),
                        float(expl_pred.item()),
                        float(remove_pred.item()),
                    ]
                )

        all_results[i] = results

        if show_plots:
            # Logits
            fig, ax = plt.subplots()
            for sampler_name in sampler_names:
                temp_data = [x[2] for x in results[sampler_name]]
                ax.plot(temp_data, label=sampler_name)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel("Prediction")
            ax.set_title("Explanation Prediction vs Sparsity")
            ax.legend()
            plt.show()

            # Probabilities
            fig, ax = plt.subplots()
            for sampler_name in sampler_names:
                temp_data = [sigmoid(x[2]) for x in results[sampler_name]]
                ax.plot(temp_data, label=sampler_name)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel("Prediction")
            ax.set_title("Explanation Prediction vs Sparsity")
            ax.legend()
            plt.show()

    return all_results


if __name__ == "__main__":
    index = 2

    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))
    run_experiment(
        model,
        test_data.x,
        test_data.edge_index,
        test_data.edge_label_index[:, index : index + 1],
        test_data.edge_label[index : index + 1],
        [
            sample_gnnexplainer,
            sample_subgraphx,
            sample_edge_subgraphx,
            sample_embedding,
            sample_degree,
            sample_random,
        ],
        [
            "GNNExplainer",
            "SubgraphX",
            "EdgeSubgraphX",
            "Embedding",
            "Degree",
            "Random",
        ],
        show_plots=True,
    )

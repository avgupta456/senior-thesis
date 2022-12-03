from collections import defaultdict
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.facebook import get_facebook_dataset
from src.datasets.imdb import get_imdb_dataset
from src.datasets.dblp import get_dblp_dataset
from src.explainers.main import (  # noqa: F401
    sample_degree,
    sample_edge_subgraphx,
    sample_embedding,
    sample_gnnexplainer,
    sample_random,
    sample_subgraphx,
)
from src.pred.model import SimpleNet, Net
from src.utils.utils import device, sigmoid
from src.utils.subgraph import edge_centered_subgraph
from src.utils.neighbors import get_neighbors


def get_dataset_and_model(name):
    if name == "facebook":
        train_data, val_data, test_data = get_facebook_dataset()
        model = Net(128, 32, metadata=train_data.metadata()).to(device)
        model.load_state_dict(torch.load("./models/facebook_model.pt"))
        key = ("person", "to", "person")
    elif name == "imdb":
        train_data, val_data, test_data = get_imdb_dataset()
        model = SimpleNet(128, 32, metadata=train_data.metadata()).to(device)
        model.load_state_dict(torch.load("./models/imdb_model.pt"))
        key = ("movie", "to", "actor")
    elif name == "dblp":
        train_data, val_data, test_data = get_dblp_dataset()
        model = SimpleNet(128, 32, metadata=train_data.metadata()).to(device)
        model.load_state_dict(torch.load("./models/dblp_model.pt"))
        key = ("paper", "to", "author")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_data, val_data, test_data, model, key


def run_experiment(
    model, data, key, start_index, end_index, samplers, sampler_names, show_plots=False
):
    start, _, end = key

    edge_label_index = data.edge_label_index_dict[key][:, start_index:end_index]
    edge_label = data.edge_label_dict[key][start_index:end_index]

    all_results = {}
    for i in range(edge_label_index.shape[1]):
        node_idx_1 = edge_label_index[0, i]
        node_idx_2 = edge_label_index[1, i]

        # Ignore negative edges
        target = int(edge_label[i].item())
        if target == 0:
            continue

        _data, _node_idx_1, _node_idx_2 = edge_centered_subgraph(
            node_idx_1, start, node_idx_2, end, data, 0
        )

        _label_index = torch.tensor([[_node_idx_1], [_node_idx_2]]).to(device)
        initial_pred = model(_data.x_dict, _data.edge_index_dict, _label_index, key)
        initial_pred = sigmoid(initial_pred[1]).item()
        if initial_pred > 0.25:
            continue

        curr_data, node_idx_1, node_idx_2 = edge_centered_subgraph(
            node_idx_1, start, node_idx_2, end, data, 2
        )

        neighbors = get_neighbors(curr_data, node_idx_1, start, node_idx_2, end)
        n_neighbors = sum(len(n) for n in neighbors.values())
        if n_neighbors <= 5:
            continue

        _label_index = torch.tensor([[node_idx_1], [node_idx_2]]).to(device)
        final_pred = model(
            curr_data.x_dict, curr_data.edge_index_dict, _label_index, key
        )
        final_pred = sigmoid(final_pred[1]).item()
        if final_pred < 0.75:
            continue

        print(
            i,
            "\t",
            start,
            node_idx_1,
            "\t",
            end,
            node_idx_2,
            "\t",
            "n_neighbors",
            n_neighbors,
            " \t",
            "initial_pred",
            round(initial_pred, 4),
            "\t",
            "final_pred",
            round(final_pred, 4),
        )

        continue

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
    if len(sys.argv) != 2:
        print("Usage: python main.py <dataset>")

    dataset_name = sys.argv[1]
    train_data, val_data, test_data, model, key = get_dataset_and_model(dataset_name)

    print(f"Dataset: {dataset_name}")
    print()

    run_experiment(
        model,
        test_data,
        key,
        0,
        test_data.edge_label_index_dict[key].shape[1],
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

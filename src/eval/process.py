import json
import sys
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from src.eval.utils import get_dataset_and_model
from src.explainers.main import (  # noqa: F401
    sample_embedding,
    sample_gnnexplainer as _sample_gnnexplainer,
    sample_random,
    sample_subgraphx,
)
from src.utils.neighbors import get_neighbors
from src.utils.subgraph import edge_centered_subgraph, remove_edge_connections
from src.utils.utils import device, sigmoid


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
        skip = False

        # Ignore negative edges
        target = int(edge_label[i].item())
        if target == 0:
            continue

        _data, _node_idx_1, _node_idx_2 = edge_centered_subgraph(
            node_idx_1, start, node_idx_2, end, data, 0
        )

        _label_index = torch.tensor([[_node_idx_1], [_node_idx_2]]).to(device)
        initial_pred = model(_data.x_dict, _data.edge_index_dict, _label_index, key)[1]
        if sigmoid(initial_pred).item() > 1 / 3:
            skip = True

        curr_data, node_idx_1, node_idx_2 = edge_centered_subgraph(
            node_idx_1, start, node_idx_2, end, data, 2
        )

        neighbors = get_neighbors(curr_data, node_idx_1, start, node_idx_2, end)
        n_neighbors = sum(len(n) for n in neighbors.values())
        if n_neighbors <= 5 or n_neighbors > 200:
            skip = True

        _label_index = torch.tensor([[node_idx_1], [node_idx_2]]).to(device)
        final_pred = model(
            curr_data.x_dict, curr_data.edge_index_dict, _label_index, key
        )[1]
        if sigmoid(final_pred).item() < 2 / 3:
            skip = True

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
            round(initial_pred.sigmoid().item(), 4),
            "\t",
            "final_pred",
            round(final_pred.sigmoid().item(), 4),
            "\t",
            "SKIPPED" if skip else "",
        )

        if skip:
            continue

        sampler_outputs = []
        for sampler, sampler_name in zip(samplers, sampler_names):
            time_start = datetime.now()
            output = sampler(model, curr_data, node_idx_1, start, node_idx_2, end)
            output = sorted(output.items(), key=lambda x: -x[1])
            sampler_outputs.append(output)
            print(sampler_name, "\t\t", datetime.now() - time_start)

        results = defaultdict(list)
        for k in range(n_neighbors + 1):
            for output, sampler_name in zip(sampler_outputs, sampler_names):
                # Remove edges [k:], leaving [:k]
                expl_data = remove_edge_connections(
                    curr_data,
                    node_idx_1,
                    start,
                    node_idx_2,
                    end,
                    [x[0] for x in output[k:]],
                )

                expl_pred = model(
                    expl_data.x_dict, expl_data.edge_index_dict, _label_index, key
                )[1]

                # Remove edges [:k], leaving [k:]
                remove_data = remove_edge_connections(
                    curr_data,
                    node_idx_1,
                    start,
                    node_idx_2,
                    end,
                    [x[0] for x in output[:k]],
                )

                remove_pred = model(
                    remove_data.x_dict, remove_data.edge_index_dict, _label_index, key
                )[1]

                new_node = None
                if k > 0:
                    new_node = output[k - 1][0][0], int(output[k - 1][0][1])

                results[sampler_name].append(
                    [
                        new_node,
                        float(initial_pred.item()),
                        float(expl_pred.item()),
                        float(remove_pred.item()),
                        float(final_pred.item()),
                    ]
                )

        all_results[i] = results

        if show_plots:
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

            fig, ax = plt.subplots()
            for sampler_name in sampler_names:
                temp_data = [sigmoid(x[3]) for x in results[sampler_name]]
                ax.plot(temp_data, label=sampler_name)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel("Prediction")
            ax.set_title("Removal Prediction vs Sparsity")
            ax.legend()
            plt.show()

    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python process.py <dataset> <start> <stop> <show_plots>")

    # Load Dataset

    dataset_name = sys.argv[1]
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    show_plots = bool(sys.argv[4]) if len(sys.argv) > 4 else False

    (
        train_data,
        val_data,
        test_data,
        model,
        key,
        gnnexplainer_config,
    ) = get_dataset_and_model(dataset_name)

    print(f"Dataset: {dataset_name}")
    print()

    # Run Experiment

    def sample_gnnexplainer(model, data, node_idx_1, start, node_idx_2, end):
        return _sample_gnnexplainer(
            model, data, node_idx_1, start, node_idx_2, end, **gnnexplainer_config
        )

    actual_stop = min(stop, test_data.edge_label_index_dict[key].shape[1])
    all_results = run_experiment(
        model,
        test_data,
        key,
        start,
        actual_stop,
        [
            sample_gnnexplainer,
            sample_subgraphx,
            sample_embedding,
            sample_random,
        ],
        [
            "GNNExplainer",
            "SubgraphX",
            "Embedding",
            "Random",
        ],
        show_plots=show_plots,
    )

    with open(f"./results/data/data_{dataset_name}_{start}_{stop}.json", "w") as f:
        json.dump(all_results, f)

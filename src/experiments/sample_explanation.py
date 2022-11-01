import torch

from src.dataset import dataset, device, test_data
from src.explainers.gnnexplainer import GNNExplainer
from src.explainers.random import RandomExplainer
from src.explainers.subgraphx import SubgraphX
from src.explainers.edge_subgraphx import EdgeSubgraphX
from src.pred import Net
from src.utils import get_neighbors


def sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2):
    # GNNExplainer, 200 queries per explanation
    gnnexplainer = GNNExplainer(model, x, edge_index, epochs=200, lr=0.1)
    output = gnnexplainer.explain_edge(node_idx_1, node_idx_2)

    print("GNNExplainer Output")
    for node_idx, weight in sorted(output.items(), key=lambda x: -x[1]):
        print(node_idx, "\t", round(weight, 4))
    print()

    return output


def sample_subgraphx(model, x, edge_index, node_idx_1, node_idx_2):
    # SubgraphX, T * size(neighborhood) queries per explanation
    subgraphx = SubgraphX(model, x, edge_index, T=10)
    output = subgraphx.explain_edge(node_idx_1, node_idx_2)

    print("SubgraphX Output")
    for node_idx, weight in sorted(output.items(), key=lambda x: -x[1]):
        print(node_idx, "\t", round(weight, 4))
    print()

    return output


def sample_edge_subgraphx(model, x, edge_index, node_idx_1, node_idx_2):
    # EdgeSubgraphX, T * size(neighborhood) queries per explanation
    edge_subgraphx = EdgeSubgraphX(model, x, edge_index, T=10)
    output = edge_subgraphx.explain_edge(node_idx_1, node_idx_2)

    print("EdgeSubgraphX Output")
    for node_idx, weight in sorted(output.items(), key=lambda x: -x[1]):
        print(node_idx, "\t", round(weight, 4))
    print()

    return output


def sample_random(model, x, edge_index, node_idx_1, node_idx_2):
    # Random, 0 queries per explanation
    random_explainer = RandomExplainer(model, x, edge_index)
    output = random_explainer.explain_edge(node_idx_1, node_idx_2)

    print("Random Output")
    for node_idx, weight in sorted(output.items(), key=lambda x: -x[1]):
        print(node_idx, "\t", round(weight, 4))
    print()

    return output


if __name__ == "__main__":
    # Load the dataset and model
    x, edge_index = test_data.x, test_data.edge_index
    node_idx_1, node_idx_2 = 24, 187

    print("Nodes:", node_idx_1, node_idx_2)
    print(
        "Neighorhood Size:", get_neighbors(edge_index, node_idx_1, node_idx_2).shape[0]
    )
    print()

    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))

    # Sample the explainers
    sample_gnnexplainer(model, x, edge_index, node_idx_1, node_idx_2)
    sample_subgraphx(model, x, edge_index, node_idx_1, node_idx_2)
    sample_edge_subgraphx(model, x, edge_index, node_idx_1, node_idx_2)
    sample_random(model, x, edge_index, node_idx_1, node_idx_2)

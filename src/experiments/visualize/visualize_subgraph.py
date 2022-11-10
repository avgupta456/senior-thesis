from math import sqrt

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.dataset import dataset, device, test_data
from src.explainers.main import (  # noqa: F401
    sample_degree,
    sample_edge_subgraphx,
    sample_embedding,
    sample_gnnexplainer,
    sample_subgraphx,
)
from src.pred import Net
from src.utils import edge_centered_subgraph


def _visualize_subgraph(node_idx_1, node_idx_2, x, edge_index, edge_mask):
    # Only operate on a k-hop subgraph around `node_idx`.
    _, edge_index, mapping, subset, hard_edge_mask = edge_centered_subgraph(
        node_idx_1, node_idx_2, x, edge_index, 1
    )
    edge_mask = edge_mask[hard_edge_mask]

    # color edge explanation edge as "red"
    edge_color = ["black"] * (edge_index.size(1) - 1) + ["red"]
    y = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
    data = Data(edge_index=edge_index, att=edge_mask, edge_color=edge_color, y=y)

    G = to_networkx(data.to("cpu"), node_attrs=["y"], edge_attrs=["att", "edge_color"])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)
    pos = nx.spring_layout(G)
    ax = plt.gca()

    node_kwargs = {"node_size": 800, "cmap": "cool"}
    for source, target, data in G.edges(data=True):
        ax.annotate(
            "",
            xy=pos[target],
            xycoords="data",
            xytext=pos[source],
            textcoords="data",
            arrowprops=dict(
                arrowstyle="->",
                alpha=max(data["att"], 0.02),
                color=data["edge_color"],
                shrinkA=sqrt(node_kwargs["node_size"]) / 2.0,
                shrinkB=sqrt(node_kwargs["node_size"]) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)

    label_kwargs = {"font_size": 10}
    nx.draw_networkx_labels(G, pos, **label_kwargs)

    return ax, G


if __name__ == "__main__":
    index = 164

    model = Net(dataset.num_features, 128, 32).to(device)
    model.load_state_dict(torch.load("./models/model.pt"))

    x = test_data.x
    edge_index = test_data.edge_index
    edge_label_index = test_data.edge_label_index[:, index : index + 1]
    edge_label = test_data.edge_label[index : index + 1].item()
    y = test_data.y
    node_idx_1 = edge_label_index[0][0].item()
    node_idx_2 = edge_label_index[1][0].item()

    print(node_idx_1, node_idx_2)

    output = sample_embedding(model, x, edge_index, node_idx_1, node_idx_2, edge_label)

    # add the edge between the two nodes
    edge_index = torch.cat([edge_index, edge_label_index], dim=1)
    edge_mask = torch.zeros_like(edge_index[0], dtype=torch.float)
    sub_edge_mask = (edge_index[0] == node_idx_1) | (edge_index[0] == node_idx_2)
    for neighbor, edge_weight in output.items():
        temp_edge_mask = sub_edge_mask & (edge_index[1] == neighbor)
        # Apply **4 to make the edge weights more visible
        edge_mask[temp_edge_mask] = edge_weight**4
    # add the edge between the two nodes
    edge_mask[edge_index.shape[1] - 1] = 1

    ax, G = _visualize_subgraph(
        node_idx_1,
        node_idx_2,
        x,
        edge_index,
        edge_mask,
    )

    plt.show()

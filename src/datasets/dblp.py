import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP

from src.utils import device


def get_dblp_dataset():
    transform = T.Compose(
        [
            T.ToDevice(device),
            T.RemoveIsolatedNodes(),
            T.RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=False,
                edge_types=[("paper", "to", "author")],
            ),
            T.ToUndirected(),
        ]
    )

    dataset = DBLP(root="./data/DBLP", transform=transform)

    train_data, val_data, test_data = dataset[0]

    for data in train_data, val_data, test_data:
        del data["term"]
        del data[("paper", "to", "term")]
        del data[("term", "to", "paper")]
        del data[("author", "to", "paper")]
        del data[("conference", "to", "paper")]

        del data[("paper", "rev_to", "author")]
        del data[("term", "rev_to", "paper")]
        del data[("paper", "rev_to", "term")]
        del data[("paper", "rev_to", "conference")]

        del data["author"].train_mask
        del data["author"].val_mask
        del data["author"].test_mask
        del data["author"].y

        data["conference"].x = torch.ones((20, 1))
        del data["conference"].num_nodes

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = get_dblp_dataset()

    print("TRAIN")
    print(train_data)
    print()

    print("VAL")
    print(val_data)
    print()

    print("TEST")
    print(test_data)
    print()

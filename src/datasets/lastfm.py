import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import LastFM

from src.utils.utils import device


def get_lastfm_dataset():
    transform = T.Compose(
        [
            T.ToDevice(device),
            T.RemoveIsolatedNodes(),
            T.RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=False,
                edge_types=[("artist", "to", "user")],
            ),
            T.ToUndirected(),
        ]
    )

    dataset = LastFM(root="./data/LastFM", transform=transform)

    train_data, val_data, test_data = dataset[0]

    for data in train_data, val_data, test_data:
        del data["tag"]
        del data[("artist", "to", "tag")]
        del data[("tag", "to", "artist")]
        del data[("tag", "rev_to", "artist")]
        del data[("artist", "rev_to", "tag")]
        del data[("user", "to", "artist")]
        del data[("artist", "rev_to", "user")]

        data["user"].x = torch.randn((data["user"].num_nodes, 32))
        data["artist"].x = torch.randn((data["artist"].num_nodes, 32))

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = get_lastfm_dataset()

    print("TRAIN")
    print(train_data)
    print()

    print("VAL")
    print(val_data)
    print()

    print("TEST")
    print(test_data)
    print()

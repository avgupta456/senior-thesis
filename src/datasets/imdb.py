import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB

from src.utils import device


def get_imdb_dataset():
    transform = T.Compose(
        [
            T.ToDevice(device),
            T.RemoveIsolatedNodes(),
            T.RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=False,
                edge_types=[("movie", "to", "actor")],
            ),
            T.ToUndirected(),
        ]
    )

    dataset = IMDB(root="./data/IMDB", transform=transform)

    train_data, val_data, test_data = dataset[0]

    for data in train_data, val_data, test_data:
        del data[("director", "to", "movie")]
        del data[("actor", "to", "movie")]

        del data[("movie", "rev_to", "director")]
        del data[("movie", "rev_to", "actor")]

        del data["movie"].train_mask
        del data["movie"].val_mask
        del data["movie"].test_mask
        del data["movie"].y

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = get_imdb_dataset()

    print("TRAIN")
    print(train_data)
    print()

    print("VAL")
    print(val_data)
    print()

    print("TEST")
    print(test_data)
    print()

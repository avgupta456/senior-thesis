import torch_geometric.transforms as T
from torch_geometric.datasets import SNAPDataset

from src.utils import device


def get_facebook_dataset():
    transform = T.Compose(
        [
            T.ToDevice(device),
            T.RemoveIsolatedNodes(),
            T.RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=False,
            ),
        ]
    )

    dataset = SNAPDataset(
        root="./data/SNAPDataset", name="ego-facebook", transform=transform
    )
    train_data, val_data, test_data = dataset[0]

    # TRAIN DATA

    train_data = train_data.to_heterogeneous(
        node_type_names=["person"], edge_type_names=[("person", "to", "person")]
    )

    train_data[("person", "to", "person")]["edge_label"] = train_data[
        "person"
    ].edge_label
    train_data[("person", "to", "person")]["edge_label_index"] = train_data[
        "person"
    ].edge_label_index
    del train_data["person"].edge_label
    del train_data["person"].edge_label_index
    del train_data["person"].circle
    del train_data["person"].circle_batch

    # VAL DATA

    val_data = val_data.to_heterogeneous(
        node_type_names=["person"], edge_type_names=[("person", "to", "person")]
    )

    val_data[("person", "to", "person")]["edge_label"] = val_data["person"].edge_label
    val_data[("person", "to", "person")]["edge_label_index"] = val_data[
        "person"
    ].edge_label_index
    del val_data["person"].edge_label
    del val_data["person"].edge_label_index
    del val_data["person"].circle
    del val_data["person"].circle_batch

    # TEST DATA

    test_data = test_data.to_heterogeneous(
        node_type_names=["person"], edge_type_names=[("person", "to", "person")]
    )

    test_data[("person", "to", "person")]["edge_label"] = test_data["person"].edge_label
    test_data[("person", "to", "person")]["edge_label_index"] = test_data[
        "person"
    ].edge_label_index
    del test_data["person"].edge_label
    del test_data["person"].edge_label_index
    del test_data["person"].circle
    del test_data["person"].circle_batch

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = get_facebook_dataset()

    print("TRAIN")
    print(train_data)
    print()

    print("VAL")
    print(val_data)
    print()

    print("TEST")
    print(test_data)
    print()

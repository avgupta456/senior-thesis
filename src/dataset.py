import random

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import SNAPDataset

random.seed(0)
torch.manual_seed(0)

# Load the dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

orig_transform = T.Compose(
    [
        T.ToDevice(device),
        T.RemoveIsolatedNodes(),
    ]
)

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

if __name__ == "__main__":
    print(dataset)
    print(train_data)
    print(val_data)
    print(test_data)

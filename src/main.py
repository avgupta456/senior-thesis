import sys

import torch

from src.datasets.facebook import get_facebook_dataset
from src.datasets.imdb import get_imdb_dataset
from src.datasets.dblp import get_dblp_dataset
from src.pred.model import SimpleNet, Net
from src.utils import device


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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <dataset>")

    dataset_name = sys.argv[1]
    train_data, val_data, test_data, model, key = get_dataset_and_model(dataset_name)

    print(f"Dataset: {dataset_name}")

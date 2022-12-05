import torch

from src.datasets.facebook import get_facebook_dataset
from src.datasets.imdb import get_imdb_dataset
from src.datasets.lastfm import get_lastfm_dataset
from src.pred.model import Net, SimpleNet
from src.utils.utils import device


def get_dataset_and_model(name):
    if name == "facebook":
        train_data, val_data, test_data = get_facebook_dataset()
        model = Net(128, 32, metadata=train_data.metadata()).to(device)
        model.load_state_dict(torch.load("./models/facebook_model.pt"))
        key = ("person", "to", "person")
        gnnexplainer_config = {"edge_size": 20, "edge_ent": -1.0}
    elif name == "imdb":
        train_data, val_data, test_data = get_imdb_dataset()
        model = SimpleNet(128, 32, metadata=train_data.metadata()).to(device)
        model.load_state_dict(torch.load("./models/imdb_model.pt"))
        key = ("movie", "to", "actor")
        gnnexplainer_config = {"edge_size": 20, "edge_ent": -1.0}
    elif name == "lastfm":
        train_data, val_data, test_data = get_lastfm_dataset()
        model = SimpleNet(128, 32, metadata=train_data.metadata()).to(device)
        model.load_state_dict(torch.load("./models/lastfm_model.pt"))
        key = ("artist", "to", "user")
        gnnexplainer_config = {"edge_size": 20, "edge_ent": -1.0}
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_data, val_data, test_data, model, key, gnnexplainer_config

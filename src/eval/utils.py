import torch

from src.datasets.facebook import get_facebook_dataset
from src.datasets.imdb import get_imdb_dataset
from src.datasets.lastfm import get_lastfm_dataset
from src.metrics.fidelity import charact_prob, fid_minus_prob, fid_plus_prob
from src.pred.model import Net, SimpleNet
from src.utils.utils import device, sigmoid


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
    elif name == "lastfm":
        train_data, val_data, test_data = get_lastfm_dataset()
        model = SimpleNet(128, 32, metadata=train_data.metadata()).to(device)
        model.load_state_dict(torch.load("./models/lastfm_model.pt"))
        key = ("artist", "to", "user")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_data, val_data, test_data, model, key


def process_data(filtered_data, sampler_names):
    processed_data = {}
    for i, data in filtered_data.items():
        processed_data[i] = {}
        n = len(data["Random"])
        for name in sampler_names:
            processed_data[i][name] = []
            for j in range(n):
                _, _, expl_pred, remove_pred, initial_pred = data[name][j]
                sig_initial_pred = sigmoid(initial_pred)
                sig_expl_pred = sigmoid(expl_pred)
                sig_remove_pred = sigmoid(remove_pred)
                fid_minus = fid_minus_prob(sig_initial_pred, sig_expl_pred)
                fid_plus = fid_plus_prob(sig_initial_pred, sig_remove_pred)
                charact = charact_prob(
                    sig_initial_pred, sig_expl_pred, sig_remove_pred, 0.2, 0.8
                )
                fid_minus = min(max(fid_minus.item(), 0), 1)
                fid_plus = min(max(fid_plus.item(), 0), 1)
                charact = min(max(charact.item(), 0), 1)
                processed_data[i][name].append(
                    {"fid_minus": fid_minus, "fid_plus": fid_plus, "charact": charact}
                )
    return processed_data

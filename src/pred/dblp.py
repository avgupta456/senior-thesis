import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from src.datasets.dblp import get_dblp_dataset
from src.pred.model import SimpleNet, run
from src.utils import device


def train_model(epochs):
    train_data, val_data, test_data = get_dblp_dataset()

    model = SimpleNet(128, 32, metadata=train_data.metadata()).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-1)
    criterion = BCEWithLogitsLoss()

    key = ("paper", "to", "author")
    run(model, optimizer, criterion, train_data, val_data, test_data, key, epochs)

    torch.save(model.state_dict(), "./models/dblp_model.pt")


if __name__ == "__main__":
    epochs = 50
    train_model(epochs)

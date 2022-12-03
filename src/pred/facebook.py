import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from src.datasets.facebook import get_facebook_dataset
from src.pred.model import Net, run
from src.utils.utils import device


def train_model(epochs):
    train_data, val_data, test_data = get_facebook_dataset()

    model = Net(128, 32, metadata=train_data.metadata()).to(device)
    optimizer = Adam(params=model.parameters(), lr=3e-3)
    criterion = BCEWithLogitsLoss()

    key = ("person", "to", "person")
    run(model, optimizer, criterion, train_data, val_data, test_data, key, epochs)

    torch.save(model.state_dict(), "./models/facebook_model.pt")


if __name__ == "__main__":
    epochs = 1000
    train_model(epochs)

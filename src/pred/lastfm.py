import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from src.datasets.lastfm import get_lastfm_dataset
from src.pred.model import SimpleNet, run
from src.utils.utils import device


# TODO: Improve model performance (currently just 70% accuracy on test dataset)
def train_model(epochs):
    train_data, val_data, test_data = get_lastfm_dataset()

    model = SimpleNet(128, 32, metadata=train_data.metadata()).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = BCEWithLogitsLoss()

    key = ("artist", "to", "user")
    run(
        "lastfm",
        model,
        optimizer,
        criterion,
        train_data,
        val_data,
        test_data,
        key,
        epochs,
    )

    torch.save(model.state_dict(), "./models/final_lastfm_model.pt")


if __name__ == "__main__":
    epochs = 50
    train_model(epochs)

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from src.datasets.imdb import get_imdb_dataset
from src.pred.model import SimpleNet, run
from src.utils.utils import device


# TODO: Improve model performance (currently just 65% accuracy on test dataset)
def train_model(epochs):
    train_data, val_data, test_data = get_imdb_dataset()

    model = SimpleNet(128, 32, metadata=train_data.metadata()).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = BCEWithLogitsLoss()

    key = ("movie", "to", "actor")
    run(
        "imdb",
        model,
        optimizer,
        criterion,
        train_data,
        val_data,
        test_data,
        key,
        epochs,
    )

    torch.save(model.state_dict(), "./models/final_imdb_model.pt")


if __name__ == "__main__":
    epochs = 50
    train_model(epochs)

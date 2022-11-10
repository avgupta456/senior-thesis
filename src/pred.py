import warnings

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

from src.dataset import dataset, device, test_data, train_data, val_data

warnings.filterwarnings("ignore", category=UserWarning)


class SimpleNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        out = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        return out.unsqueeze(-1)

    def forward(self, x, edge_index, edge_label_index, data=None):
        z = self.encode(x, edge_index)
        out = self.decode(z, edge_label_index)
        return torch.hstack((-out, out)).T


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # TODO: look into SAGEConv, GATConv, GINConv, comparison between
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.W1 = nn.Linear(out_channels * 2, out_channels)
        self.W2 = nn.Linear(out_channels, 1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        z1 = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), dim=1)
        out1 = self.W2(F.relu(self.W1(z1)).squeeze())

        z2 = torch.cat((z[edge_label_index[1]], z[edge_label_index[0]]), dim=1)
        out2 = self.W2(F.relu(self.W1(z2)).squeeze())

        return (out1 + out2) / 2

    def forward(self, x, edge_index, edge_label_index, data=None):
        z = self.encode(x, edge_index)
        out = self.decode(z, edge_label_index)
        return torch.hstack((-out, out)).T


def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.shape[1],
        method="sparse",
    )

    edge_label_index = torch.cat([data.edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat(
        [data.edge_label, data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0
    )

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    a, b = data.edge_label.cpu().numpy(), out.cpu().numpy()
    c = (out > 0.5).float().cpu().numpy()
    return roc_auc_score(a, b), accuracy_score(a, c)


def train_simple_model(epochs):
    simple_model = SimpleNet(dataset.num_features, 128, 32).to(device)
    simple_optimizer = torch.optim.Adam(params=simple_model.parameters(), lr=3e-3)
    simple_criterion = nn.BCEWithLogitsLoss()

    best_val_auc = final_test_auc = final_test_acc = 0
    best_model_dict = simple_model.state_dict()
    for epoch in range(1, epochs + 1):
        loss = train(simple_model, simple_optimizer, simple_criterion, train_data)
        val_auc, val_acc = test(simple_model, val_data)
        test_auc, test_acc = test(simple_model, test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            final_test_acc = test_acc
            best_model_dict = simple_model.state_dict()
        if epoch % 50 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f} {val_acc:.4f}, Test: {test_auc:.4f} {test_acc:.4f}"
            )

    print(f"Final Test: {final_test_auc:.4f} {final_test_acc:.4f}")
    print()

    return simple_model, best_model_dict


def train_model(epochs):
    model = Net(dataset.num_features, 128, 32).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = final_test_auc = final_test_acc = 0
    best_model_dict = model.state_dict()
    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, criterion, train_data)
        val_auc, val_acc = test(model, val_data)
        test_auc, test_acc = test(model, test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            final_test_acc = test_acc
            best_model_dict = model.state_dict()
        if epoch % 50 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f} {val_acc:.4f}, Test: {test_auc:.4f} {test_acc:.4f}"
            )

    print(f"Final Test: {final_test_auc:.4f} {final_test_acc:.4f}")
    print()

    return model, best_model_dict


if __name__ == "__main__":
    epochs = 1000

    _, simple_model_dict = train_simple_model(epochs)
    _, model_dict = train_model(epochs)

    # Save the models
    torch.save(simple_model_dict, "./models/simple_model.pt")
    torch.save(model_dict, "./models/model.pt")

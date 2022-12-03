import warnings

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import negative_sampling

warnings.filterwarnings("ignore", category=UserWarning)


class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.encoder = to_hetero(
            Encoder(hidden_channels=hidden_channels, out_channels=out_channels),
            metadata,
        )

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z1, z2, edge_label_index):
        x1 = z1[edge_label_index[0]]
        x2 = z2[edge_label_index[1]]
        return (x1 * x2).sum(dim=-1)


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.encoder = to_hetero(
            Encoder(hidden_channels=hidden_channels, out_channels=out_channels),
            metadata,
        )

        self.W1 = torch.nn.Linear(out_channels * 2, out_channels)
        self.W2 = torch.nn.Linear(out_channels, 1)

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z1, z2, edge_label_index):
        z = torch.cat((z1[edge_label_index[0]], z2[edge_label_index[1]]), dim=1)
        out1 = self.W2(torch.relu(self.W1(z)).squeeze()).squeeze()

        z_rev = torch.cat((z2[edge_label_index[1]], z1[edge_label_index[0]]), dim=1)
        out2 = self.W2(torch.relu(self.W1(z_rev)).squeeze()).squeeze()

        return (out1 + out2) / 2


def train(model, optimizer, criterion, data, key):
    start, _, end = key
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x_dict, data.edge_index_dict)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index_dict[key],
        num_nodes=(data.x_dict[start].shape[0], data.x_dict[end].shape[0]),
        num_neg_samples=data.edge_label_index_dict[key].shape[1],
        method="sparse",
    )

    edge_label_index = data.edge_label_index_dict[key]
    edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=-1)

    edge_label = data.edge_label_dict[key]
    edge_label = torch.cat(
        [edge_label, edge_label.new_zeros(neg_edge_index.size(1))], dim=0
    )

    out = model.decode(z[start], z[end], edge_label_index)
    loss = criterion(out, edge_label)

    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(model, data, key):
    start, _, end = key
    model.eval()
    z = model.encode(data.x_dict, data.edge_index_dict)
    out = (
        model.decode(z[start], z[end], data.edge_label_index_dict[key])
        .view(-1)
        .sigmoid()
    )
    a, b = data.edge_label_dict[key].cpu().numpy(), out.cpu().numpy()
    c = (out > 0.5).float().cpu().numpy()

    return roc_auc_score(a, b), accuracy_score(a, c)


def run(model, optimizer, criterion, train_data, val_data, test_data, key, epochs):
    best_val_auc = final_test_auc = final_test_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, criterion, train_data, key)
        val_auc, val_acc = test(model, val_data, key)
        test_auc, test_acc = test(model, test_data, key)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            final_test_acc = test_acc

        if epoch % 1 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f} {val_acc:.4f}, Test: {test_auc:.4f} {test_acc:.4f}"
            )

    print(f"Final Test: {final_test_auc:.4f} {final_test_acc:.4f}")

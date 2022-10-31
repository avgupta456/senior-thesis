import torch

from dataset import device, dataset, test_data
from pred import Net

from explainers.gnnexplainer import GNNExplainer
from explainers.subgraphx import SubgraphX
from explainers.random import RandomExplainer

# Load the dataset and model
x, edge_index = test_data.x, test_data.edge_index
node_idx_1, node_idx_2 = 24, 187

model = Net(dataset.num_features, 128, 32).to(device)
model.load_state_dict(torch.load('./models/model.pt'))

# GNNExplainer
gnnexplainer = GNNExplainer(model, x, edge_index, epochs=200, lr=0.1)
output = gnnexplainer.explain_edge(node_idx_1, node_idx_2)

print("GNNExplainer Output")
for node_idx, weight in sorted(output.items(), key=lambda x: -x[1]):
    print(node_idx, "\t", round(weight, 4))
print()

# SubgraphX
subgraphx = SubgraphX(model, x, edge_index, T=10)
output = subgraphx.explain_edge(node_idx_1, node_idx_2)

print("SubgraphX Output")
for node_idx, weight in sorted(output.items(), key=lambda x: -x[1]):
    print(node_idx, "\t", round(weight, 4))
print()

# Random
random_explainer = RandomExplainer(model, x, edge_index)
output = random_explainer.explain_edge(node_idx_1, node_idx_2)

print("Random Output")
for node_idx, weight in sorted(output.items(), key=lambda x: -x[1]):
    print(node_idx, "\t", round(weight, 4))
print()

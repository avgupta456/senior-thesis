from src.explainers.degree import DegreeExplainer
from src.explainers.edge_subgraphx import EdgeSubgraphX
from src.explainers.embedding import EmbeddingExplainer
from src.explainers.gnnexplainer import GNNExplainer
from src.explainers.random import RandomExplainer
from src.explainers.subgraphx import SubgraphX


def sample_gnnexplainer(
    model, x, edge_index, node_idx_1, node_idx_2, target, epochs=100, lr=0.01
):
    # GNNExplainer, 100 queries per explanation
    gnnexplainer = GNNExplainer(model, x, edge_index, epochs=epochs, lr=lr)
    return gnnexplainer.explain_edge(node_idx_1, node_idx_2, target)


def sample_subgraphx(model, x, edge_index, node_idx_1, node_idx_2, target, T=5):
    # SubgraphX, 5 * size(neighborhood) queries per explanation
    subgraphx = SubgraphX(model, x, edge_index, T=T)
    return subgraphx.explain_edge(node_idx_1, node_idx_2, target)


def sample_edge_subgraphx(model, x, edge_index, node_idx_1, node_idx_2, target, T=5):
    # EdgeSubgraphX, 5 * size(neighborhood) queries per explanation
    edge_subgraphx = EdgeSubgraphX(model, x, edge_index, T=T)
    return edge_subgraphx.explain_edge(node_idx_1, node_idx_2, target)


def sample_embedding(model, x, edge_index, node_idx_1, node_idx_2, target):
    embedding_explainer = EmbeddingExplainer(model, x, edge_index)
    return embedding_explainer.explain_edge(node_idx_1, node_idx_2, target)


def sample_degree(model, x, edge_index, node_idx_1, node_idx_2, target):
    degree_explainer = DegreeExplainer(model, x, edge_index)
    return degree_explainer.explain_edge(node_idx_1, node_idx_2, target)


def sample_random(model, x, edge_index, node_idx_1, node_idx_2, target):
    random_explainer = RandomExplainer(model, x, edge_index)
    return random_explainer.explain_edge(node_idx_1, node_idx_2, target)

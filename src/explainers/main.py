from src.explainers.edge_subgraphx import EdgeSubgraphX
from src.explainers.embedding import EmbeddingExplainer
from src.explainers.gnnexplainer import GNNExplainer
from src.explainers.random import RandomExplainer
from src.explainers.subgraphx import SubgraphX


def sample_gnnexplainer(
    model,
    data,
    node_idx_1,
    node_1_type,
    node_idx_2,
    node_2_type,
    target,
    epochs=100,
    lr=0.01,
):
    # GNNExplainer, 100 queries per explanation
    gnnexplainer = GNNExplainer(model, epochs=epochs, lr=lr)
    return gnnexplainer.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type, target
    )


def sample_subgraphx(
    model, data, node_idx_1, node_1_type, node_idx_2, node_2_type, target, T=5
):
    # SubgraphX, 5 * size(neighborhood) queries per explanation
    subgraphx = SubgraphX(model, T=T)
    return subgraphx.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type, target
    )


def sample_edge_subgraphx(
    model, data, node_idx_1, node_1_type, node_idx_2, node_2_type, target, T=5
):
    # EdgeSubgraphX, 5 * size(neighborhood) queries per explanation
    edge_subgraphx = EdgeSubgraphX(model, T=T)
    return edge_subgraphx.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type, target
    )


def sample_embedding(
    model, data, node_idx_1, node_1_type, node_idx_2, node_2_type, target
):
    embedding_explainer = EmbeddingExplainer(model)
    return embedding_explainer.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type, target
    )


def sample_random(
    model, data, node_idx_1, node_1_type, node_idx_2, node_2_type, target
):
    random_explainer = RandomExplainer(model)
    return random_explainer.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type, target
    )

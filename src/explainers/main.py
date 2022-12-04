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
    epochs=100,
    lr=0.01,
    **kwargs
):
    # GNNExplainer, 100 queries per explanation
    gnnexplainer = GNNExplainer(model, epochs=epochs, lr=lr, **kwargs)
    return gnnexplainer.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type
    )


def sample_subgraphx(
    model, data, node_idx_1, node_1_type, node_idx_2, node_2_type, T=5
):
    # SubgraphX, 5 * size(neighborhood) queries per explanation
    subgraphx = SubgraphX(model, T=T)
    return subgraphx.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type
    )


def sample_embedding(model, data, node_idx_1, node_1_type, node_idx_2, node_2_type):
    embedding_explainer = EmbeddingExplainer(model)
    return embedding_explainer.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type
    )


def sample_random(model, data, node_idx_1, node_1_type, node_idx_2, node_2_type):
    random_explainer = RandomExplainer(model)
    return random_explainer.explain_edge(
        data, node_idx_1, node_1_type, node_idx_2, node_2_type
    )

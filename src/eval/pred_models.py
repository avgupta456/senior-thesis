import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.eval.utils import get_dataset_and_model


def evaluate_pred_model(data, model, key):
    data = test_data
    start, _, end = key

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x_dict, data.edge_index_dict)
        out = (
            model.decode(z[start], z[end], data.edge_label_index_dict[key])
            .view(-1)
            .sigmoid()
        )

    a, b = data.edge_label_dict[key].cpu().numpy(), out.cpu().numpy()
    c = (out > 0.5).float().cpu().numpy()

    print("ROC AUC:\t", roc_auc_score(a, b))
    print("Accuracy:\t", accuracy_score(a, c))


if __name__ == "__main__":
    for dataset_name in ["facebook", "imdb", "lastfm"]:
        (
            train_data,
            val_data,
            test_data,
            model,
            key,
            gnnexplainer_config,
        ) = get_dataset_and_model(dataset_name)

        print(dataset_name)
        evaluate_pred_model(test_data, model, key)
        print()

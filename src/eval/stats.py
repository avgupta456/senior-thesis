from src.eval.utils import get_dataset_and_model

if __name__ == "__main__":
    (
        train_data,
        val_data,
        test_data,
        model,
        key,
        gnnexplainer_config,
    ) = get_dataset_and_model("imdb")

    data = test_data

    print(
        model.forward(
            data.x_dict,
            data.edge_index_dict,
            data.edge_label_index_dict[key],
            key,
        )
    )

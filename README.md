## Senior Thesis: Link Explanation for Heterogeneous Graphs

### Abstract

Graph Neural Networks (GNNs) utilize message passing to operate directly on graph structured data, and have achieved state-of-the-art performance in node classification, link prediction, and graph classification tasks. In order to build trust, promote transparency, and facilitate real-world applications, GNNs must be made explainable. But while models such as GNNExplainer and SubgraphX explain node and graph classification tasks, link explanation, particularly on heterogeneous graphs, remains underexplored. In this work, we propose a new explanation format for link prediction, where only a subset of immediate neighbors of the target edge are selected. We test four explanation methods on the Facebook, IMDB, and LastFM heterogeneous graph datasets. Each dataset has distinct properties that encourage a robust explanation model. We sample explanations of varying sparsity and measure the characterization score to assess a candidate explanation's necessity and sufficiency. We report characterization scores for 1-node, 5-node, and 10-node explanations. We make key changes to the GNNExplainer loss function and to the SubgraphX filtering method that yield significant improvements in explanation quality. Overall, we find our modified SubgraphX method outperforms existing baselines by $11\%$ and our modified GNNExplainer outperforms existing baselines by $21\%$ on average. The embedding baseline remains a fast and viable approach for small explanations on high degree graphs, while GNNExplainer produces the strongest explanations across all other configurations. Additionally, we make open-source contributions to the PyTorch Geometric library to allow for future extensions. Altogether, our work is the first exploration of heterogeneous link explanation and lays the foundation for future explanation approaches.

### File Structure

- `data/` contains the datasets used in this project. These include the Facebook Ego Network, IMDB, and LastFM datasets.
- `docs/` contains all documentation for the project, including the Project Proposal, Final Report, Poster Presentation, and Presentation Slides.
- `models/` contains the link prediction models for each dataset. These are loaded during link explanation.
- `notebooks/` contains all Jupyter notebooks, primarly for data exploration and debugging.
- `results/` contains the raw data and figures of the link explanation experiments. This is separated into `poster_figures` and `report_figures` for the poster and final report, respectively.
- `src/` contains all source code for the project. This is further separated into several components.
  - `src/datasets/` performs data loading and preprocessing for the datasets.
  - `src/pred/` contains code to train and load the link prediction models for each dataset.
  - `src/explainers/` contains the code for the explanation methods. These include the Random Explainer, Embedding Explainer, GNNExplainer, and SubgraphX.
  - `src/metrics/` contains functions for Fidelity and Characterization scores for evaluating explanations.
  - `src/eval/` contains scripts to run the explanation experiments, generate figures, print statistics, and create LaTeX tables for the final report.
  - `src/utils/` contains utility functions for data manipulation.

### Usage

#### Installation

This project requires Python 3. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

#### Prediction Models

The link prediction models are trained using the PyTorch Geometric library. To train the models, run the following command:

```bash
python -m src.pred.facebook
python -m src.pred.imdb
python -m src.pred.lastfm
```

#### Explanation Methods

The explanation methods are run using the `src/eval/process.py` script. This script takes the following arguments in order:

- Dataset: The dataset to run the explanation method on. This can be `facebook`, `imdb`, or `lastfm`.
- Start Index: The index of the first edge to explain in the test data edge label index.
- End Index: The index of the last edge to explain in the test data edge label index.
- Show Plots: Whether to show plots of the explanation results. This can be `True` or `False`.

To reproduce the Facebook results, run the following command:

```bash
python -m src.eval.process facebook 0 1000 False
```

To reproduce the IMDB results, run the following command:

```bash
python -m src.eval.process imdb 0 10000 False
```

To reproduce the LastFM results, run the following command:

```bash
python -m src.eval.process lastfm 0 1000 False
```

#### Figures

The figures for the final report and poster are generated using the `src/eval/visualize.py` script. This script takes the following arguments in order:

- Dataset: The dataset to run the explanation method on. This can be `facebook`, `imdb`, or `lastfm`.
- Start Index: The index of the first edge to explain in the test data edge label index.
- End Index: The index of the last edge to explain in the test data edge label index.
- Subset: The subset of explanation methods to generate figures for. This can be `all`, `original`, `gnnexplainer`, `subgraphx`, or `final`.

To generate the figures for the Facebook dataset, run the following command:

```bash
python -m src.eval.visualize facebook 0 1000 final
```

To generate the figures for the IMDB dataset, run the following command:

```bash
python -m src.eval.visualize imdb 0 10000 final
```

To generate the figures for the LastFM dataset, run the following command:

```bash
python -m src.eval.visualize lastfm 0 1000 final
```

#### Statistics

Run the following command to print the statistics for the datasets and link prediction models:

```bash
python -m src.eval.statistics
```

#### LaTeX Tables

Run the following command to generate the LaTeX tables for the final report:

```bash
python -m src.eval.latex_table original
python -m src.eval.latex_table gnnexplainer
python -m src.eval.latex_table subgraphx
python -m src.eval.latex_table final
```

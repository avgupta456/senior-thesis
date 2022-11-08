import json


def common_explanations(data):
    samplers = list(data.keys())
    print("\t\t" + "\t".join(samplers[:-1]) + "\t\t" + "Random" + "\t\t" + "Average")
    neighbors = [x[0] for x in data[samplers[0]][1:]]
    neighbor_ranks = {n: {} for n in neighbors}
    for sampler in samplers:
        for i, x in enumerate(data[sampler][1:]):
            neighbor_ranks[x[0]][sampler] = i
    for n in neighbors:
        neighbor_ranks[n]["Average"] = (
            neighbor_ranks[n]["GNNExplainer"]
            + neighbor_ranks[n]["SubgraphX"]
            + neighbor_ranks[n]["EdgeSubgraphX"]
            + neighbor_ranks[n]["Embedding"]
        )
        neighbor_ranks[n]["Average"] = round(neighbor_ranks[n]["Average"] / 4, 2)
    for n in sorted(neighbors, key=lambda x: neighbor_ranks[x]["Average"]):
        print(n, end="\t\t")
        for sampler in samplers:
            print(neighbor_ranks[n][sampler], end="\t\t")
        print(neighbor_ranks[n]["Average"])


if __name__ == "__main__":
    with open("./results/vary_sparsity/data_0_200.json", "r") as f:
        all_data = json.load(f)
    common_explanations(list(all_data.values())[5])

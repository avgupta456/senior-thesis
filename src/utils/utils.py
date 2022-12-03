import random

import numpy as np
import torch

random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    return 1 / (1 + np.exp(-x))


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return tensor_to_numpy(x)
    return x


# TODO: delete if unused
def _mask_nodes(x, mask):
    new_x = x.clone()
    new_x[~mask] = 0
    return new_x

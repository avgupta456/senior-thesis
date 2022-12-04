import numpy as np

from src.utils.utils import _to_numpy

"""
Phenomenon Metrics
"""


def fid_plus_phenom(ground_truth, full_pred, remove_pred):
    # Necessary explanation maximizes fid+
    ground_truth = _to_numpy(ground_truth)
    full_pred = _to_numpy(full_pred)
    remove_pred = _to_numpy(remove_pred)
    return np.mean(
        (full_pred == ground_truth).astype(int)
        - (remove_pred == ground_truth).astype(int)
    )


def fid_minus_phenom(ground_truth, full_pred, expl_pred):
    # Sufficient explanation maximizes (1 - fid-)
    ground_truth = _to_numpy(ground_truth)
    full_pred = _to_numpy(full_pred)
    expl_pred = _to_numpy(expl_pred)
    return np.mean(
        (full_pred == ground_truth).astype(int)
        - (expl_pred == ground_truth).astype(int)
    )


def charact_phenom(
    ground_truth,
    full_pred,
    expl_pred,
    remove_pred,
    w_plus: int = 0.5,
    w_minus: int = 0.5,
):
    fid_plus = fid_plus_phenom(ground_truth, full_pred, remove_pred)
    fid_minus = 1 - fid_minus_phenom(ground_truth, full_pred, expl_pred)
    return ((w_plus + w_minus) * fid_plus * fid_minus) / (
        w_plus * fid_minus + w_minus * fid_plus
    )


"""
Model Metrics
"""


def fid_plus_model(full_pred, remove_pred):
    # Necessary explanation maximizes fid+
    full_pred = _to_numpy(full_pred)
    remove_pred = _to_numpy(remove_pred)
    return 1 - np.mean((remove_pred == full_pred).astype(int))


def fid_minus_model(full_pred, expl_pred):
    # Sufficient explanation maximizes (1 - fid-)
    full_pred = _to_numpy(full_pred)
    expl_pred = _to_numpy(expl_pred)
    return 1 - np.mean((expl_pred == full_pred).astype(int))


def charact_model(
    full_pred, expl_pred, remove_pred, w_plus: int = 0.5, w_minus: int = 0.5
):
    fid_plus = fid_plus_model(full_pred, remove_pred)
    fid_minus = 1 - fid_minus_model(full_pred, expl_pred)
    return ((w_plus + w_minus) * fid_plus * fid_minus) / (
        w_plus * fid_minus + w_minus * fid_plus
    )


"""
Probabalistic Metrics
"""


def fid_plus_prob(full_pred, remove_pred):
    # Necessary explanation maximizes fid+
    full_pred = _to_numpy(full_pred)
    remove_pred = _to_numpy(remove_pred)
    return np.mean((full_pred - remove_pred))


def fid_minus_prob(full_pred, expl_pred):
    # Sufficient explanation maximizes (1 - fid-)
    full_pred = _to_numpy(full_pred)
    expl_pred = _to_numpy(expl_pred)
    return np.mean((full_pred - expl_pred))


def charact_prob(
    full_pred, expl_pred, remove_pred, w_plus: int = 0.5, w_minus: int = 0.5
):
    fid_plus = fid_plus_prob(full_pred, remove_pred)
    fid_minus = 1 - fid_minus_prob(full_pred, expl_pred)
    return ((w_plus + w_minus) * fid_plus * fid_minus) / (
        w_plus * fid_minus + w_minus * fid_plus
    )


if __name__ == "__main__":
    ground_truth = np.array([1, 1, 1, 1, 0, 0])
    full_pred = np.array([1, 1, 1, 0, 0, 0])
    expl_pred = np.array([0, 1, 1, 1, 0, 0])
    remove_pred = np.array([1, 0, 0, 0, 0, 1])

    print("FID+ Phenom:", fid_plus_phenom(ground_truth, full_pred, remove_pred))
    print("FID- Phenom:", 1 - fid_minus_phenom(ground_truth, full_pred, expl_pred))
    print(
        "Characterization Phenom:",
        charact_phenom(ground_truth, full_pred, expl_pred, remove_pred),
    )
    print("FID+ Model:", fid_plus_model(full_pred, remove_pred))
    print("FID- Model:", 1 - fid_minus_model(full_pred, expl_pred))
    print("Characterization Model:", charact_model(full_pred, expl_pred, remove_pred))

    full_pred = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    expl_pred = np.array([0.4, 0.5, 0.75, 0.85, 0.8, 0.9])
    remove_pred = np.array([0.5, 0.2, 0.5, 0.7, 0.3, 0.4])

    print("FID+ Prob:", fid_plus_prob(full_pred, remove_pred))
    print("FID- Prob:", 1 - fid_minus_prob(full_pred, expl_pred))
    print("Characterization Prob:", charact_prob(full_pred, expl_pred, remove_pred))

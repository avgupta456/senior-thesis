import numpy as np
from numpy import ndarray


def fid_plus_phenom(ground_truth: ndarray, full_pred: ndarray, remove_pred: ndarray):
    # Necessary explanation maximizes fid+
    return np.mean(
        (full_pred == ground_truth).astype(int)
        - (remove_pred == ground_truth).astype(int)
    )


def fid_minus_phenom(ground_truth: ndarray, full_pred: ndarray, expl_pred: ndarray):
    # Sufficient explanation maximizes (1 - fid-)
    return np.mean(
        (full_pred == ground_truth).astype(int)
        - (expl_pred == ground_truth).astype(int)
    )


def charact_phenom(
    ground_truth: ndarray,
    full_pred: ndarray,
    expl_pred: ndarray,
    remove_pred: ndarray,
    w_plus: int = 0.5,
    w_minus: int = 0.5,
):
    fid_plus = fid_plus_phenom(ground_truth, full_pred, remove_pred)
    fid_minus = 1 - fid_minus_phenom(ground_truth, full_pred, expl_pred)
    return ((w_plus + w_minus) * fid_plus * fid_minus) / (
        w_plus * fid_minus + w_minus * fid_plus
    )


def fid_plus_model(full_pred: ndarray, remove_pred: ndarray):
    # Necessary explanation maximizes fid+
    return 1 - np.mean((remove_pred == full_pred).astype(int))


def fid_minus_model(full_pred: ndarray, expl_pred: ndarray):
    # Sufficient explanation maximizes (1 - fid-)
    return 1 - np.mean((expl_pred == full_pred).astype(int))


def charact_model(
    full_pred: ndarray,
    expl_pred: ndarray,
    remove_pred: ndarray,
    w_plus: int = 0.5,
    w_minus: int = 0.5,
):
    fid_plus = fid_plus_model(full_pred, remove_pred)
    fid_minus = 1 - fid_minus_model(full_pred, expl_pred)
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

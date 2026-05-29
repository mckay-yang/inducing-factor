import numpy as np
from typing import Tuple


def spatial_block_cv_split(
    coords: np.ndarray,
    n_folds: int = 5,
    fold: int = 0,
    block_size: float = 10.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x_min, y_min = coords.min(axis=0)
    block_x = ((coords[:, 0] - x_min) / block_size).astype(int)
    block_y = ((coords[:, 1] - y_min) / block_size).astype(int)
    block_ids = block_x * 10000 + block_y

    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)
    fold_assignment = {bid: i % n_folds for i, bid in enumerate(unique_blocks)}

    sample_folds = np.array([fold_assignment.get(int(bid), 0) for bid in block_ids])
    test_idx = np.where(sample_folds == fold)[0]
    train_idx = np.where(sample_folds != fold)[0]
    return train_idx, test_idx


def morans_i(
    values: np.ndarray,
    coords: np.ndarray,
    k: int = 8,
) -> Tuple[float, float]:
    n = len(values)
    if n < 3:
        return 0.0, 1.0

    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    rows_list, cols_list = [], []
    for i in range(n):
        for j in indices[i][1:]:
            rows_list.append(i)
            cols_list.append(j)
    W = csr_matrix((np.ones(len(rows_list)), (rows_list, cols_list)), shape=(n, n))
    W = W + W.T
    W = W.maximum(csr_matrix(W.shape))

    z = values - values.mean()
    S0 = float(W.sum())
    if S0 == 0:
        return 0.0, 1.0

    Wz = W.dot(z)
    numerator = float(np.sum(z * Wz))
    denominator = float(np.sum(z ** 2))
    if denominator == 0:
        return 0.0, 1.0

    I = (n / S0) * (numerator / denominator)

    n_perm = 99
    I_perm = np.zeros(n_perm)
    rng = np.random.default_rng(42)
    for p in range(n_perm):
        z_perm = rng.permutation(z)
        Wz_perm = W.dot(z_perm)
        I_perm[p] = (n / S0) * (float(np.sum(z_perm * Wz_perm)) / denominator)

    p_value = float(np.mean(I_perm >= I)) if I > 0 else float(np.mean(I_perm <= I))
    return float(I), p_value
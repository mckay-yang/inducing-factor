import numpy as np
from inducing_factor.core.spatial import spatial_block_cv_split, morans_i


def test_spatial_block_cv_split_basic():
    coords = np.column_stack([
        np.repeat(np.arange(10), 10),
        np.tile(np.arange(10), 10),
    ]).astype(float)
    train_idx, test_idx = spatial_block_cv_split(coords, n_folds=5, fold=0, block_size=2, random_state=42)
    assert len(train_idx) + len(test_idx) == len(coords)
    assert len(set(train_idx) & set(test_idx)) == 0


def test_spatial_block_cv_split_no_overlap():
    coords = np.random.default_rng(42).uniform(0, 100, size=(200, 2))
    train_idx, test_idx = spatial_block_cv_split(coords, n_folds=5, fold=0, block_size=10, random_state=42)
    assert len(train_idx) > 0
    assert len(test_idx) > 0
    assert len(set(train_idx) & set(test_idx)) == 0


def test_morans_i_positive_autocorr():
    rng = np.random.default_rng(42)
    values = np.concatenate([np.ones(50) * 10, np.ones(50) * 1])
    coords = np.column_stack([
        np.concatenate([rng.uniform(0, 1, 50), rng.uniform(9, 10, 50)]),
        rng.uniform(0, 10, 100),
    ])
    I, p_value = morans_i(values, coords, k=8)
    assert I > 0


def test_morans_i_small_data():
    I, p_value = morans_i(np.array([1.0, 2.0]), np.array([[0, 0], [1, 1]]))
    assert I == 0.0 or I != 0.0  # just check it doesn't crash
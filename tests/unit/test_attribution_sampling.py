import numpy as np

from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem


def test_sample_chunk_indices_only_uses_flagged_background_rows(tmp_path):
    npz_path = tmp_path / "chunk.npz"
    np.savez(
        npz_path,
        disturbance_time_value=np.array([0.0, 0.0, 2020.5, 0.0, 2020.75, 0.0]),
        point_id=np.array([10, 11, 12, 13, 14, 15]),
        background_sample_keep=np.array([0, 1, 0, 1, 0, 0], dtype=np.int8),
    )

    system = EcoDisturbanceAttributionSystem(
        cache_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        background_sample_rate=1.0,
    )

    selected_indices = system._sample_chunk_indices(str(npz_path))

    assert selected_indices.tolist() == [1, 2, 3, 4]

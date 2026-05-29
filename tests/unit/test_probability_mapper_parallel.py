import numpy as np
import pandas as pd

from inducing_factor.config.models import AppConfig, AttributionConfig, DataSourcesConfig, PathsConfig, PipelineConfig
from inducing_factor.pipeline.probability_mapper import AnnualProbabilityMapper


def test_write_maps_uses_parallel_prediction_iterator(monkeypatch, tmp_path):
    config = AppConfig(
        paths=PathsConfig(cache_dir=str(tmp_path / "cache"), out_dir=str(tmp_path / "out")),
        pipeline=PipelineConfig(),
        attribution=AttributionConfig(predict_num_workers=2),
        data_sources=DataSourcesConfig(dem_tif=str(tmp_path / "dem.tif")),
    )

    class FakeSystem:
        def __init__(self):
            self.selected_factors = [object()]
            self.discretizer = object()
            self.occurrence_engine = type("Occ", (), {"model": object()})()
            self.predict_num_workers = 2
            self.predict_max_auto_workers = 4
            self.predict_memory_target_fraction = 0.7
            self.predict_estimated_worker_gb = 1.0

        def _iter_chunk_files(self):
            return ["chunk_a.npz", "chunk_b.npz"]

        def load_artifacts(self):
            return self

    mapper = AnnualProbabilityMapper(config=config, system=FakeSystem())

    class FakeReadDataset:
        profile = {"dtype": "float32", "count": 1}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    writes = []

    class FakeWriteDataset:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, arr, band, window=None):
            writes.append((arr.shape, band, window))

    def fake_open(path, mode="r", **kwargs):
        if mode == "r":
            return FakeReadDataset()
        return FakeWriteDataset()

    monkeypatch.setattr("inducing_factor.pipeline.probability_mapper.rasterio.open", fake_open)

    seen = {}

    def fake_execute_chunk_tasks(tasks, **kwargs):
        task_list = list(tasks)
        seen["tasks"] = task_list
        seen["worker_count"] = kwargs["worker_count"]
        payload = []
        for task in task_list:
            pred = pd.DataFrame(
                {
                    "raster_row": np.array([10, 11]),
                    "raster_col": np.array([20, 21]),
                    "disturbance_probability": np.array([0.6, 0.4]),
                    "dominant_disturbance_class": np.array([1, 2]),
                    "class_probability_1": np.array([0.7, 0.2]),
                    "class_probability_2": np.array([0.2, 0.6]),
                    "class_probability_3": np.array([0.05, 0.1]),
                    "class_probability_4": np.array([0.05, 0.1]),
                }
            )
            payload.append((task.chunk_idx, pred, None))
        return iter(payload)

    monkeypatch.setattr("inducing_factor.pipeline.probability_mapper.execute_chunk_tasks", fake_execute_chunk_tasks)

    outputs = mapper.write_maps(2020)

    assert seen["worker_count"] == 2
    assert [task.chunk_idx for task in seen["tasks"]] == [1, 2]
    assert "disturbance_probability" in outputs
    assert len(writes) > 0

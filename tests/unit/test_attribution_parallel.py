from collections import Counter

import pandas as pd

from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem


def build_sample_frame(seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": [seed * 10 + 1, seed * 10 + 2],
            "raster_row": [1, 2],
            "raster_col": [3, 4],
            "x": [90.1, 90.2],
            "y": [28.1, 28.2],
            "year": [2019, 2019],
            "disturbance_time_value": [2019.5, 0.0],
            "disturbance_type": [2019.5, 0.0],
            "disturbance_class": [1, 0],
            "disturbance_month": [6.0, 8.0],
            "reference_month": [6.0, 8.0],
            "disturbance_flag": [1, 0],
            "slope": [3.0, 3.5],
            "aspect": [2.0, 2.0],
            "dist_to_river": [11.0, 12.0],
            "temperature_prev_last": [10.0, 10.5],
            "resident_prev_last": [1.0, 1.0],
            "dist_to_road_prev_last": [100.0, 100.0],
            "dist_to_railway_prev_last": [50.0, 50.0],
            "dist_to_waterway_prev_last": [20.0, 20.0],
            "landuse_prev_last": [2, 3],
            "rainfall_prev_1m": [20.0, 19.0],
            "rainfall_prev_2m": [18.0, 18.0],
            "rainfall_prev_3m": [17.0, 16.0],
            "rainfall_prev_total_3m": [55.0, 53.0],
        }
    )


def test_run_parallel_sample_iterator_uses_execute_chunk_tasks(monkeypatch, tmp_path):
    system = EcoDisturbanceAttributionSystem(cache_dir=str(tmp_path), out_dir=str(tmp_path / "out"))
    system.run_num_workers = 2

    chunk_files = ["a.npz", "b.npz"]
    monkeypatch.setattr(system, "_iter_chunk_files", lambda: chunk_files)

    seen = {}

    def fake_execute_chunk_tasks(tasks, **kwargs):
        seen["tasks"] = list(tasks)
        seen["worker_count"] = kwargs["worker_count"]
        return iter([(task.chunk_idx, f"chunk-{task.chunk_idx}", build_sample_frame(task.chunk_idx), None) for task in seen["tasks"]])

    monkeypatch.setattr("inducing_factor.core.attribution_system.execute_chunk_tasks", fake_execute_chunk_tasks)

    rows = list(system._load_chunk_samples(chunk_files))

    assert len(rows) == 2
    assert seen["worker_count"] == 2
    assert [task.chunk_idx for task in seen["tasks"]] == [1, 2]


def test_streaming_stage_two_merges_parallel_pattern_counts(monkeypatch, tmp_path):
    system = EcoDisturbanceAttributionSystem(cache_dir=str(tmp_path), out_dir=str(tmp_path / "out"), bootstrap_iterations=1)
    system.run_num_workers = 2
    sample = build_sample_frame(1)
    chunk_files = ["chunk_a.npz", "chunk_b.npz"]

    monkeypatch.setattr(system, "_iter_chunk_files", lambda: chunk_files)

    def fake_load_chunk_samples(files):
        for idx, path in enumerate(files, start=1):
            yield idx, path, sample.copy(), None

    monkeypatch.setattr(system, "_load_chunk_samples", fake_load_chunk_samples)

    def fake_map_pattern_counts(files, num_workers=1):
        return [
            {
                "event_counts": Counter({("a",): 2}),
                "background_counts": Counter({("a",): 1}),
                "n_event": 2,
                "n_bg": 1,
                "event_label_sample": pd.DataFrame({"f": ["x", "y"]}),
                "background_label_sample": pd.DataFrame({"f": ["z"]}),
                "event_training_sample": sample[sample["disturbance_flag"] == 1].copy(),
                "background_training_sample": sample[sample["disturbance_flag"] == 0].copy(),
            },
            {
                "event_counts": Counter({("a",): 3}),
                "background_counts": Counter({("a",): 2}),
                "n_event": 3,
                "n_bg": 2,
                "event_label_sample": pd.DataFrame({"f": ["m"]}),
                "background_label_sample": pd.DataFrame({"f": ["n"]}),
                "event_training_sample": sample[sample["disturbance_flag"] == 1].copy(),
                "background_training_sample": sample[sample["disturbance_flag"] == 0].copy(),
            },
        ]

    monkeypatch.setattr(system, "_map_pattern_counts", fake_map_pattern_counts)

    class DummyDiscretizer:
        features_ = ["f"]

        def fit(self, X, y):
            return self

        def transform(self, X, as_labels=False):
            if as_labels:
                return pd.DataFrame({"f": ["x"] * len(X)}, index=X.index)
            return X

    system.discretizer = DummyDiscretizer()

    class DummyFactorMiner:
        max_bootstrap_samples = 10

        def mine_pattern_candidates_from_counts(self, event_counts, background_counts, n_event, n_bg):
            assert event_counts[("a",)] == 5
            assert background_counts[("a",)] == 3
            assert n_event == 5
            assert n_bg == 3
            return {("a",): {"support_event": 1.0, "support_background": 1.0, "relative_risk": 1.0}}

        def transactions_from_labels(self, df):
            return [tuple(df.columns)] * len(df)

        def estimate_pattern_stability(self, event_transactions, background_transactions):
            return {("a",): 1.0}

        def reduce_redundant_patterns(self, pattern_candidates, pattern_stability):
            return {("a",): {"support_event": 1.0, "support_background": 1.0, "relative_risk": 1.0, "stability": 1.0}}

        def build_factor_library(self, reduced_patterns, n_event=0, n_background=0):
            from inducing_factor.core.models import InducingFactor

            return [
                InducingFactor(
                    factor_id="IF_001",
                    name="a",
                    semantics="a",
                    items=("a",),
                    support_event=1.0,
                    support_background=1.0,
                    relative_risk=1.0,
                    stability=1.0,
                    score=1.0,
                )
            ]

        def to_frame(self, library):
            return pd.DataFrame({"factor_id": ["IF_001"]})

    system.factor_miner = DummyFactorMiner()

    class DummyArtifacts:
        def save_factor_library(self, *args, **kwargs):
            return None

        def save_training_samples(self, *args, **kwargs):
            return None

        def save_occurrence_model(self, *args, **kwargs):
            return None

        def save_class_model(self, *args, **kwargs):
            return None

        def save_model_metadata(self, *args, **kwargs):
            return None

        def save_attribution_preview(self, *args, **kwargs):
            return None

    system.artifacts = DummyArtifacts()

    class DummyOccurrence:
        model = object()

        def fit(self, *args, **kwargs):
            return None

    system.occurrence_engine = DummyOccurrence()

    class DummyAttribution:
        model_backend = "logistic"
        inference_engine = None
        network_feature_columns = []

        def fit(self, *args, **kwargs):
            return object()

    system.attribution_engine = DummyAttribution()

    class DummyClass:
        model = object()
        model_status = "ok"
        class_labels = [1, 2, 3, 4]

        def fit(self, *args, **kwargs):
            return None

        def summarize_factor_class_associations(self, *args, **kwargs):
            return pd.DataFrame({"factor_id": ["IF_001"]})

        def predict_proba(self, *args, **kwargs):
            return pd.DataFrame({"class_probability_1": [0.7], "class_probability_2": [0.2], "class_probability_3": [0.05], "class_probability_4": [0.05]})

    system.class_engine = DummyClass()

    monkeypatch.setattr(system, "attribute_dataframe", lambda samples, limit=100: pd.DataFrame({"ok": [1]}))

    result = system._streaming_run()

    assert not result["training_samples"].empty
    assert not result["factors"].empty

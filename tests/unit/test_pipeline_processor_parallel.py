from dataclasses import asdict

import pytest

from inducing_factor.config.models import AppConfig, AttributionConfig, DataSourcesConfig, PathsConfig, PipelineConfig
from inducing_factor.pipeline.execution import ChunkAudit, ChunkResult, ChunkTask
from inducing_factor.pipeline.processor import PipelineProcessor


def build_test_config(tmp_path):
    return AppConfig(
        paths=PathsConfig(cache_dir=str(tmp_path), out_dir=str(tmp_path / "out")),
        pipeline=PipelineConfig(chunk_size=1024, num_workers=2),
        attribution=AttributionConfig(),
        data_sources=DataSourcesConfig(),
    )


def test_resolve_pending_tasks_skips_existing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(PipelineProcessor, "_load_aoi", lambda self, path: None)
    monkeypatch.setattr(PipelineProcessor, "_discover_ccdc_years", lambda self: None)
    monkeypatch.setattr(PipelineProcessor, "_load_dem_metadata", lambda self: None)

    processor = PipelineProcessor(config=build_test_config(tmp_path))
    (tmp_path / "chunk_1.npz").write_bytes(b"done")

    tasks = [
        ChunkTask(chunk_idx=0, row_off=0, col_off=0, width=4, height=4),
        ChunkTask(chunk_idx=1, row_off=4, col_off=0, width=4, height=4),
    ]

    pending = processor._resolve_pending_tasks(tasks)

    assert [task.chunk_idx for task in pending] == [0]


def test_record_chunk_result_appends_serializable_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(PipelineProcessor, "_load_aoi", lambda self, path: None)
    monkeypatch.setattr(PipelineProcessor, "_discover_ccdc_years", lambda self: None)
    monkeypatch.setattr(PipelineProcessor, "_load_dem_metadata", lambda self: None)

    processor = PipelineProcessor(config=build_test_config(tmp_path))

    audit = ChunkAudit(
        chunk_idx=3,
        row_off=16,
        col_off=32,
        width=1024,
        height=1024,
        valid_pixels=99,
        years=(2015, 2020),
        disturbance_count=8,
        disturbance_class_count=6,
        dist_to_road_non_null=99,
        dist_to_river_non_null=99,
        resident_non_null=80,
        rainfall_non_null=1200,
        cache_file=str(tmp_path / "chunk_3.npz"),
    )
    result = ChunkResult(chunk_idx=3, saved=True, cache_file=audit.cache_file, audit=audit)

    processor._record_chunk_result(result)

    assert processor.audit_records == [asdict(audit)]


def test_process_all_chunks_uses_execute_chunk_tasks_when_worker_count_above_one(tmp_path, monkeypatch):
    monkeypatch.setattr(PipelineProcessor, "_load_aoi", lambda self, path: None)
    monkeypatch.setattr(PipelineProcessor, "_discover_ccdc_years", lambda self: None)
    monkeypatch.setattr(PipelineProcessor, "_load_dem_metadata", lambda self: None)

    processor = PipelineProcessor(config=build_test_config(tmp_path))
    processor.river_loader = type("RiverLoader", (), {"set_aoi": lambda self, aoi: None, "load": lambda self: None})()

    tasks = [
        ChunkTask(chunk_idx=0, row_off=0, col_off=0, width=4, height=4),
        ChunkTask(chunk_idx=1, row_off=4, col_off=0, width=4, height=4),
    ]
    expected_results = [ChunkResult(chunk_idx=1, saved=True), ChunkResult(chunk_idx=0, saved=True)]
    observed = {}

    class FakeDataset:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("inducing_factor.pipeline.processor.rasterio.open", lambda path: FakeDataset())
    monkeypatch.setattr(processor, "_build_chunk_tasks", lambda src_dem: tasks)
    monkeypatch.setattr(processor, "_resolve_pending_tasks", lambda built_tasks: built_tasks)
    monkeypatch.setattr(processor, "_resolve_worker_count", lambda: 2)

    def fake_execute_chunk_tasks(pending_tasks, **kwargs):
        observed["tasks"] = list(pending_tasks)
        observed["worker_count"] = kwargs["worker_count"]
        observed["submit_fn"] = kwargs["submit_fn"]
        observed["initargs"] = kwargs.get("initargs", ())
        return iter(expected_results)

    monkeypatch.setattr("inducing_factor.pipeline.processor.execute_chunk_tasks", fake_execute_chunk_tasks)

    processor.process_all_chunks()

    assert observed["tasks"] == tasks
    assert observed["worker_count"] == 2
    assert observed["submit_fn"].__name__ == "_process_chunk_task"


@pytest.mark.parametrize("worker_count", [0, 1])
def test_process_all_chunks_falls_back_to_inline_processing_for_zero_or_one_worker(tmp_path, monkeypatch, worker_count):
    monkeypatch.setattr(PipelineProcessor, "_load_aoi", lambda self, path: None)
    monkeypatch.setattr(PipelineProcessor, "_discover_ccdc_years", lambda self: None)
    monkeypatch.setattr(PipelineProcessor, "_load_dem_metadata", lambda self: None)

    processor = PipelineProcessor(config=build_test_config(tmp_path))
    processor.river_loader = type("RiverLoader", (), {"set_aoi": lambda self, aoi: None, "load": lambda self: None})()

    tasks = [
        ChunkTask(chunk_idx=0, row_off=0, col_off=0, width=4, height=4),
        ChunkTask(chunk_idx=1, row_off=4, col_off=0, width=4, height=4),
    ]
    processed = []

    class FakeDataset:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("inducing_factor.pipeline.processor.rasterio.open", lambda path: FakeDataset())
    monkeypatch.setattr(processor, "_build_chunk_tasks", lambda src_dem: tasks)
    monkeypatch.setattr(processor, "_resolve_pending_tasks", lambda built_tasks: built_tasks)
    monkeypatch.setattr(processor, "_resolve_worker_count", lambda: worker_count)
    monkeypatch.setattr(
        "inducing_factor.pipeline.processor.execute_chunk_tasks",
        lambda *args, **kwargs: pytest.fail("execute_chunk_tasks should not be used for inline fallback"),
    )

    def fake_process_single_chunk(task):
        processed.append(task.chunk_idx)
        return ChunkResult(chunk_idx=task.chunk_idx, saved=True)

    monkeypatch.setattr(processor, "_process_single_chunk", fake_process_single_chunk)

    processor.process_all_chunks()

    assert processed == [0, 1]

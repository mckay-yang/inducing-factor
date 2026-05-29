import os

from inducing_factor.config.loader import load_config
from inducing_factor.pipeline.execution import ChunkTask, execute_chunk_tasks, resolve_num_workers


def _identity(task: ChunkTask) -> int:
    return task.chunk_idx


def _chunk_pid(task: ChunkTask) -> tuple[int, int]:
    return task.chunk_idx, os.getpid()


def test_load_config_supports_parallel_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
pipeline:
  chunk_size: 1024
  num_workers: 0
  max_auto_workers: 4
  memory_target_fraction: 0.7
  estimated_worker_gb: 3.0
attribution:
  run_num_workers: 0
  run_max_auto_workers: 4
  run_memory_target_fraction: 0.7
  run_estimated_worker_gb: 1.5
  predict_num_workers: 0
  predict_max_auto_workers: 4
  predict_memory_target_fraction: 0.7
  predict_estimated_worker_gb: 1.0
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.pipeline.chunk_size == 1024
    assert config.pipeline.num_workers == 0
    assert config.pipeline.max_auto_workers == 4
    assert config.pipeline.memory_target_fraction == 0.7
    assert config.pipeline.estimated_worker_gb == 3.0
    assert config.attribution.run_num_workers == 0
    assert config.attribution.predict_num_workers == 0


def test_resolve_num_workers_auto_respects_caps(monkeypatch):
    monkeypatch.setattr("inducing_factor.pipeline.execution.os.cpu_count", lambda: 16)

    class Memory:
        available = 24 * 1024**3

    monkeypatch.setattr("inducing_factor.pipeline.execution.PSUTIL_AVAILABLE", True)
    monkeypatch.setattr("inducing_factor.pipeline.execution.psutil.virtual_memory", lambda: Memory())

    workers = resolve_num_workers(
        num_workers=0,
        max_auto_workers=4,
        memory_target_fraction=0.7,
        estimated_worker_gb=3.0,
    )

    assert workers == 4


def test_resolve_num_workers_allows_zero_when_memory_is_too_small(monkeypatch):
    monkeypatch.setattr("inducing_factor.pipeline.execution.os.cpu_count", lambda: 16)

    class Memory:
        available = 2 * 1024**3

    monkeypatch.setattr("inducing_factor.pipeline.execution.PSUTIL_AVAILABLE", True)
    monkeypatch.setattr("inducing_factor.pipeline.execution.psutil.virtual_memory", lambda: Memory())

    workers = resolve_num_workers(
        num_workers=0,
        max_auto_workers=4,
        memory_target_fraction=0.25,
        estimated_worker_gb=3.0,
    )

    assert workers == 0


def test_execute_chunk_tasks_runs_inline_when_worker_count_is_one():
    tasks = [
        ChunkTask(chunk_idx=2, row_off=0, col_off=0, width=4, height=4),
        ChunkTask(chunk_idx=5, row_off=4, col_off=4, width=4, height=4),
    ]

    results = list(execute_chunk_tasks(tasks, worker_count=1, submit_fn=_identity))

    assert results == [2, 5]


def test_execute_chunk_tasks_runs_with_parallel_backend(monkeypatch):
    tasks = [
        ChunkTask(chunk_idx=2, row_off=0, col_off=0, width=4, height=4),
        ChunkTask(chunk_idx=5, row_off=4, col_off=4, width=4, height=4),
    ]
    seen = {}

    class FakeParallel:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def __call__(self, jobs):
            kwargs = seen["kwargs"]
            initializer = kwargs.get("initializer")
            initargs = kwargs.get("initargs", ())
            if initializer is not None:
                initializer(*initargs)
            output = []
            for fn, args, kw in jobs:
                output.append(fn(*args, **kw))
            return output

    init_state = {"value": 0}

    def init(value):
        init_state["value"] = value

    def run(task: ChunkTask) -> int:
        return task.chunk_idx + init_state["value"]

    results = list(
        execute_chunk_tasks(
            tasks,
            worker_count=2,
            submit_fn=run,
            initializer=init,
            initargs=(10,),
            parallel_cls=FakeParallel,
        )
    )

    assert results == [12, 15]
    assert seen["kwargs"]["n_jobs"] == 2
    assert seen["kwargs"]["backend"] == "loky"


def test_execute_chunk_tasks_uses_real_process_pool_with_pickle_safe_function():
    tasks = [
        ChunkTask(chunk_idx=2, row_off=0, col_off=0, width=4, height=4),
        ChunkTask(chunk_idx=5, row_off=4, col_off=4, width=4, height=4),
    ]

    results = list(execute_chunk_tasks(tasks, worker_count=2, submit_fn=_chunk_pid))

    assert sorted(chunk_idx for chunk_idx, _ in results) == [2, 5]
    assert any(pid != os.getpid() for _, pid in results)

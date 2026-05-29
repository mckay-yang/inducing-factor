import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = SimpleNamespace(virtual_memory=None)
    PSUTIL_AVAILABLE = False


@dataclass(frozen=True)
class ChunkTask:
    chunk_idx: int
    row_off: int
    col_off: int
    width: int
    height: int


@dataclass(frozen=True)
class ChunkAudit:
    chunk_idx: int
    row_off: int = 0
    col_off: int = 0
    width: int = 0
    height: int = 0
    valid_pixels: int = 0
    years: tuple[int, int] | tuple[()] = ()
    disturbance_count: int = 0
    disturbance_class_count: int = 0
    dist_to_road_non_null: int = 0
    dist_to_river_non_null: int = 0
    resident_non_null: int = 0
    rainfall_non_null: int = 0
    cache_file: str | None = None
    worker_id: int | None = None


@dataclass(frozen=True)
class ChunkResult:
    chunk_idx: int
    saved: bool = False
    cache_file: str | None = None
    audit: ChunkAudit | None = None


def resolve_num_workers(
    num_workers: int,
    max_auto_workers: int,
    memory_target_fraction: float,
    estimated_worker_gb: float,
) -> int:
    if num_workers > 0:
        return num_workers

    cpu_total = os.cpu_count() or 1
    cpu_limit = max(1, cpu_total - 1)
    worker_limit = max(1, max_auto_workers)

    if not PSUTIL_AVAILABLE or estimated_worker_gb <= 0:
        return min(cpu_limit, worker_limit)

    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    memory_limit = int((available_memory_gb * memory_target_fraction) // estimated_worker_gb)
    return max(0, min(cpu_limit, worker_limit, memory_limit))


def chunk_cache_path(cache_dir: str | Path, chunk_idx: int) -> Path:
    return Path(cache_dir) / f"chunk_{chunk_idx}.npz"


def atomic_write_npz(output_path: str | Path, **arrays: Any) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, output)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return output


def execute_chunk_tasks(
    tasks: Iterable[ChunkTask],
    worker_count: int,
    submit_fn: Callable[[ChunkTask], Any],
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
    parallel_cls: type[Parallel] = Parallel,
) -> Iterator[Any]:
    task_list = list(tasks)
    if not task_list:
        return iter(())

    logger.info("execute_chunk_tasks: %d tasks, %d workers", len(task_list), worker_count)
    def _run() -> Iterator[Any]:
        if worker_count <= 1:
            logger.debug("execute_chunk_tasks: sequential execution")
            if initializer is not None:
                initializer(*initargs)
            for task in task_list:
                yield submit_fn(task)
            return

        if initializer is not None:
            initializer(*initargs)

        logger.info("execute_chunk_tasks: starting parallel execution with %d workers", worker_count)
        import multiprocessing as mp
        fork_ctx = mp.get_context("fork")
        pool = fork_ctx.Pool(processes=worker_count)
        try:
            results = pool.imap_unordered(submit_fn, task_list)
            for item in results:
                yield item
        finally:
            pool.close()
            pool.join()
        logger.info("execute_chunk_tasks: parallel execution completed")

    return _run()

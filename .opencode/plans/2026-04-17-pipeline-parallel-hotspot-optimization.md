# Pipeline Parallelism And Disturbance Hotspot Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up `python -m inducing_factor.cli pipeline --config configs/production_fresh.yaml` by reducing chunk size, adding chunk-level multiprocessing, and introducing conservative disturbance-hotspot-aware thinning for cold chunks.

**Architecture:** Keep `PipelineProcessor` as the public entrypoint, but extract parallel execution helpers into `inducing_factor/pipeline/execution.py` and hotspot classification helpers into `inducing_factor/pipeline/hotspots.py`. The parent process will enumerate chunk tasks, decide worker count, skip already-saved chunks, and aggregate progress plus audit output; worker processes will own their own loader state and write each chunk atomically. Hotspot optimization stays conservative: disturbed chunks keep full fidelity, and only chunks with zero disturbance history are eligible for early background thinning.

**Tech Stack:** Python 3.11, `concurrent.futures.ProcessPoolExecutor`, `multiprocessing`, `numpy`, `pandas`, `rasterio`, `geopandas`, `shapely`, `pytest`

---

## Scope Check

This is one coherent plan, not multiple unrelated plans. Chunk-size tuning, multiprocessing, and disturbance-hotspot optimization all change the same execution path in `inducing_factor/pipeline/processor.py`, and they should be designed and verified together. This plan explicitly excludes precomputing yearly road/railway/waterway distance rasters; that is a larger architectural change and should be a separate follow-up plan after this pass is stable.

## File Structure

- `inducing_factor/config/models.py`
  Add pipeline-level tuning knobs for automatic worker sizing and hotspot behavior.
- `inducing_factor/pipeline/execution.py`
  Hold pure execution primitives: chunk task/result dataclasses, worker-count resolution, chunk cache path helpers, atomic `.npz` writes, and executor orchestration.
- `inducing_factor/pipeline/hotspots.py`
  Hold pure disturbance-hotspot classification and cold-chunk point-selection helpers.
- `inducing_factor/pipeline/processor.py`
  Orchestrate chunk task generation, skip logic, parent/worker separation, hotspot-aware chunk processing, and audit aggregation.
- `tests/unit/test_pipeline_execution.py`
  Verify config loading, worker auto-sizing, and executor helpers.
- `tests/unit/test_pipeline_hotspots.py`
  Verify hotspot classification and deterministic cold-chunk sampling.
- `tests/unit/test_pipeline_processor.py`
  Verify processor orchestration helpers such as pending-task resolution and audit aggregation.
- `tests/unit/test_production_config.py`
  Verify production YAML defaults are aligned with the new pipeline mode.
- `configs/production.yaml`
  Set production defaults for smaller chunks and safe auto-sized multiprocessing.
- `configs/production_fresh.yaml`
  Move the active production run to a fresh cache directory because the chunk grid and write pattern change.

### Task 1: Add Pipeline Execution Settings And Pure Execution Helpers

**Files:**
- Create: `tests/unit/test_pipeline_execution.py`
- Create: `inducing_factor/pipeline/execution.py`
- Modify: `inducing_factor/config/models.py`

- [ ] **Step 1: Write the failing test**

```python
from concurrent.futures import Future

from inducing_factor.config.loader import load_config
from inducing_factor.pipeline.execution import ChunkTask, execute_chunk_tasks, resolve_num_workers


class InlineExecutor:
    def __init__(self, max_workers, initializer=None, initargs=(), mp_context=None):
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        if self.initializer is not None:
            self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def _identity(task: ChunkTask) -> int:
    return task.chunk_idx


def test_load_config_supports_parallel_pipeline_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
pipeline:
  chunk_size: 1024
  num_workers: 0
  max_auto_workers: 4
  memory_target_fraction: 0.70
  estimated_worker_gb: 3.0
  hotspot_mode: cold_sampling
  cold_chunk_background_rate: 0.25
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.pipeline.chunk_size == 1024
    assert config.pipeline.num_workers == 0
    assert config.pipeline.max_auto_workers == 4
    assert config.pipeline.memory_target_fraction == 0.70
    assert config.pipeline.estimated_worker_gb == 3.0
    assert config.pipeline.hotspot_mode == "cold_sampling"
    assert config.pipeline.cold_chunk_background_rate == 0.25


def test_resolve_num_workers_auto_respects_cpu_memory_and_cap(monkeypatch):
    monkeypatch.setattr("inducing_factor.pipeline.execution.os.cpu_count", lambda: 16)

    class Memory:
        available = 24 * 1024**3

    monkeypatch.setattr("inducing_factor.pipeline.execution.PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(
        "inducing_factor.pipeline.execution.psutil.virtual_memory",
        lambda: Memory(),
    )

    workers = resolve_num_workers(
        num_workers=0,
        max_auto_workers=4,
        memory_target_fraction=0.70,
        estimated_worker_gb=3.0,
    )

    assert workers == 4


def test_execute_chunk_tasks_runs_every_task():
    tasks = [
        ChunkTask(chunk_idx=2, row_off=0, col_off=0, width=4, height=4),
        ChunkTask(chunk_idx=5, row_off=4, col_off=4, width=4, height=4),
    ]

    results = list(
        execute_chunk_tasks(
            tasks,
            worker_count=2,
            submit_fn=_identity,
            executor_cls=InlineExecutor,
        )
    )

    assert sorted(results) == [2, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_execution.py -q`

Expected: FAIL with import and config errors such as:
- `ModuleNotFoundError: No module named 'inducing_factor.pipeline.execution'`
- `TypeError: PipelineConfig.__init__() got an unexpected keyword argument 'num_workers'`

- [ ] **Step 3: Write minimal implementation**

Update `inducing_factor/config/models.py`:

```python
@dataclass
class PipelineConfig:
    start_year: int = 1990
    end_year: int = 2020
    chunk_size: int = 1024
    num_workers: int = 1
    max_auto_workers: int = 4
    memory_target_fraction: float = 0.70
    estimated_worker_gb: float = 3.0
    hotspot_mode: str = "off"
    cold_chunk_background_rate: float = 1.0
```

Create `inducing_factor/pipeline/execution.py`:

```python
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
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
    row_off: int
    col_off: int
    width: int
    height: int
    valid_pixels: int
    years: tuple[int, int]
    disturbance_count: int
    disturbance_class_count: int
    dist_to_road_non_null: int
    dist_to_river_non_null: int
    resident_non_null: int
    rainfall_non_null: int
    cache_file: str


@dataclass(frozen=True)
class ChunkResult:
    chunk_idx: int
    saved: bool
    cache_file: str | None
    audit: ChunkAudit | None


def resolve_num_workers(
    *,
    num_workers: int,
    max_auto_workers: int,
    memory_target_fraction: float,
    estimated_worker_gb: float,
) -> int:
    if num_workers > 0:
        return num_workers

    cpu_cap = os.cpu_count() or 1
    if not PSUTIL_AVAILABLE:
        return max(1, min(cpu_cap, max_auto_workers))

    available_gb = psutil.virtual_memory().available / 1024**3
    memory_cap = max(1, int((available_gb * memory_target_fraction) // estimated_worker_gb))
    return max(1, min(cpu_cap, max_auto_workers, memory_cap))


def chunk_cache_path(cache_dir: str, chunk_idx: int) -> str:
    return os.path.join(cache_dir, f"chunk_{chunk_idx}.npz")


def atomic_write_npz(cache_dir: str, chunk_idx: int, payload: dict[str, np.ndarray]) -> str:
    final_path = chunk_cache_path(cache_dir, chunk_idx)
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(tmp_path, final_path)
    return final_path


def execute_chunk_tasks(
    tasks,
    *,
    worker_count: int,
    submit_fn,
    executor_cls=ProcessPoolExecutor,
    initializer=None,
    initargs=(),
    mp_context=None,
):
    if worker_count <= 1:
        for task in tasks:
            yield submit_fn(task)
        return

    with executor_cls(
        max_workers=worker_count,
        initializer=initializer,
        initargs=initargs,
        mp_context=mp_context,
    ) as executor:
        futures = [executor.submit(submit_fn, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_execution.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_pipeline_execution.py inducing_factor/pipeline/execution.py inducing_factor/config/models.py
git commit -m "feat: add pipeline execution settings and helpers"
```

### Task 2: Add Disturbance Hotspot Classification Helpers

**Files:**
- Create: `tests/unit/test_pipeline_hotspots.py`
- Create: `inducing_factor/pipeline/hotspots.py`

- [ ] **Step 1: Write the failing test**

```python
from inducing_factor.pipeline.hotspots import classify_chunk_hotspot, select_chunk_point_indices


def test_classify_chunk_hotspot_marks_disturbance_free_chunk_as_cold():
    hotspot = classify_chunk_hotspot(7, {2015: 0, 2016: 0, 2017: 0})

    assert hotspot.chunk_idx == 7
    assert hotspot.level == "cold"
    assert hotspot.total_disturbance_pixels == 0
    assert hotspot.active_years == ()


def test_select_chunk_point_indices_keeps_hot_chunks_full_density():
    hotspot = classify_chunk_hotspot(3, {2015: 8, 2016: 6})
    selected = select_chunk_point_indices(
        hotspot,
        point_count=20,
        sample_rate=0.25,
        random_state=42,
    )

    assert selected.tolist() == list(range(20))


def test_select_chunk_point_indices_downsamples_cold_chunks_deterministically():
    hotspot = classify_chunk_hotspot(9, {2015: 0, 2016: 0, 2017: 0})

    selected_a = select_chunk_point_indices(
        hotspot,
        point_count=20,
        sample_rate=0.25,
        random_state=42,
    )
    selected_b = select_chunk_point_indices(
        hotspot,
        point_count=20,
        sample_rate=0.25,
        random_state=42,
    )

    assert len(selected_a) == 5
    assert selected_a.tolist() == selected_b.tolist()
    assert selected_a.tolist() == sorted(selected_a.tolist())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_hotspots.py -q`

Expected: FAIL with:
- `ModuleNotFoundError: No module named 'inducing_factor.pipeline.hotspots'`

- [ ] **Step 3: Write minimal implementation**

Create `inducing_factor/pipeline/hotspots.py`:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkHotspot:
    chunk_idx: int
    level: str
    total_disturbance_pixels: int
    active_years: tuple[int, ...]


def classify_chunk_hotspot(chunk_idx: int, yearly_disturbance_counts: dict[int, int]) -> ChunkHotspot:
    active_years = tuple(sorted(year for year, count in yearly_disturbance_counts.items() if count > 0))
    total_disturbance_pixels = int(sum(yearly_disturbance_counts.values()))

    if total_disturbance_pixels == 0:
        level = "cold"
    elif total_disturbance_pixels >= 1000 or len(active_years) >= 2:
        level = "hot"
    else:
        level = "warm"

    return ChunkHotspot(
        chunk_idx=chunk_idx,
        level=level,
        total_disturbance_pixels=total_disturbance_pixels,
        active_years=active_years,
    )


def select_chunk_point_indices(
    hotspot: ChunkHotspot,
    *,
    point_count: int,
    sample_rate: float,
    random_state: int,
) -> np.ndarray:
    if hotspot.level != "cold" or sample_rate >= 1.0 or point_count == 0:
        return np.arange(point_count)

    keep_count = max(1, int(round(point_count * sample_rate)))
    rng = np.random.default_rng(random_state)
    selected = rng.choice(np.arange(point_count), size=keep_count, replace=False)
    return np.sort(selected)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_hotspots.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_pipeline_hotspots.py inducing_factor/pipeline/hotspots.py
git commit -m "feat: add disturbance hotspot helpers for pipeline chunking"
```

### Task 3: Refactor PipelineProcessor For Parent/Worker Chunk Orchestration

**Files:**
- Create: `tests/unit/test_pipeline_processor.py`
- Modify: `inducing_factor/pipeline/processor.py`

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import asdict

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_processor.py -q`

Expected: FAIL with:
- `AttributeError: 'PipelineProcessor' object has no attribute '_resolve_pending_tasks'`
- `AttributeError: 'PipelineProcessor' object has no attribute '_record_chunk_result'`

- [ ] **Step 3: Write minimal implementation**

Update `inducing_factor/pipeline/processor.py` with the new imports and worker entrypoints:

```python
import multiprocessing as mp
from dataclasses import asdict

from inducing_factor.pipeline.execution import (
    ChunkAudit,
    ChunkResult,
    ChunkTask,
    atomic_write_npz,
    chunk_cache_path,
    execute_chunk_tasks,
    resolve_num_workers,
)
from inducing_factor.pipeline.hotspots import classify_chunk_hotspot, select_chunk_point_indices


_WORKER: "PipelineProcessor | None" = None


def _init_chunk_worker(config: AppConfig, cache_dir: str):
    global _WORKER
    _WORKER = PipelineProcessor(config=config, cache_dir=cache_dir)


def _process_chunk_task(task: ChunkTask) -> ChunkResult:
    if _WORKER is None:
        raise RuntimeError("Chunk worker was not initialized")
    return _WORKER._process_single_chunk(task)
```

Add the helper methods inside `PipelineProcessor`:

```python
def _load_dem_metadata(self):
    with rasterio.open(self.dem_loader.file_path) as src:
        self.dem_crs = src.crs
        self.dem_transform = src.transform
        self.dem_width = src.width
        self.dem_height = src.height


def _build_chunk_tasks(self, src_dem):
    tasks = []
    for chunk_idx, (window, row_off, col_off) in enumerate(self._compute_chunks_intersecting_aoi(src_dem)):
        tasks.append(
            ChunkTask(
                chunk_idx=chunk_idx,
                row_off=row_off,
                col_off=col_off,
                width=int(window.width),
                height=int(window.height),
            )
        )
    return tasks


def _resolve_worker_count(self) -> int:
    return resolve_num_workers(
        num_workers=self.config.pipeline.num_workers,
        max_auto_workers=self.config.pipeline.max_auto_workers,
        memory_target_fraction=self.config.pipeline.memory_target_fraction,
        estimated_worker_gb=self.config.pipeline.estimated_worker_gb,
    )


def _resolve_pending_tasks(self, tasks: list[ChunkTask]) -> list[ChunkTask]:
    pending = []
    for task in tasks:
        cache_file = chunk_cache_path(self.cache_dir, task.chunk_idx)
        if os.path.exists(cache_file):
            logger.info("Skipping already processed chunk %s", task.chunk_idx)
            continue
        pending.append(task)
    return pending


def _record_chunk_result(self, result: ChunkResult):
    if result.audit is not None:
        self.audit_records.append(asdict(result.audit))
```

Update `__init__` so it caches DEM metadata once:

```python
self._load_aoi(ds.aoi_shp)
self._discover_ccdc_years()
self._load_dem_metadata()
```

Replace `process_all_chunks` with parent/worker orchestration:

```python
def process_all_chunks(self):
    logger.info("Starting pipeline processing. Output cache: %s", self.cache_dir)
    if self.aoi is not None:
        self.river_loader.set_aoi(self.aoi)

    with rasterio.open(self.dem_loader.file_path) as src_dem:
        tasks = self._build_chunk_tasks(src_dem)

    pending_tasks = self._resolve_pending_tasks(tasks)
    progress = ProgressTracker(logger, "pipeline", log_every=1)
    progress.start_stage(1, 1, "process chunks", total=len(tasks))

    skipped = len(tasks) - len(pending_tasks)
    if skipped:
        progress.advance(skipped, {"saved": len(self.audit_records), "skipped": skipped})

    worker_count = self._resolve_worker_count()
    logger.info("Pipeline worker count resolved to %s", worker_count)

    if worker_count <= 1:
        results_iter = (self._process_single_chunk(task) for task in pending_tasks)
    else:
        mp_context = mp.get_context("spawn")
        results_iter = execute_chunk_tasks(
            pending_tasks,
            worker_count=worker_count,
            submit_fn=_process_chunk_task,
            initializer=_init_chunk_worker,
            initargs=(self.config, self.cache_dir),
            mp_context=mp_context,
        )

    for result in results_iter:
        self._record_chunk_result(result)
        progress.advance(1, {"saved": len(self.audit_records), "chunk_idx": result.chunk_idx})

    progress.finish_stage({"saved": len(self.audit_records)})
    self._save_audit_report()
```

Change `_process_single_chunk` to accept a `ChunkTask`, classify the chunk before expensive yearly feature generation, apply cold-chunk thinning only when configured, and return a `ChunkResult`:

```python
def _process_single_chunk(self, task: ChunkTask) -> ChunkResult:
    window = Window(task.col_off, task.row_off, task.width, task.height)
    dem_data, (res_x, res_y), dem_nodata = self.dem_loader.load(window)
    if np.all(dem_data == dem_nodata) or np.all(np.isnan(dem_data)):
        return ChunkResult(chunk_idx=task.chunk_idx, saved=False, cache_file=None, audit=None)

    slope_data, aspect_data = self.dem_loader.calculate_slope_aspect(dem_data, res_x, res_y, nodata=dem_nodata)
    valid_mask = (dem_data != dem_nodata) & (~np.isnan(dem_data))
    chunk_transform = window_transform(window, self.dem_transform)

    if self.aoi is not None and self.dem_crs is not None:
        aoi_proj = self.aoi.to_crs(self.dem_crs)
        geom = [geom for geom in aoi_proj.geometry if geom is not None and not geom.is_empty]
        aoi_mask = geometry_mask(geom, transform=chunk_transform, out_shape=dem_data.shape, invert=True)
        valid_mask = valid_mask & aoi_mask

    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        return ChunkResult(chunk_idx=task.chunk_idx, saved=False, cache_file=None, audit=None)

    x_coords, y_coords = rasterio.transform.xy(chunk_transform, rows, cols)
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)

    yearly_disturbance_values = {}
    yearly_disturbance_months = {}
    yearly_disturbance_counts = {}

    for year in range(self.start_year, self.end_year + 1):
        ccdc_res = self.ccdc_loader.load(
            year,
            self.dem_crs,
            self.dem_transform,
            self.dem_width,
            self.dem_height,
            window=window,
        )

        if ccdc_res is None:
            yearly_disturbance_values[year] = np.zeros(len(rows), dtype=float)
            yearly_disturbance_months[year] = np.full(len(rows), np.nan, dtype=float)
            yearly_disturbance_counts[year] = 0
            continue

        ccdc_data, ccdc_nodata = ccdc_res
        disturbance_vals = ccdc_data[rows, cols]
        if ccdc_nodata is not None:
            disturbance_mask = (disturbance_vals != ccdc_nodata) & (~np.isnan(disturbance_vals))
        else:
            disturbance_mask = ~np.isnan(disturbance_vals)

        disturbance_years = np.floor(np.where(disturbance_mask, disturbance_vals, np.nan)).astype(float)
        year_match = disturbance_years == float(year)
        disturbance_vals = np.where(disturbance_mask & year_match, disturbance_vals, 0.0)

        yearly_disturbance_values[year] = disturbance_vals
        yearly_disturbance_months[year] = _decimal_year_to_month(
            disturbance_vals,
            np.full(len(rows), year, dtype=float),
        )
        yearly_disturbance_counts[year] = int((pd.to_numeric(disturbance_vals, errors="coerce") > 0).sum())

    hotspot = classify_chunk_hotspot(task.chunk_idx, yearly_disturbance_counts)
    sample_rate = self.config.pipeline.cold_chunk_background_rate if self.config.pipeline.hotspot_mode == "cold_sampling" else 1.0
    selector = select_chunk_point_indices(
        hotspot,
        point_count=len(rows),
        sample_rate=sample_rate,
        random_state=self.config.attribution.random_state + task.chunk_idx,
    )

    rows = rows[selector]
    cols = cols[selector]
    x_coords = x_coords[selector]
    y_coords = y_coords[selector]
    if len(rows) == 0:
        return ChunkResult(chunk_idx=task.chunk_idx, saved=False, cache_file=None, audit=None)

    dist_river = self.river_loader.get_distance(x_coords, y_coords, self.dem_crs, max_dist=10000)
    point_ids = np.arange(len(rows)) + (task.chunk_idx * 10_000_000)

    df_static = pd.DataFrame(
        {
            "point_id": point_ids,
            "raster_row": rows + task.row_off,
            "raster_col": cols + task.col_off,
            "x": x_coords,
            "y": y_coords,
            "elevation": dem_data[rows, cols],
            "slope": slope_data[rows, cols],
            "aspect": aspect_data[rows, cols],
            "dist_to_river": dist_river,
        }
    )

    self.rainfall_loader.clear_state()
    self.temp_loader.clear_state()
    self.landuse_loader.clear_state()
    self.resident_loader.clear_state()
    chunk_records = []

    for year in range(self.start_year, self.end_year + 1):
        df_year = df_static.copy()
        df_year["year"] = year
        temp_data = self.temp_loader.load(year, self.dem_crs, self.dem_transform, self.dem_width, self.dem_height, window=window)
        lu_data = self.landuse_loader.load(year, self.dem_crs, self.dem_transform, self.dem_width, self.dem_height, window=window)
        res_data = self.resident_loader.load(year, self.dem_crs, self.dem_transform, self.dem_width, self.dem_height, window=window)

        df_year["temperature"] = temp_data[rows, cols] if temp_data is not None else np.nan
        df_year["landuse"] = lu_data[rows, cols] if lu_data is not None else np.nan
        if res_data is not None:
            resident_vals = res_data[rows, cols]
            resident_vals = np.where(resident_vals == self.resident_nodata, np.nan, resident_vals)
            df_year["resident"] = resident_vals
        else:
            df_year["resident"] = 0

        if year not in self.road_loaders_by_year:
            road_path = os.path.join(self.road_shp_dir, self.road_file_pattern.format(year=year))
            if os.path.exists(road_path):
                loader = VectorLoader(road_path)
                loader.set_aoi(self.aoi)
                self.road_loaders_by_year[year] = loader
            else:
                self.road_loaders_by_year[year] = None

        if year not in self.railway_loaders_by_year:
            railway_path = os.path.join(self.road_shp_dir, self.railway_file_pattern.format(year=year))
            if os.path.exists(railway_path):
                loader = VectorLoader(railway_path)
                loader.set_aoi(self.aoi)
                self.railway_loaders_by_year[year] = loader
            else:
                self.railway_loaders_by_year[year] = None

        if year not in self.waterway_loaders_by_year:
            waterway_path = os.path.join(self.road_shp_dir, self.waterway_file_pattern.format(year=year))
            if os.path.exists(waterway_path):
                loader = VectorLoader(waterway_path)
                loader.set_aoi(self.aoi)
                self.waterway_loaders_by_year[year] = loader
            else:
                self.waterway_loaders_by_year[year] = None

        dist_road = self.road_loaders_by_year[year].get_distance(x_coords, y_coords, self.dem_crs, max_dist=10000) if self.road_loaders_by_year[year] is not None else np.full(len(x_coords), np.nan)
        dist_railway = self.railway_loaders_by_year[year].get_distance(x_coords, y_coords, self.dem_crs, max_dist=10000) if self.railway_loaders_by_year[year] is not None else np.full(len(x_coords), np.nan)
        dist_waterway = self.waterway_loaders_by_year[year].get_distance(x_coords, y_coords, self.dem_crs, max_dist=10000) if self.waterway_loaders_by_year[year] is not None else np.full(len(x_coords), np.nan)

        df_year["dist_to_road"] = dist_road
        df_year["dist_to_railway"] = dist_railway
        df_year["dist_to_waterway"] = dist_waterway

        for month in range(1, 13):
            rain_data = self.rainfall_loader.load_month(year, month, self.dem_crs, self.dem_transform, self.dem_width, self.dem_height, window=window)
            col = f"rainfall_month_{month:02d}"
            df_year[col] = rain_data[rows, cols] if rain_data is not None else np.nan

        df_year["disturbance_time_value"] = yearly_disturbance_values[year][selector]
        df_year["disturbance_type"] = yearly_disturbance_values[year][selector]
        df_year["disturbance_month"] = yearly_disturbance_months[year][selector]

        class_res = self.disturbance_class_loader.load(year, self.dem_crs, self.dem_transform, self.dem_width, self.dem_height, window=window)
        if class_res is not None:
            class_data, class_nodata = class_res
            class_vals = class_data[rows, cols]
            if class_nodata is not None:
                class_vals = np.where(class_vals == class_nodata, 0, class_vals)
            class_vals = np.where(np.isin(class_vals, [1, 2, 3, 4]), class_vals, 0)
            df_year["disturbance_class"] = class_vals.astype(np.int16)
        else:
            df_year["disturbance_class"] = 0

        chunk_records.append(df_year)

    if not chunk_records:
        return ChunkResult(chunk_idx=task.chunk_idx, saved=False, cache_file=None, audit=None)

    df_chunk = pd.concat(chunk_records, ignore_index=True)
    df_chunk["point_id"] = pd.to_numeric(df_chunk["point_id"], downcast="integer")
    df_chunk["year"] = pd.to_numeric(df_chunk["year"], downcast="integer")

    payload = {col: df_chunk[col].values for col in df_chunk.columns}
    cache_file = atomic_write_npz(self.cache_dir, task.chunk_idx, payload)

    audit = ChunkAudit(
        chunk_idx=task.chunk_idx,
        row_off=task.row_off,
        col_off=task.col_off,
        width=task.width,
        height=task.height,
        valid_pixels=int(len(rows)),
        years=(self.start_year, self.end_year),
        disturbance_count=int((pd.to_numeric(df_chunk["disturbance_time_value"], errors="coerce") > 0).sum()),
        disturbance_class_count=int(pd.to_numeric(df_chunk["disturbance_class"], errors="coerce").isin([1, 2, 3, 4]).sum()),
        dist_to_road_non_null=int(pd.notna(df_chunk["dist_to_road"]).sum()),
        dist_to_river_non_null=int(np.isfinite(dist_river).sum()),
        resident_non_null=int(pd.notna(df_chunk["resident"]).sum()),
        rainfall_non_null=int(pd.notna(df_chunk.filter(like="rainfall_month_")).sum().sum()),
        cache_file=cache_file,
    )

    logger.info("Saved chunk %s with %s sampled pixels to %s", task.chunk_idx, len(rows), cache_file)
    return ChunkResult(chunk_idx=task.chunk_idx, saved=True, cache_file=cache_file, audit=audit)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_processor.py tests/unit/test_pipeline_execution.py tests/unit/test_pipeline_hotspots.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_pipeline_processor.py inducing_factor/pipeline/processor.py
git commit -m "feat: parallelize chunk processing with parent-worker orchestration"
```

### Task 4: Update Production Configs For Smaller Chunks And Safe Auto-Sized Workers

**Files:**
- Create: `tests/unit/test_production_config.py`
- Modify: `configs/production.yaml`
- Modify: `configs/production_fresh.yaml`

- [ ] **Step 1: Write the failing test**

```python
from inducing_factor.config.loader import load_config


def test_production_config_uses_safe_parallel_defaults():
    config = load_config("configs/production.yaml")

    assert config.pipeline.chunk_size == 1024
    assert config.pipeline.num_workers == 0
    assert config.pipeline.max_auto_workers == 4
    assert config.pipeline.memory_target_fraction == 0.70
    assert config.pipeline.estimated_worker_gb == 3.0
    assert config.pipeline.hotspot_mode == "cold_sampling"
    assert config.pipeline.cold_chunk_background_rate == 0.25


def test_production_fresh_config_uses_new_parallel_cache_dir():
    config = load_config("configs/production_fresh.yaml")

    assert config.paths.cache_dir == "data/.cache/chunks_aoi_v3_parallel"
    assert config.paths.out_dir == "data/.cache/patent_system_aoi_v3_parallel"
    assert config.pipeline.chunk_size == 1024
    assert config.pipeline.num_workers == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_production_config.py -q`

Expected: FAIL because the YAML files do not yet contain the new fields and paths.

- [ ] **Step 3: Write minimal implementation**

Update `configs/production.yaml`:

```yaml
paths:
  cache_dir: "data/.cache/chunks"
  out_dir: "data/.cache/patent_system"

pipeline:
  start_year: 1990
  end_year: 2020
  chunk_size: 1024
  num_workers: 0
  max_auto_workers: 4
  memory_target_fraction: 0.70
  estimated_worker_gb: 3.0
  hotspot_mode: "cold_sampling"
  cold_chunk_background_rate: 0.25
```

Update `configs/production_fresh.yaml`:

```yaml
paths:
  cache_dir: "data/.cache/chunks_aoi_v3_parallel"
  out_dir: "data/.cache/patent_system_aoi_v3_parallel"

pipeline:
  start_year: 1990
  end_year: 2020
  chunk_size: 1024
  num_workers: 0
  max_auto_workers: 4
  memory_target_fraction: 0.70
  estimated_worker_gb: 3.0
  hotspot_mode: "cold_sampling"
  cold_chunk_background_rate: 0.25
```

- [ ] **Step 4: Run the config and unit tests**

Run: `pytest tests/unit/test_production_config.py tests/unit/test_pipeline_execution.py tests/unit/test_pipeline_hotspots.py tests/unit/test_pipeline_processor.py -q`

Expected: PASS

- [ ] **Step 5: Run the existing regression test**

Run: `pytest tests/integration/test_synthetic_pipeline.py -q`

Expected: PASS

- [ ] **Step 6: Run a production smoke command on the real config**

Run: `python -m inducing_factor.cli pipeline --config configs/production_fresh.yaml`

Expected: startup logs include:
- `Starting pipeline processing. Output cache: data/.cache/chunks_aoi_v3_parallel`
- `Pipeline worker count resolved to`
- chunk progress logs under `[pipeline][stage 1/1]`

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_production_config.py configs/production.yaml configs/production_fresh.yaml
git commit -m "chore: tune production pipeline defaults for parallel chunk processing"
```

## Self-Review

- Spec coverage: the plan covers chunk-size reduction, automatic worker sizing, chunk-level multiprocessing, atomic writes, parent-side progress and audit aggregation, and conservative disturbance-hotspot thinning for cold chunks.
- Placeholder scan: there are no unfinished placeholder markers in the plan steps; the only literal `...` token is the Python variadic tuple annotation `tuple[int, ...]`.
- Type consistency: `ChunkTask`, `ChunkAudit`, `ChunkResult`, `ChunkHotspot`, `resolve_num_workers`, `execute_chunk_tasks`, `_resolve_pending_tasks`, and `_record_chunk_result` are named consistently across tests and implementation steps.

## Notes For The Implementer

- Do not reuse `data/.cache/chunks_aoi_v2` for the new run because the chunk grid and write strategy change.
- Keep `hotspot_mode: "off"` as the dataclass default so non-production configs remain behaviorally unchanged until explicitly enabled.
- Do not attempt process-per-core scaling in the first implementation. The `max_auto_workers: 4` cap is deliberate because the workload duplicates loader state per process.
- Do not change the attribution streaming path in `inducing_factor/core/attribution_system.py` in this plan.

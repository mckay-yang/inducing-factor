import json
import logging
import os
from dataclasses import asdict
from datetime import date, timedelta
from typing import Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import rowcol
from rasterio.windows import Window, transform as window_transform

from inducing_factor.config.models import AppConfig
from inducing_factor.data.loaders import CCDCLoader, ClimateLoader, DEMLoader, DisturbanceClassLoader, LandUseLoader, VectorLoader
from inducing_factor.pipeline.execution import ChunkAudit, ChunkResult, ChunkTask, atomic_write_npz, chunk_cache_path, execute_chunk_tasks, resolve_num_workers
from inducing_factor.utils.progress import ProgressTracker

logger = logging.getLogger("PipelineProcessor")

_WORKER: "PipelineProcessor | None" = None


def _process_chunk_task(task: ChunkTask) -> ChunkResult:
    if _WORKER is None:
        raise RuntimeError("Chunk worker was not initialized")
    return _WORKER._process_single_chunk(task)


def _decimal_year_to_month(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    months = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return months

    value_years = np.floor(values[valid]).astype(int)
    matching = value_years == years[valid].astype(int)
    if not np.any(matching):
        return months

    matched_values = values[valid][matching]
    matched_years = value_years[matching]
    fractions = np.clip(matched_values - matched_years, 0.0, 0.999999)
    decoded_months = []
    for year, fraction in zip(matched_years, fractions):
        days_in_year = 366 if date(year, 12, 31).toordinal() - date(year, 1, 1).toordinal() + 1 == 366 else 365
        day_of_year = min(days_in_year, max(1, int(round(fraction * days_in_year))))
        event_date = date(year, 1, 1) + timedelta(days=day_of_year - 1)
        decoded_months.append(event_date.month)
    months[np.where(valid)[0][matching]] = np.asarray(decoded_months, dtype=float)
    return months


class PipelineProcessor:
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        chunk_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        if config is None:
            from inducing_factor.config.loader import load_config

            config = load_config()
        self.config = config
        self.start_year = start_year if start_year is not None else config.pipeline.start_year
        self.end_year = end_year if end_year is not None else config.pipeline.end_year
        self.chunk_size = chunk_size if chunk_size is not None else config.pipeline.chunk_size
        self.cache_dir = cache_dir if cache_dir is not None else config.paths.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        ds = config.data_sources
        self.dem_loader = DEMLoader(ds.dem_tif)
        self.road_shp_dir = ds.road_shp_dir
        self.road_file_pattern = ds.road_file_pattern
        self.railway_file_pattern = ds.railway_file_pattern
        self.waterway_file_pattern = ds.waterway_file_pattern
        self.river_loader = VectorLoader(os.path.join(ds.river_shp_dir, "As_15s_Chimaps.shp"))
        self.road_loaders_by_year = {}
        self.railway_loaders_by_year = {}
        self.waterway_loaders_by_year = {}
        self.rainfall_loader = ClimateLoader(ds.rainfall_dir)
        self.temp_loader = ClimateLoader(ds.temperature_dir, file_pattern=ds.temperature_file_pattern)
        self.landuse_loader = LandUseLoader(
            ds.landuse_dir,
            is_temporal=True,
            file_pattern=ds.landuse_file_pattern,
            data_type="Land use",
        )
        self.resident_loader = LandUseLoader(
            ds.resident_dir,
            is_temporal=True,
            file_pattern=ds.resident_file_pattern,
            data_type="Resident",
        )
        self.ccdc_loader = CCDCLoader(ds.ccdc_dir, file_pattern=ds.ccdc_file_pattern)
        self.disturbance_class_loader = DisturbanceClassLoader(
            ds.disturbance_class_tif,
            start_year=ds.disturbance_class_start_year,
        )
        self.resident_nodata = ds.resident_nodata
        self.aoi = None
        self.audit_records: list[dict[str, object]] = []
        self.dem_crs = None
        self.dem_transform = None
        self.dem_width = 0
        self.dem_height = 0
        self._load_aoi(ds.aoi_shp)
        self._discover_ccdc_years()
        self._load_dem_metadata()

    def _load_aoi(self, aoi_shp: str):
        try:
            import geopandas as gpd

            read_errors = []
            aoi_gdf = None
            for encoding in (None, "utf-8", "gb18030", "gbk", "latin1"):
                try:
                    kwargs = {} if encoding is None else {"encoding": encoding}
                    aoi_gdf = gpd.read_file(aoi_shp, **kwargs)
                    break
                except Exception as exc:
                    read_errors.append(f"{encoding or 'default'}: {exc}")
            if aoi_gdf is None:
                raise ValueError("; ".join(read_errors))
            if aoi_gdf.crs is None:
                aoi_gdf.set_crs("EPSG:4326", inplace=True)
            self.aoi = aoi_gdf
            logger.info("AOI loaded: %s", aoi_shp)
        except ImportError:
            logger.warning("geopandas not installed; AOI clipping disabled")
        except Exception as e:
            logger.warning("Failed to load AOI %s: %s", aoi_shp, e)

    def _discover_ccdc_years(self):
        import re

        pattern = re.compile(r"(\d{4})\.tif$")
        years = []
        ccdc_dir = self.config.data_sources.ccdc_dir
        if not os.path.isdir(ccdc_dir):
            logger.warning("CCDC directory does not exist: %s", ccdc_dir)
            return
        for fname in os.listdir(ccdc_dir):
            match = pattern.match(fname)
            if match:
                years.append(int(match.group(1)))
        if years:
            min_year = min(years)
            max_year = max(years)
            logger.info("CCDC data available from %d to %d", min_year, max_year)
            if self.start_year < min_year:
                logger.warning("start_year %d adjusted to earliest CCDC year %d", self.start_year, min_year)
                self.start_year = min_year
            if self.end_year > max_year:
                logger.warning("end_year %d adjusted to latest CCDC year %d", self.end_year, max_year)
                self.end_year = max_year
        else:
            logger.warning("No CCDC annual files found in %s", ccdc_dir)

    def _load_dem_metadata(self):
        with rasterio.open(self.dem_loader.file_path) as src_dem:
            self.dem_crs = src_dem.crs
            self.dem_transform = src_dem.transform
            self.dem_width = src_dem.width
            self.dem_height = src_dem.height

    def _compute_chunks_intersecting_aoi(self, src_dem):
        width = src_dem.width
        height = src_dem.height
        transform = src_dem.transform
        crs = src_dem.crs
        if self.aoi is None or crs is None:
            logger.info("AOI not available; processing all DEM chunks")
            for row_off in range(0, height, self.chunk_size):
                for col_off in range(0, width, self.chunk_size):
                    w = min(self.chunk_size, width - col_off)
                    h = min(self.chunk_size, height - row_off)
                    window = Window(col_off, row_off, w, h)
                    yield window, row_off, col_off
            return
        aoi_proj = self.aoi.to_crs(crs)
        left, bottom, right, top = aoi_proj.total_bounds
        row_top, col_left = rowcol(transform, left, top)
        row_bottom, col_right = rowcol(transform, right, bottom)
        row_start = max(0, min(row_top, row_bottom))
        row_end = min(height, max(row_top, row_bottom) + 1)
        col_start = max(0, min(col_left, col_right))
        col_end = min(width, max(col_left, col_right) + 1)
        for row_off in range(row_start, row_end, self.chunk_size):
            for col_off in range(col_start, col_end, self.chunk_size):
                w = min(self.chunk_size, width - col_off)
                h = min(self.chunk_size, height - row_off)
                window = Window(col_off, row_off, w, h)
                yield window, row_off, col_off

    def _build_chunk_tasks(self, src_dem) -> list[ChunkTask]:
        tasks = []
        for chunk_idx, (_, row_off, col_off) in enumerate(self._compute_chunks_intersecting_aoi(src_dem)):
            tasks.append(
                ChunkTask(
                    chunk_idx=chunk_idx,
                    row_off=row_off,
                    col_off=col_off,
                    width=min(self.chunk_size, src_dem.width - col_off),
                    height=min(self.chunk_size, src_dem.height - row_off),
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
        return [task for task in tasks if not chunk_cache_path(self.cache_dir, task.chunk_idx).exists()]

    def _record_chunk_result(self, result: ChunkResult):
        if result.audit is not None:
            self.audit_records.append(asdict(result.audit))

    def _save_audit_report(self):
        if not self.audit_records:
            return
        audit_path = os.path.join(self.cache_dir, "data_audit.json")
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(self.audit_records, handle, ensure_ascii=False, indent=2)
        logger.info("Wrote data audit report: %s", audit_path)

    def _preload_vector_data(self):
        logger.info("Pre-loading vector data for all years...")
        for year in range(self.start_year, self.end_year + 1):
            road_path = os.path.join(self.road_shp_dir, self.road_file_pattern.format(year=year))
            if os.path.exists(road_path) and year not in self.road_loaders_by_year:
                try:
                    loader = VectorLoader(road_path)
                    loader.set_aoi(self.aoi)
                    loader.load()
                    self.road_loaders_by_year[year] = loader
                except Exception as e:
                    logger.warning("Failed to pre-load road data for %d: %s", year, e)
                    self.road_loaders_by_year[year] = None

            railway_path = os.path.join(self.road_shp_dir, self.railway_file_pattern.format(year=year))
            if os.path.exists(railway_path) and year not in self.railway_loaders_by_year:
                try:
                    loader = VectorLoader(railway_path)
                    loader.set_aoi(self.aoi)
                    loader.load()
                    self.railway_loaders_by_year[year] = loader
                except Exception as e:
                    logger.warning("Failed to pre-load railway data for %d: %s", year, e)
                    self.railway_loaders_by_year[year] = None

            waterway_path = os.path.join(self.road_shp_dir, self.waterway_file_pattern.format(year=year))
            if os.path.exists(waterway_path) and year not in self.waterway_loaders_by_year:
                try:
                    loader = VectorLoader(waterway_path)
                    loader.set_aoi(self.aoi)
                    loader.load()
                    self.waterway_loaders_by_year[year] = loader
                except Exception as e:
                    logger.warning("Failed to pre-load waterway data for %d: %s", year, e)
                    self.waterway_loaders_by_year[year] = None
        loaded_roads = sum(1 for v in self.road_loaders_by_year.values() if v is not None)
        loaded_railways = sum(1 for v in self.railway_loaders_by_year.values() if v is not None)
        loaded_waterways = sum(1 for v in self.waterway_loaders_by_year.values() if v is not None)
        logger.info("Pre-loaded vector data: %d roads, %d railways, %d waterways", loaded_roads, loaded_railways, loaded_waterways)

    def process_all_chunks(self):
        logger.info("Starting pipeline processing. Output cache: %s", self.cache_dir)
        if self.aoi is not None:
            self.river_loader.set_aoi(self.aoi)
        self.river_loader.load()
        self._preload_vector_data()
        with rasterio.open(self.dem_loader.file_path) as src_dem:
            tasks = self._build_chunk_tasks(src_dem)

        pending_tasks = self._resolve_pending_tasks(tasks)
        skipped = len(tasks) - len(pending_tasks)
        progress = ProgressTracker(logger, "pipeline", log_every=1)
        progress.start_stage(1, 1, "process chunks", total=len(tasks))
        if skipped:
            progress.advance(skipped, {"saved": len(self.audit_records), "skipped": skipped})

        worker_count = self._resolve_worker_count()
        if worker_count <= 1:
            if PSUTIL_AVAILABLE:
                mem = psutil.virtual_memory()
                logger.info("Memory usage before inline processing: %.1f%% (%.1f GB / %.1f GB)", mem.percent, mem.used / 1024**3, mem.total / 1024**3)
            results = (self._process_single_chunk(task) for task in pending_tasks)
        else:
            global _WORKER
            _WORKER = self
            results = execute_chunk_tasks(
                pending_tasks,
                worker_count=worker_count,
                submit_fn=_process_chunk_task,
            )

        for result in results:
            self._record_chunk_result(result)
            if self.audit_records and len(self.audit_records) % 10 == 0:
                self._save_audit_report()
            progress.advance(1, {"saved": len(self.audit_records), "chunk_idx": result.chunk_idx})

        progress.finish_stage({"saved": len(self.audit_records), "skipped": skipped})
        self._save_audit_report()

    def _process_single_chunk(self, task: ChunkTask) -> ChunkResult:

        window = Window(task.col_off, task.row_off, task.width, task.height)
        dem_data, (res_x, res_y), dem_nodata = self.dem_loader.load(window)
        if np.all(dem_data == dem_nodata) or np.all(np.isnan(dem_data)):
            return ChunkResult(chunk_idx=task.chunk_idx, saved=False)

        slope_data, aspect_data = self.dem_loader.calculate_slope_aspect(dem_data, res_x, res_y, nodata=dem_nodata)
        valid_mask = (dem_data != dem_nodata) & (~np.isnan(dem_data))
        chunk_transform = window_transform(window, self.dem_transform)
        if self.aoi is not None and self.dem_crs is not None:
            try:
                aoi_proj = self.aoi.to_crs(self.dem_crs)
                geom = [geom for geom in aoi_proj.geometry if geom is not None and not geom.is_empty]
                aoi_mask = geometry_mask(geom, transform=chunk_transform, out_shape=dem_data.shape, invert=True)
                valid_mask = valid_mask & aoi_mask
            except Exception as e:
                logger.warning("AOI masking failed: %s", e)
        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            return ChunkResult(chunk_idx=task.chunk_idx, saved=False)

        x_coords, y_coords = rasterio.transform.xy(self.dem_transform, rows + task.row_off, cols + task.col_off)
        x_coords = np.asarray(x_coords)
        y_coords = np.asarray(y_coords)
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
            ccdc_res = self.ccdc_loader.load(year, self.dem_crs, self.dem_transform, self.dem_width, self.dem_height, window=window)
            if ccdc_res is not None:
                ccdc_data, ccdc_nodata = ccdc_res
                disturbance_vals = ccdc_data[rows, cols]
                if ccdc_nodata is not None:
                    disturbance_mask = (disturbance_vals != ccdc_nodata) & (~np.isnan(disturbance_vals))
                else:
                    disturbance_mask = ~np.isnan(disturbance_vals)
                disturbance_years = np.floor(np.where(disturbance_mask, disturbance_vals, np.nan)).astype(float)
                year_match = disturbance_years == float(year)
                disturbance_vals = np.where(disturbance_mask & year_match, disturbance_vals, 0.0)
                df_year["disturbance_time_value"] = disturbance_vals
                df_year["disturbance_type"] = disturbance_vals
                df_year["disturbance_month"] = _decimal_year_to_month(disturbance_vals, df_year["year"].to_numpy())
            else:
                df_year["disturbance_time_value"] = 0.0
                df_year["disturbance_type"] = 0.0
                df_year["disturbance_month"] = np.nan
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
            return ChunkResult(chunk_idx=task.chunk_idx, saved=False)

        df_chunk = pd.concat(chunk_records, ignore_index=True)
        df_chunk["point_id"] = pd.to_numeric(df_chunk["point_id"], downcast="integer")
        df_chunk["year"] = pd.to_numeric(df_chunk["year"], downcast="integer")
        cache_file = str(chunk_cache_path(self.cache_dir, task.chunk_idx))
        atomic_write_npz(cache_file, **{col: df_chunk[col].values for col in df_chunk.columns})
        logger.info("Saved chunk %s with %s valid pixels to %s", task.chunk_idx, len(rows), cache_file)

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
            worker_id=os.getpid(),
        )
        return ChunkResult(chunk_idx=task.chunk_idx, saved=True, cache_file=cache_file, audit=audit)

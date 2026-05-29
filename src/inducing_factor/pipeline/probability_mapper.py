import logging
import os
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window, from_bounds

from inducing_factor.config.models import AppConfig
from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem
from inducing_factor.pipeline.execution import ChunkTask, execute_chunk_tasks, resolve_num_workers
from inducing_factor.utils.progress import ProgressTracker

CLASS_LABELS = {1: "fire", 2: "logging", 3: "geo_hazard", 4: "road_construction"}
_ABS_COL_SHORT = {1: "fire", 2: "logging", 3: "geo_hazard", 4: "road"}

_PREDICT_WORKER_SYSTEM: EcoDisturbanceAttributionSystem | None = None
_PREDICT_WORKER_CHUNK_FILES: tuple[str, ...] = ()
_PREDICT_WORKER_YEAR: int | None = None


def _init_predict_worker(system: EcoDisturbanceAttributionSystem, chunk_files: list[str], year: int):
    global _PREDICT_WORKER_SYSTEM, _PREDICT_WORKER_CHUNK_FILES, _PREDICT_WORKER_YEAR
    _PREDICT_WORKER_SYSTEM = system
    _PREDICT_WORKER_CHUNK_FILES = tuple(chunk_files)
    _PREDICT_WORKER_YEAR = year


def _predict_chunk_worker(task: ChunkTask):
    if _PREDICT_WORKER_SYSTEM is None or _PREDICT_WORKER_YEAR is None:
        raise RuntimeError("Predict worker was not initialized")
    npz_path = _PREDICT_WORKER_CHUNK_FILES[task.chunk_idx - 1]
    try:
        df_chunk = _PREDICT_WORKER_SYSTEM._load_chunk_dataframe(npz_path)
        lb = _PREDICT_WORKER_SYSTEM.feature_builder.lookback_years
        target = _PREDICT_WORKER_YEAR
        df_chunk = df_chunk[df_chunk["year"].isin(range(target - lb - 1, target + 1))].copy()
        if df_chunk.empty:
            return task.chunk_idx, None, None
        df_chunk_samples = _PREDICT_WORKER_SYSTEM._build_chunk_training_samples(df_chunk, skip_sampling=True)
        if df_chunk_samples.empty:
            return task.chunk_idx, None, None
        df_year = df_chunk_samples[df_chunk_samples["year"] == target].copy()
        if df_year.empty or "raster_row" not in df_year.columns or "raster_col" not in df_year.columns:
            return task.chunk_idx, None, None
        for col in _PREDICT_WORKER_SYSTEM.feature_columns:
            if col not in df_year.columns:
                df_year[col] = np.nan
        pred = _PREDICT_WORKER_SYSTEM.predict_dataframe(df_year)
        return task.chunk_idx, pred, None
    except Exception as exc:
        return task.chunk_idx, None, exc


class AnnualProbabilityMapper:
    def __init__(self, config: AppConfig, system: EcoDisturbanceAttributionSystem | None = None):
        self.config = config
        self.system = system or EcoDisturbanceAttributionSystem(config=config)

    def _ensure_system(self):
        if self.system.selected_factors and self.system.discretizer is not None and self.system.occurrence_engine.model is not None:
            return
        self.system.load_artifacts()

    def _resolve_predict_worker_count(self) -> int:
        num_workers = getattr(self.system, "predict_num_workers", getattr(self.config.attribution, "predict_num_workers", 0))
        max_auto_workers = getattr(self.system, "predict_max_auto_workers", getattr(self.config.attribution, "predict_max_auto_workers", 4))
        memory_target_fraction = getattr(
            self.system,
            "predict_memory_target_fraction",
            getattr(self.config.attribution, "predict_memory_target_fraction", 0.7),
        )
        estimated_worker_gb = getattr(
            self.system,
            "predict_estimated_worker_gb",
            getattr(self.config.attribution, "predict_estimated_worker_gb", 1.0),
        )
        return resolve_num_workers(
            num_workers=num_workers,
            max_auto_workers=max_auto_workers,
            memory_target_fraction=memory_target_fraction,
            estimated_worker_gb=estimated_worker_gb,
        )

    def _iter_year_predictions(self, chunk_files: list[str], year: int):
        worker_count = self._resolve_predict_worker_count()
        if worker_count <= 1:
            for idx, npz_path in enumerate(chunk_files, start=1):
                try:
                    df_chunk = self.system._load_chunk_dataframe(npz_path)
                    df_chunk_samples = self.system._build_chunk_training_samples(df_chunk, skip_sampling=True)
                    if df_chunk_samples.empty:
                        yield idx, None, None
                        continue
                    df_year = df_chunk_samples[df_chunk_samples["year"] == year].copy()
                    if df_year.empty or "raster_row" not in df_year.columns or "raster_col" not in df_year.columns:
                        yield idx, None, None
                        continue
                    for col in self.system.feature_columns:
                        if col not in df_year.columns:
                            df_year[col] = np.nan
                    pred = self.system.predict_dataframe(df_year)
                    yield idx, pred, None
                except Exception as exc:  # pragma: no cover - defensive branch
                    yield idx, None, exc
            return

        tasks = [ChunkTask(chunk_idx=idx, row_off=0, col_off=0, width=1, height=1) for idx in range(1, len(chunk_files) + 1)]
        results = execute_chunk_tasks(
            tasks,
            worker_count=worker_count,
            submit_fn=_predict_chunk_worker,
            initializer=_init_predict_worker,
            initargs=(self.system, chunk_files, year),
        )
        for chunk_idx, pred, exc in results:
            yield chunk_idx, pred, exc

    def write_maps(self, year: int, output_dir: str | None = None) -> dict[str, str]:
        self._ensure_system()
        output_dir = output_dir or os.path.join(self.config.paths.out_dir, "probability_maps")
        os.makedirs(output_dir, exist_ok=True)
        target_crs = "EPSG:4326"

        with rasterio.open(self.config.data_sources.dem_tif) as src_dem:
            dem_crs = src_dem.crs
            dem_transform = src_dem.transform

            aoi_path = self.config.data_sources.aoi_shp
            aoi = gpd.read_file(aoi_path)
            aoi_dem = aoi.to_crs(dem_crs)
            aoi_bounds = aoi_dem.total_bounds
            aoi_window = from_bounds(*aoi_bounds, transform=dem_transform)
            aoi_window = aoi_window.round_lengths().round_offsets()
            out_width = int(aoi_window.width)
            out_height = int(aoi_window.height)
            out_transform = src_dem.window_transform(aoi_window)

            src_profile = dict(
                driver="GTiff",
                height=out_height,
                width=out_width,
                count=1,
                dtype="float32",
                crs=dem_crs,
                transform=out_transform,
                nodata=-9999.0,
                compress="lzw",
            )
            src_class_profile = {**src_profile, "dtype": "int16", "nodata": 0}

            aoi_row0 = int(aoi_window.row_off)
            aoi_col0 = int(aoi_window.col_off)

            outputs = {
                "disturbance_probability": os.path.join(output_dir, f"disturbance_probability_{year}.tif"),
                "dominant_disturbance_class": os.path.join(output_dir, f"dominant_disturbance_class_{year}.tif"),
            }
            for cls in [1, 2, 3, 4]:
                outputs[f"class_probability_{cls}"] = os.path.join(output_dir, f"class_probability_{year}_{cls}.tif")
                outputs[f"abs_probability_{cls}"] = os.path.join(output_dir, f"prob_{CLASS_LABELS[cls]}_{year}.tif")

            temp_dir = tempfile.mkdtemp(prefix="prob_maps_")
            temp_paths = {}
            for key, path in outputs.items():
                temp_paths[key] = os.path.join(temp_dir, os.path.basename(path))
                _ = key

            profile_for_key = {k: src_profile for k in outputs}
            profile_for_key["dominant_disturbance_class"] = src_class_profile

            dst_handles = {}
            for key, path in temp_paths.items():
                dst_handles[key] = rasterio.open(path, "w", **profile_for_key[key])

            with dst_handles["disturbance_probability"] as disturbance_dst, \
                 dst_handles["dominant_disturbance_class"] as dominant_dst, \
                 dst_handles["class_probability_1"] as class1_dst, \
                 dst_handles["class_probability_2"] as class2_dst, \
                 dst_handles["class_probability_3"] as class3_dst, \
                 dst_handles["class_probability_4"] as class4_dst, \
                 dst_handles["abs_probability_1"] as abs_fire_dst, \
                 dst_handles["abs_probability_2"] as abs_logging_dst, \
                 dst_handles["abs_probability_3"] as abs_geo_dst, \
                 dst_handles["abs_probability_4"] as abs_road_dst:

                class_dsts = {1: class1_dst, 2: class2_dst, 3: class3_dst, 4: class4_dst}
                abs_dsts = {1: abs_fire_dst, 2: abs_logging_dst, 3: abs_geo_dst, 4: abs_road_dst}
                chunk_files = list(self.system._iter_chunk_files())
                progress = ProgressTracker(logging.getLogger("InducingFactorSystem"), "predict-annual-map", log_every=1)
                progress.start_stage(1, 1, f"write maps for {year}", total=len(chunk_files))
                for idx, pred, exc in self._iter_year_predictions(chunk_files, year):
                    if exc is not None:
                        progress.advance(1, {"chunk": idx, "written": 0, "error": str(exc)})
                        continue
                    if pred is None or pred.empty:
                        progress.advance(1, {"chunk": idx, "written": 0})
                        continue
                    rows = pred["raster_row"].astype(int).to_numpy()
                    cols = pred["raster_col"].astype(int).to_numpy()
                    row0, row1 = rows.min(), rows.max() + 1
                    col0, col1 = cols.min(), cols.max() + 1

                    out_row0 = row0 - aoi_row0
                    out_col0 = col0 - aoi_col0
                    out_row1 = row1 - aoi_row0
                    out_col1 = col1 - aoi_col0

                    if out_row1 <= 0 or out_col1 <= 0 or out_row0 >= out_height or out_col0 >= out_width:
                        progress.advance(1, {"chunk": idx, "written": 0})
                        continue

                    out_row0 = max(0, out_row0)
                    out_col0 = max(0, out_col0)
                    out_row1 = min(out_height, out_row1)
                    out_col1 = min(out_width, out_col1)

                    rr = rows - row0
                    cc = cols - col0

                    out_rr = rr + (out_row0 - (row0 - aoi_row0))
                    out_cc = cc + (out_col0 - (col0 - aoi_col0))
                    out_h = out_row1 - out_row0
                    out_w = out_col1 - out_col0

                    disturbance_arr = np.full((out_h, out_w), -9999.0, dtype=np.float32)
                    mask = (out_rr >= 0) & (out_rr < out_h) & (out_cc >= 0) & (out_cc < out_w)
                    disturbance_arr[out_rr[mask], out_cc[mask]] = pred["disturbance_probability"].astype(float).to_numpy()[mask]
                    window_out = Window(out_col0, out_row0, out_w, out_h)
                    disturbance_dst.write(disturbance_arr, 1, window=window_out)

                    dominant_arr = np.zeros((out_h, out_w), dtype=np.int16)
                    if "dominant_disturbance_class" in pred.columns:
                        dominant_arr[out_rr[mask], out_cc[mask]] = pred["dominant_disturbance_class"].astype(int).to_numpy()[mask]
                    dominant_dst.write(dominant_arr, 1, window=window_out)

                    for cls in [1, 2, 3, 4]:
                        prob_col = f"class_probability_{cls}"
                        abs_col = f"prob_{_ABS_COL_SHORT[cls]}_abs"
                        class_arr = np.full((out_h, out_w), -9999.0, dtype=np.float32)
                        abs_arr = np.full((out_h, out_w), -9999.0, dtype=np.float32)
                        if prob_col in pred.columns:
                            probs = pred[prob_col].astype(float).to_numpy()
                            class_arr[out_rr[mask], out_cc[mask]] = probs[mask]
                            if abs_col in pred.columns:
                                abs_arr[out_rr[mask], out_cc[mask]] = pred[abs_col].astype(float).to_numpy()[mask]
                            else:
                                abs_arr[out_rr[mask], out_cc[mask]] = probs[mask] * pred["disturbance_probability"].astype(float).to_numpy()[mask]
                        class_dsts[cls].write(class_arr, 1, window=window_out)
                        abs_dsts[cls].write(abs_arr, 1, window=window_out)
                    progress.advance(1, {"chunk": idx, "written": len(pred)})
                progress.finish_stage({"outputs": len(outputs)})

            log = logging.getLogger("InducingFactorSystem")
            for key, path in outputs.items():
                temp_path = temp_paths[key]
                log.info("[predict] reprojecting %s to %s", key, target_crs)
                with rasterio.open(temp_path) as src:
                    dst_crs_proj, dst_width, dst_height = calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds
                    )
                    dst_profile = src.profile.copy()
                    dst_profile.update(
                        crs=target_crs,
                        transform=dst_crs_proj,
                        width=dst_width,
                        height=dst_height,
                    )
                    src_nodata = src.profile.get("nodata", -9999.0)
                    dst_nodata = src_nodata
                    resampling = Resampling.nearest if key == "dominant_disturbance_class" else Resampling.bilinear
                    with rasterio.open(path, "w", **dst_profile) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_crs_proj,
                            dst_crs=target_crs,
                            src_nodata=src_nodata,
                            dst_nodata=dst_nodata,
                            resampling=resampling,
                        )
                os.remove(temp_path)
            os.rmdir(temp_dir)
        return outputs

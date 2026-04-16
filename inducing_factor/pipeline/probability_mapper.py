import logging
import os

import numpy as np
import rasterio
from rasterio.windows import Window

from inducing_factor.config.models import AppConfig
from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem
from inducing_factor.utils.progress import ProgressTracker


class AnnualProbabilityMapper:
    def __init__(self, config: AppConfig, system: EcoDisturbanceAttributionSystem | None = None):
        self.config = config
        self.system = system or EcoDisturbanceAttributionSystem(config=config)

    def _ensure_system(self):
        if self.system.selected_factors and self.system.discretizer is not None and self.system.occurrence_engine.model is not None:
            return
        self.system.load_artifacts()

    def write_maps(self, year: int, output_dir: str | None = None) -> dict[str, str]:
        self._ensure_system()
        output_dir = output_dir or os.path.join(self.config.paths.out_dir, "probability_maps")
        os.makedirs(output_dir, exist_ok=True)

        with rasterio.open(self.config.data_sources.dem_tif) as src_dem:
            float_profile = src_dem.profile.copy()
            float_profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
            class_profile = src_dem.profile.copy()
            class_profile.update(dtype="int16", count=1, nodata=0, compress="lzw")

            outputs = {
                "disturbance_probability": os.path.join(output_dir, f"disturbance_probability_{year}.tif"),
                "dominant_disturbance_class": os.path.join(output_dir, f"dominant_disturbance_class_{year}.tif"),
            }
            for cls in [1, 2, 3, 4]:
                outputs[f"class_probability_{cls}"] = os.path.join(output_dir, f"class_probability_{year}_{cls}.tif")
                outputs[f"joint_class_probability_{cls}"] = os.path.join(output_dir, f"joint_class_probability_{year}_{cls}.tif")

            with rasterio.open(outputs["disturbance_probability"], "w", **float_profile) as disturbance_dst, rasterio.open(
                outputs["dominant_disturbance_class"], "w", **class_profile
            ) as dominant_dst, rasterio.open(outputs["class_probability_1"], "w", **float_profile) as class1_dst, rasterio.open(
                outputs["class_probability_2"], "w", **float_profile
            ) as class2_dst, rasterio.open(outputs["class_probability_3"], "w", **float_profile) as class3_dst, rasterio.open(
                outputs["class_probability_4"], "w", **float_profile
            ) as class4_dst, rasterio.open(outputs["joint_class_probability_1"], "w", **float_profile) as joint1_dst, rasterio.open(
                outputs["joint_class_probability_2"], "w", **float_profile
            ) as joint2_dst, rasterio.open(outputs["joint_class_probability_3"], "w", **float_profile) as joint3_dst, rasterio.open(
                outputs["joint_class_probability_4"], "w", **float_profile
            ) as joint4_dst:
                class_dsts = {1: class1_dst, 2: class2_dst, 3: class3_dst, 4: class4_dst}
                joint_dsts = {1: joint1_dst, 2: joint2_dst, 3: joint3_dst, 4: joint4_dst}
                chunk_files = list(self.system._iter_chunk_files())
                progress = ProgressTracker(logging.getLogger("InducingFactorSystem"), "predict-annual-map", log_every=1)
                progress.start_stage(1, 1, f"write maps for {year}", total=len(chunk_files))
                for idx, npz_path in enumerate(chunk_files, start=1):
                    df_chunk = self.system._load_chunk_dataframe(npz_path)
                    df_year = df_chunk[df_chunk["year"] == year].copy()
                    if df_year.empty or "raster_row" not in df_year.columns or "raster_col" not in df_year.columns:
                        progress.advance(1, {"chunk": idx, "written": 0})
                        continue
                    pred = self.system.predict_dataframe(df_year)
                    rows = pred["raster_row"].astype(int).to_numpy()
                    cols = pred["raster_col"].astype(int).to_numpy()
                    row0, row1 = rows.min(), rows.max() + 1
                    col0, col1 = cols.min(), cols.max() + 1
                    window = Window(col0, row0, col1 - col0, row1 - row0)
                    rr = rows - row0
                    cc = cols - col0

                    disturbance_arr = np.full((row1 - row0, col1 - col0), -9999.0, dtype=np.float32)
                    disturbance_arr[rr, cc] = pred["disturbance_probability"].astype(float).to_numpy()
                    disturbance_dst.write(disturbance_arr, 1, window=window)

                    dominant_arr = np.zeros((row1 - row0, col1 - col0), dtype=np.int16)
                    if "dominant_disturbance_class" in pred.columns:
                        dominant_arr[rr, cc] = pred["dominant_disturbance_class"].astype(int).to_numpy()
                    dominant_dst.write(dominant_arr, 1, window=window)

                    for cls in [1, 2, 3, 4]:
                        prob_col = f"class_probability_{cls}"
                        class_arr = np.full((row1 - row0, col1 - col0), -9999.0, dtype=np.float32)
                        joint_arr = np.full((row1 - row0, col1 - col0), -9999.0, dtype=np.float32)
                        if prob_col in pred.columns:
                            probs = pred[prob_col].astype(float).to_numpy()
                            class_arr[rr, cc] = probs
                            joint_arr[rr, cc] = probs * pred["disturbance_probability"].astype(float).to_numpy()
                        class_dsts[cls].write(class_arr, 1, window=window)
                        joint_dsts[cls].write(joint_arr, 1, window=window)
                    progress.advance(1, {"chunk": idx, "written": len(pred)})
                progress.finish_stage({"outputs": len(outputs)})
        return outputs

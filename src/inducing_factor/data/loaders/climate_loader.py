import os
import threading
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile
import numpy as np
from rasterio.windows import Window
import xarray as xr
from .base import BaseLoader

class ClimateLoader(BaseLoader):
    """
    Temporal loader for climate data (Temperature, Rainfall).
    Handles missing years by implementing a backfill/forward-fill mechanism.
    Aligns dynamically to the DEM using WarpedVRT.
    Thread-safe: per-thread state for backfill cache.
    """
    def __init__(self, dir_path: str, file_pattern: str = "{year}.tif", config: dict = None):
        super().__init__(dir_path, config)
        self.dir_path = dir_path
        self.file_pattern = file_pattern
        self._local = threading.local()
        self._nc_cache = {}

    def check_data(self, start_year: int, end_year: int):
        self.logger.info(f"--- Checking Climate Data in: {self.dir_path} ---")
        for year in range(start_year, end_year + 1):
            file_name = self.file_pattern.format(year=year)
            full_path = os.path.join(self.dir_path, file_name)
            if os.path.exists(full_path):
                self.logger.info(f"Year {year}: Found -> {file_name}")
            else:
                self.logger.warning(f"Year {year}: MISSING -> {file_name}")

    def _open_as_rasterio(self, full_path, year):
        """Helper to open both TIF and NC files as rasterio datasets."""
        if full_path.endswith('.nc'):
            # Temperature NC files don't work with rasterio directly due to missing netCDF driver.
            # We open with xarray, extract July (index 6, assuming 0-11 for Jan-Dec), and write to MemoryFile.
            try:
                if full_path not in self._nc_cache:
                    ds = xr.open_dataset(full_path)
                    # Extract the 7th month (July) as a representative summer temperature, or mean
                    # The data has time dimension, usually 12 months.
                    if 'time' in ds.dims:
                        ds_month = ds.isel(time=6) # July
                    else:
                        ds_month = ds
                    
                    val_array = ds_month['tmp'].values.astype('float32')
                    lats = ds_month['lat'].values
                    lons = ds_month['lon'].values
                    
                    # Create transform from lat/lon
                    res_x = (lons[-1] - lons[0]) / len(lons)
                    res_y = (lats[-1] - lats[0]) / len(lats)
                    transform = rasterio.transform.from_origin(lons[0], lats[0], abs(res_x), abs(res_y))
                    
                    memfile = MemoryFile()
                    with memfile.open(
                        driver='GTiff',
                        height=val_array.shape[0],
                        width=val_array.shape[1],
                        count=1,
                        dtype=str(val_array.dtype),
                        crs='+proj=latlong',
                        transform=transform,
                    ) as dataset:
                        dataset.write(val_array, 1)
                    self._nc_cache[full_path] = memfile
                
                return self._nc_cache[full_path].open()
            except Exception as e:
                self.logger.error(f"Failed to parse NC file {full_path}: {e}")
                raise e
        else:
            return rasterio.open(full_path)

    def _build_monthly_path(self, year: int, month: int) -> str:
        return os.path.join(self.dir_path, f"pre{year:04d}{month:02d}.tif")

    def _read_aligned_window(self, full_path: str, dem_crs, dem_transform, dem_width, dem_height, window: Window):
        with self._open_as_rasterio(full_path, None) as src:
            with WarpedVRT(src, crs=dem_crs, transform=dem_transform, width=dem_width, height=dem_height) as vrt:
                return vrt.read(1, window=window)

    def load(self, year: int, dem_crs, dem_transform, dem_width, dem_height, window: Window):
        """
        Loads data using WarpedVRT to dynamically align with the DEM's properties.
        """
        # We assume July for rainfall if file_pattern is pre{year}07.tif
        file_name = self.file_pattern.format(year=year)
        full_path = os.path.join(self.dir_path, file_name)

        if not os.path.exists(full_path):
            self.logger.warning(f"Climate data missing for year {year} at {full_path}.")
            if self._local.last_valid_data is not None:
                return self._local.last_valid_data
            return None

        try:
            data = self._read_aligned_window(full_path, dem_crs, dem_transform, dem_width, dem_height, window)
            self._local.last_valid_data = data
            self._local.last_valid_year = year
            return data
        except Exception as e:
            self.logger.error(f"Failed to read climate data {year} ({full_path}): {e}")
            if self._local.last_valid_data is not None:
                return self._local.last_valid_data
            return None

    def load_month(self, year: int, month: int, dem_crs, dem_transform, dem_width, dem_height, window: Window):
        if month < 1 or month > 12:
            raise ValueError(f"Month must be in 1..12, got {month}")

        full_path = self._build_monthly_path(year, month)
        if not os.path.exists(full_path):
            self.logger.warning(f"Monthly climate data missing for {year}-{month:02d} at {full_path}.")
            return None

        try:
            return self._read_aligned_window(full_path, dem_crs, dem_transform, dem_width, dem_height, window)
        except Exception as e:
            self.logger.error(f"Failed to read monthly climate data {year}-{month:02d} ({full_path}): {e}")
            return None

    def clear_state(self):
        self._local.last_valid_data = None
        self._local.last_valid_year = None

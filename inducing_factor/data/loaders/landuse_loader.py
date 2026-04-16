import os
import rasterio
from rasterio.vrt import WarpedVRT
import numpy as np
from rasterio.windows import Window
from .base import BaseLoader

class LandUseLoader(BaseLoader):
    """
    Loader for Land Use / Land Cover data.
    Can operate as a static loader (single year) or temporal loader.
    """
    def __init__(self, path: str, is_temporal: bool = False, file_pattern: str = "{year}.tif", config: dict = None, data_type: str = "LULC"):
        super().__init__(path, config)
        self.is_temporal = is_temporal
        self.file_pattern = file_pattern
        self.data_type = data_type
        self._last_valid_data = None
        self._last_valid_year = None

    def check_data(self):
        self.logger.info(f"--- Checking {self.data_type} Data: {self.file_path} ---")
        if not self.is_temporal:
            if os.path.exists(self.file_path):
                self.logger.info(f"Static {self.data_type} dataset found.")
            else:
                self.logger.error(f"Static {self.data_type} file not found.")
        else:
            self.logger.info(f"Temporal mode enabled for {self.data_type}.")

    def load(self, year: int, dem_crs, dem_transform, dem_width, dem_height, window: Window):
        """
        Loads Land Use data aligned via WarpedVRT. Uses backfill logic if in temporal mode.
        """
        if not self.is_temporal:
            target_path = self.file_path
        else:
            if year is None:
                raise ValueError("Year must be provided in temporal mode.")
            target_path = os.path.join(self.file_path, self.file_pattern.format(year=year))

        if not os.path.exists(target_path):
            if self.is_temporal and self._last_valid_data is not None:
                 self.logger.warning(f"{self.data_type} data missing for {year}, backfilling from {self._last_valid_year}.")
                 return self._last_valid_data
            return None

        try:
            with rasterio.open(target_path) as src:
                with WarpedVRT(src, crs=dem_crs, transform=dem_transform, width=dem_width, height=dem_height) as vrt:
                    data = vrt.read(1, window=window)
                    if self.is_temporal:
                        self._last_valid_data = data
                        self._last_valid_year = year
                    return data
        except Exception as e:
            self.logger.error(f"Error reading {self.data_type} file at {target_path}: {e}")
            if self.is_temporal and self._last_valid_data is not None:
                return self._last_valid_data
            return None

    def clear_state(self):
        self._last_valid_data = None
        self._last_valid_year = None

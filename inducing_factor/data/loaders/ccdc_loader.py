import os
import rasterio
from rasterio.vrt import WarpedVRT
import numpy as np
from rasterio.windows import Window
from .base import BaseLoader

class CCDCLoader(BaseLoader):
    """
    CCDC Disturbance Data Loader.
    Reads yearly disturbance files where pixel values represent disturbance types:
    1: Fire
    2: Logging
    3: Geo-hazard
    4: Construction
    """
    DISTURBANCE_TYPES = {
        1: "Fire",
        2: "Logging",
        3: "Geo-hazard",
        4: "Construction"
    }

    def __init__(self, dir_path: str, file_pattern: str = "{year}.tif", config: dict = None):
        super().__init__(dir_path, config)
        self.dir_path = dir_path
        self.file_pattern = file_pattern

    def check_data(self, start_year: int, end_year: int, window_size: int = 100):
        """
        Checks the availability and validity of CCDC files over a range of years.
        """
        self.logger.info(f"--- Checking CCDC Directory: {self.dir_path} ---")
        for year in range(start_year, end_year + 1):
            file_name = self.file_pattern.format(year=year)
            full_path = os.path.join(self.dir_path, file_name)
            
            if not os.path.exists(full_path):
                self.logger.warning(f"Year {year}: MISSING -> {file_name}")
                continue
                
            try:
                with rasterio.open(full_path) as src:
                    self.logger.info(f"Year {year}: {file_name} -> {src.width}x{src.height}, nodata: {src.nodata}")
            except Exception as e:
                self.logger.error(f"Failed to read CCDC {year}: {e}")

    def load(self, year: int, dem_crs, dem_transform, dem_width, dem_height, window: Window = None):
        """
        Loads the disturbance data aligned via WarpedVRT for a specific year and window.
        Returns the disturbance array, where values 1-4 indicate the type of disturbance.
        If the file is missing, returns None.
        """
        file_name = self.file_pattern.format(year=year)
        full_path = os.path.join(self.dir_path, file_name)

        if not os.path.exists(full_path):
            self.logger.warning(f"CCDC data missing for year {year} at {full_path}.")
            return None

        try:
            with rasterio.open(full_path) as src:
                with WarpedVRT(src, crs=dem_crs, transform=dem_transform, width=dem_width, height=dem_height) as vrt:
                    data = vrt.read(1, window=window)
                    return data, src.nodata
        except Exception as e:
            self.logger.error(f"Error reading CCDC data for {year}: {e}")
            return None

    def extract_disturbance_points(self, data_array, nodata_value, x_coords, y_coords):
        """
        Helper method to extract the coordinates and types of disturbances 
        from a loaded array for downstream spatial analysis.
        """
        if nodata_value is not None:
            valid_mask = (data_array != nodata_value) & (~np.isnan(data_array))
        else:
            valid_mask = ~np.isnan(data_array)
            
        # Disturbance values are typically 1, 2, 3, 4 based on project definition
        disturbed_mask = valid_mask & (data_array > 0)
        
        rows, cols = np.where(disturbed_mask)
        types = data_array[rows, cols]
        
        dist_x = x_coords[rows, cols]
        dist_y = y_coords[rows, cols]
        
        return dist_x, dist_y, types

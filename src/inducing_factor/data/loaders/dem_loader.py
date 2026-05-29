import rasterio
import numpy as np
from rasterio.windows import Window
from .base import BaseLoader

class DEMLoader(BaseLoader):
    """
    DEM data loader.
    Handles reading elevation and calculating slope and aspect dynamically for a given spatial window.
    """
    def check_data(self):
        self.logger.info(f"--- Checking DEM file: {self.file_path} ---")
        try:
            with rasterio.open(self.file_path) as src:
                self.logger.info(f"Dimensions: {src.width} x {src.height}, Bands: {src.count}")
                self.logger.info(f"CRS: {src.crs}, Resolution: {src.res}")
                self.logger.info(f"NoData Value: {src.nodatavals}")
        except Exception as e:
            self.logger.error(f"Failed to read DEM: {e}")

    def load(self, window: Window = None):
        """
        Loads elevation data for a specific window.
        Returns:
            elevation_array, (res_x, res_y), nodata_value
        """
        with rasterio.open(self.file_path) as src:
            data = src.read(1, window=window)
            res_x, res_y = src.res
            nodata = src.nodata
            return data, (res_x, res_y), nodata

    def calculate_slope_aspect(self, elev_array, res_x, res_y, nodata=None):
        """
        Dynamically calculates slope and aspect for the loaded elevation array.
        This prevents needing to store massive slope/aspect rasters on disk or in memory.
        """
        elev_array = elev_array.astype(float)
        
        if nodata is not None:
            nodata_mask = (elev_array == nodata) | np.isnan(elev_array)
        else:
            nodata_mask = np.isnan(elev_array)

        # Use numpy gradient for fast full-array calculation
        dz_dx, dz_dy = np.gradient(elev_array, res_x, res_y)
        
        # Slope
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = slope_rad * 57.29578
        
        # Aspect
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        aspect_deg = aspect_rad * 57.29578
        
        # Convert aspect to 0-360 azimuth
        aspect_deg = np.where(aspect_deg < 0, 90.0 - aspect_deg, 
                              np.where(aspect_deg > 90.0, 360.0 - aspect_deg + 90.0, 90.0 - aspect_deg))
        
        # Apply NoData mask
        slope_deg[nodata_mask] = np.nan
        aspect_deg[nodata_mask] = np.nan
        
        return slope_deg, aspect_deg

import os

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from .base import BaseLoader


class DisturbanceClassLoader(BaseLoader):
    """Loads annual disturbance classes from a multi-band raster."""

    def __init__(
        self,
        file_path: str,
        start_year: int = 2016,
        class_labels: dict[int, str] | None = None,
        config: dict | None = None,
    ):
        super().__init__(file_path, config)
        self.file_path = file_path
        self.start_year = start_year
        self.class_labels = class_labels or {
            1: "fire",
            2: "logging",
            3: "geo_hazard",
            4: "road_construction",
        }
        self._band_year_map: dict[int, int] | None = None
        self._year_band_map: dict[int, int] | None = None

    @property
    def band_year_map(self) -> dict[int, int]:
        if self._band_year_map is None:
            self._build_band_mappings()
        return self._band_year_map or {}

    @property
    def year_band_map(self) -> dict[int, int]:
        if self._year_band_map is None:
            self._build_band_mappings()
        return self._year_band_map or {}

    def _build_band_mappings(self):
        with rasterio.open(self.file_path) as src:
            self._band_year_map = {band: self.start_year + band - 1 for band in range(1, src.count + 1)}
            self._year_band_map = {year: band for band, year in self._band_year_map.items()}

    def available_years(self) -> list[int]:
        return sorted(self.year_band_map.keys())

    def load(self, year: int, dem_crs, dem_transform, dem_width, dem_height, window: Window | None = None):
        band = self.year_band_map.get(year)
        if band is None:
            return None
        if not os.path.exists(self.file_path):
            self.logger.warning("Disturbance class raster missing at %s", self.file_path)
            return None
        try:
            with rasterio.open(self.file_path) as src:
                with WarpedVRT(src, crs=dem_crs, transform=dem_transform, width=dem_width, height=dem_height) as vrt:
                    data = vrt.read(band, window=window)
                    return data, src.nodata
        except Exception as exc:
            self.logger.error("Error reading disturbance class raster for year %s: %s", year, exc)
            return None

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.strtree import STRtree
from .base import BaseLoader

class VectorLoader(BaseLoader):
    """
    Vector data loader for calculating distances to features like roads, rivers, or residents.
    Loads vector data once, reprojects to UTM (EPSG:32646) for accurate metric distance,
    and builds a spatial index (KDTree) for fast queries.
    """
    def __init__(self, file_path: str, config: dict = None):
        super().__init__(file_path, config)
        self.gdf = None
        self.tree = None
        self.metric_crs = CRS.from_epsg(32646)
        self.transformers_to_metric = {}
        self.geometries = []
        self.aoi_geom = None

    def check_data(self):
        self.logger.info(f"--- Checking Vector file: {self.file_path} ---")
        try:
            gdf = self._read_file(rows=10)
            self.logger.info(f"CRS: {gdf.crs}")
            self.logger.info(f"Geometry Types: {gdf.geom_type.unique()}")
        except Exception as e:
            self.logger.error(f"Failed to read Vector: {e}")

    def _read_file(self, **kwargs):
        read_errors = []
        for encoding in (None, "utf-8", "gb18030", "gbk", "latin1"):
            try:
                options = dict(kwargs)
                if encoding is not None:
                    options["encoding"] = encoding
                return gpd.read_file(self.file_path, **options)
            except Exception as exc:
                read_errors.append(f"{encoding or 'default'}: {exc}")
        raise ValueError("; ".join(read_errors))

    def set_aoi(self, aoi_gdf):
        if aoi_gdf is None:
            self.aoi_geom = None
            return
        try:
            # Convert AOI to metric CRS (EPSG:32646) for clipping
            aoi_metric = aoi_gdf.to_crs(self.metric_crs)
            # Union all geometries into a single polygon/multipolygon
            self.aoi_geom = aoi_metric.geometry.unary_union
            self.logger.info("AOI set for vector clipping")
        except Exception as e:
            self.logger.warning(f"Failed to set AOI: {e}")
            self.aoi_geom = None

    def load(self):
        if self.gdf is None:
            self.logger.info(f"Loading vector data into memory from {self.file_path}...")
            self.gdf = self._read_file()
            # Clip to AOI if set
            if self.aoi_geom is not None:
                # Ensure gdf is in metric CRS
                if self.gdf.crs and self.gdf.crs.to_string() != self.metric_crs.to_string():
                    self.gdf = self.gdf.to_crs(self.metric_crs)
                # Filter geometries intersecting AOI
                self.gdf = self.gdf[self.gdf.geometry.intersects(self.aoi_geom)].copy()
                self.logger.info(f"Vector data clipped to AOI: {len(self.gdf)} features remaining")
                # After clipping, ensure geometry is in metric CRS
                if self.gdf.crs and self.gdf.crs.to_string() != self.metric_crs.to_string():
                    self.gdf = self.gdf.to_crs(self.metric_crs)
            else:
                # No AOI clipping, still need to reproject to metric CRS for distance calculations
                if self.gdf.crs and self.gdf.crs.to_string() != self.metric_crs.to_string():
                    self.logger.info("Reprojecting vector data to EPSG:32646 (UTM Zone 46N) for accurate distances...")
                    self.gdf = self.gdf.to_crs(self.metric_crs)
                
            self._build_kdtree()
        return self.gdf

    def _build_kdtree(self):
        self.geometries = [geom for geom in self.gdf.geometry if geom is not None and not geom.is_empty]
        if self.geometries:
            self.tree = STRtree(self.geometries)
            self.logger.info(f"Built STRtree with {len(self.geometries)} metric geometries.")
        else:
            self.logger.warning("No valid geometries found to build STRtree.")

    def _metric_transformer(self, input_crs):
        crs_key = CRS.from_user_input(input_crs).to_string()
        if crs_key not in self.transformers_to_metric:
            self.transformers_to_metric[crs_key] = Transformer.from_crs(input_crs, self.metric_crs, always_xy=True)
        return self.transformers_to_metric[crs_key]

    def get_distance(self, x_coords: np.ndarray, y_coords: np.ndarray, input_crs, max_dist=None):
        if self.tree is None:
            self.load()

        if self.tree is None:
            return np.full(len(x_coords), np.nan)

        transformer = self._metric_transformer(input_crs)
        x_metric, y_metric = transformer.transform(x_coords, y_coords)
        distances = np.empty(len(x_metric), dtype=float)
        for idx, (x_val, y_val) in enumerate(zip(x_metric, y_metric)):
            point = Point(x_val, y_val)
            nearest_idx = self.tree.nearest(point)
            if nearest_idx is None:
                distances[idx] = np.nan
                continue
            distance = point.distance(self.geometries[int(nearest_idx)])
            distances[idx] = np.nan if max_dist is not None and distance > max_dist else distance
        return distances

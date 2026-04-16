from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathsConfig:
    cache_dir: str = "data/.cache/chunks"
    out_dir: str = "data/.cache/patent_system"


@dataclass
class DataSourcesConfig:
    aoi_shp: str = "data/aoi/LangToBasighat.shp"
    dem_tif: str = "data/dem/tibet_dem.tif"
    landuse_dir: str = "data/landuse"
    rainfall_dir: str = "data/rainfall"
    temperature_dir: str = "data/temperature"
    resident_dir: str = "data/resident"
    river_shp_dir: str = "data/river"
    road_shp_dir: str = "data/road"
    ccdc_dir: str = "data/ccdc_res"
    disturbance_class_tif: str = "data/disturbance_class/disturbance_class_res.tif"
    # Optional patterns
    landuse_file_pattern: str = "CLCD_v01_{year}_albert_province/CLCD_v01_{year}_albert_xizang.tif"
    road_file_pattern: str = "osm_{year}_roads.shp"
    railway_file_pattern: str = "osm_{year}_railways.shp"
    waterway_file_pattern: str = "osm_{year}_waterways.shp"
    rainfall_monthly_file_pattern: str = "pre{year}{month:02d}.tif"
    temperature_file_pattern: str = "tmp_{year}.nc"
    resident_file_pattern: str = "GURS_{year}.tif"
    ccdc_file_pattern: str = "{year}.tif"
    disturbance_class_start_year: int = 2016
    # NoData values
    dem_nodata: float = -9999.0
    resident_nodata: float = 127.0


@dataclass
class PipelineConfig:
    start_year: int = 1990
    end_year: int = 2020
    chunk_size: int = 1024


@dataclass
class AttributionConfig:
    lookback_years: int = 3
    background_sample_rate: float = 0.05
    discretizer_method: str = "supervised_tree"
    discretize_bins: int = 3
    max_pattern_size: int = 3
    min_support: float = 0.05
    min_relative_risk: float = 1.5
    bootstrap_iterations: int = 10
    min_stability: float = 0.6
    max_bootstrap_samples: int = 3000
    max_network_factors: int = 6
    dominant_threshold: float = 0.55
    synergy_threshold: float = 0.75
    random_state: int = 42
    discretizer_sample_limit: int = 200000
    model_sample_limit: int = 200000
    progress_log_every: int = 1


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)

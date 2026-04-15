from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathsConfig:
    cache_dir: str = "/Volumes/T9/Datasets/.cache/chunks"
    out_dir: str = "/Volumes/T9/Datasets/.cache/patent_system"


@dataclass
class DataSourcesConfig:
    aoi_shp: str = "/Volumes/T9/Datasets/aoi/yarlung_zangbo_downstream_lang_to_paisighat_buffered_2km/LangToBasighat.shp"
    dem_tif: str = "/Volumes/T9/Datasets/dem/tibet/tibet_dem.tif"
    landuse_dir: str = "/Volumes/T9/Datasets/landuse/【立方数据学社】分省份的数据"
    rainfall_dir: str = "/Volumes/T9/Datasets/rainfall/【立方数据学社】西藏自治区"
    temperature_dir: str = "/Volumes/T9/Datasets/temperature"
    resident_dir: str = "/Volumes/T9/Datasets/resident/方创琳教授团队发布全球首套100 m城市和农村居住区数据集(2000-2020年)"
    river_shp_dir: str = "/Volumes/T9/Datasets/river/Global_Chimaps_shp_Data_Record_A"
    road_shp_dir: str = "/Volumes/T9/Datasets/road/2014-2024osm历史路网/2024/China_OSM_2024_3年路网"
    ccdc_dir: str = "/Volumes/T9/ccdc_res"
    disturbance_class_tif: str = "/Volumes/T9/Datasets/disturbance_class/disturbance_class_res.tif"
    # Optional patterns
    landuse_file_pattern: str = "CLCD_v01_{year}_albert_province/CLCD_v01_{year}_albert_xizang.tif"
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


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)

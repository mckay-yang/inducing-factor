from pathlib import Path
import os

import pytest
import rasterio
from rasterio.windows import Window

from inducing_factor.data.loaders import CCDCLoader, ClimateLoader, DEMLoader, LandUseLoader, VectorLoader


DATA_ROOT = Path("/Volumes/T9/Datasets")


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_DATA_TESTS") != "1",
    reason="set RUN_REAL_DATA_TESTS=1 to enable slow real-data tests",
)


def test_dem_loader():
    loader = DEMLoader(str(DATA_ROOT / "dem/yarlung_zangbo_downstream_dem.tif"))
    window = Window(1000, 1000, 100, 100)
    data, res, nodata = loader.load(window)
    slope, aspect = loader.calculate_slope_aspect(data, res[0], res[1], nodata)
    assert data.shape == slope.shape == aspect.shape


def test_other_loaders():
    dem_loader = DEMLoader(str(DATA_ROOT / "dem/yarlung_zangbo_downstream_dem.tif"))
    with rasterio.open(dem_loader.file_path) as src_dem:
        dem_crs = src_dem.crs
        dem_transform = src_dem.transform
        dem_width = src_dem.width
        dem_height = src_dem.height
    window = Window(1000, 1000, 100, 100)
    road_loader = VectorLoader(str(DATA_ROOT / "road/2014-2024osm历史路网/2024/China_OSM_2024_3年路网/gis_osm_roads_free_1.shp"))
    rain_loader = ClimateLoader(str(DATA_ROOT / "rainfall/【立方数据学社】西藏自治区"), file_pattern="pre{year}07.tif")
    temp_loader = ClimateLoader(str(DATA_ROOT / "temperature"), file_pattern="tmp_{year}.nc")
    landuse_loader = LandUseLoader(str(DATA_ROOT / "landuse/【立方数据学社】分省份的数据"), is_temporal=True, file_pattern="CLCD_v01_{year}_albert_province/CLCD_v01_{year}_albert_xizang.tif", data_type="Land use")
    ccdc_loader = CCDCLoader("/Volumes/T9/ccdc_res", file_pattern="{year}.tif")
    road_loader.load()
    assert rain_loader.load(2017, dem_crs, dem_transform, dem_width, dem_height, window) is not None
    assert temp_loader.load(2017, dem_crs, dem_transform, dem_width, dem_height, window) is not None
    assert landuse_loader.load(2017, dem_crs, dem_transform, dem_width, dem_height, window) is not None
    assert ccdc_loader.load(2017, dem_crs, dem_transform, dem_width, dem_height, window) is not None

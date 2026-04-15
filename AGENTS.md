# Drivers Repository - Ecosystem Disturbance Attribution System

## Critical Constraints
- **Data paths are hardcoded to `/Volumes/T9/Datasets/`** - local storage inaccessible to agents. Always use synthetic data or ask for alternative paths.
- Requires `pgmpy` for full Bayesian network functionality; falls back to logistic regression if missing (see `src/inducing_factor_system.py:643-670`).
- Full-extent processing (29k×38k pixels) uses 1024×1024 chunking to manage memory.
- `test_loaders.py` requires actual data files at hardcoded paths and will fail without them. `test_inducing_factor_system.py` uses synthetic data and is the safe entry point.

## Development Commands
```bash
# Run synthetic data test (safe for agents)
python test_inducing_factor_system.py

# Run loader tests (requires /Volumes/T9/Datasets/ access)
python test_loaders.py

# Run core attribution system (requires cached data)
python -c "from src.inducing_factor_system import EcoDisturbanceAttributionSystem; system = EcoDisturbanceAttributionSystem(); system.run()"

# Run processing pipeline (requires data access)
python src/pipeline.py
```

## Key Components
- `src/data_loader/`: Unified loaders for DEM, vector, climate, land use, and CCDC disturbance data
- `src/inducing_factor_system.py`: Core algorithm for inducing factor mining and Bayesian attribution
- `src/pipeline.py`: Chunked processing pipeline (1024×1024 tiles)
- `config_advanced.yaml`: Configuration for full-extent analysis (contains hardcoded paths)

## Environment & Dependencies
- Conda environment `geo-science` (see `.env`). For agents, `pip install -r requirements.txt` is sufficient.
- Key packages: `pgmpy` (Bayesian networks), `rasterio`, `geopandas`, `xarray`, `scikit-learn`

## Formal Data Sources
- **AOI**: `/Volumes/T9/Datasets/aoi/yarlung_zangbo_downstream_lang_to_paisighat_buffered_2km/LangToBasighat.shp`
- **DEM**: `/Volumes/T9/Datasets/dem/tibet/tibet_dem.tif`
- **Land Use**: `/Volumes/T9/Datasets/landuse/【立方数据学社】分省份的数据`
- **Rainfall**: `/Volumes/T9/Datasets/rainfall/【立方数据学社】西藏自治区` (monthly files `preYYYYMM.tif`)
- **Resident**: `/Volumes/T9/Datasets/resident/方创琳教授团队发布全球首套100 m城市和农村居住区数据集(2000-2020年)`
- **River**: `/Volumes/T9/Datasets/river/Global_Chimaps_shp_Data_Record_A` (use `As_15s_Chimaps.shp`)
- **Road**: `/Volumes/T9/Datasets/road/2014-2024osm历史路网/2024/China_OSM_2024_3年路网`
- **Temperature**: `/Volumes/T9/Datasets/temperature` (annual files `tmp_YYYY.nc`)
- **CCDC**: `/Volumes/T9/ccdc_res` (annual files `YYYY.tif`)

**Usage Rules**:
1. Only use data that intersects with the AOI (both raster chunks and vector features).
2. For global datasets (river, road), clip to AOI bounding box before processing.
3. Use the full time span of CCDC data (discover years automatically).
4. Disturbance points: use all. Background points: sample appropriately.
5. Precipitation: use monthly features and 3‑month total preceding disturbance month.
6. Road distance: compute true geometric shortest distance (not representative point).
7. Respect NoData values: DEM `-9999`, resident `127.0`, etc.

## Verification Steps
After changes:
1. Run `python test_inducing_factor_system.py` to ensure core logic works
2. Check for `pgmpy` import errors if modifying Bayesian networks
3. Verify discretization logic in `AdaptiveDiscretizer` if changing feature engineering
4. Ensure chunked processing maintains point_id uniqueness across chunks
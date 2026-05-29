# Drivers Repository - Ecosystem Disturbance Attribution System

## Critical Constraints
- **Data paths are now relative to `data/` directory** (previously hardcoded to `/Volumes/T9/Datasets/`). Ensure the `data/` directory contains required datasets.
- Requires `pgmpy` for full Bayesian network functionality; falls back to logistic regression if missing (see `src/inducing_factor/core/attribution_system.py`).
- Full-extent processing (29k×38k pixels) uses 1024×1024 chunking to manage memory.
- `tests/integration/test_loaders_real_data.py` requires actual data files at hardcoded paths and will fail without them. `tests/integration/test_synthetic_pipeline.py` uses synthetic data and is the safe entry point.

## Development Commands
```bash
# Run synthetic data test (safe for agents)
pytest tests/integration/test_synthetic_pipeline.py

# Run loader tests (requires data/ directory access)
pytest tests/integration/test_loaders_real_data.py

# Run core attribution system (requires cached data)
python -m inducing_factor.cli run --config configs/quick_test.yaml

# Run processing pipeline (requires data access)
python -m inducing_factor.cli pipeline --config configs/quick_multi.yaml
```

## Key Components
- `src/inducing_factor/data/loaders/`: Unified loaders for DEM, vector, climate, land use, and CCDC disturbance data
- `src/inducing_factor/core/attribution_system.py`: Core algorithm for inducing factor mining and Bayesian attribution
- `src/inducing_factor/pipeline/processor.py`: Chunked processing pipeline (1024×1024 tiles)
- `configs/production_fresh.yaml`: Configuration for full-extent analysis (contains hardcoded paths)

## Environment & Dependencies
- Conda environment `geo-science` (see `.env`). For agents, `pip install -r requirements.txt` is sufficient.
- Key packages: `pgmpy` (Bayesian networks), `rasterio`, `geopandas`, `xarray`, `scikit-learn`

## Formal Data Sources
- **AOI**: `data/aoi/LangToBasighat.shp`
- **DEM**: `data/dem/tibet_dem.tif`
- **Land Use**: `data/landuse`
- **Rainfall**: `data/rainfall` (monthly files `preYYYYMM.tif`)
- **Resident**: `data/resident`
- **River**: `data/river` (use `As_15s_Chimaps.shp`)
- **Road**: `data/road` (yearly files `osm_{year}_roads.shp`, `osm_{year}_railways.shp`, `osm_{year}_waterways.shp`)
- **Temperature**: `data/temperature` (annual files `tmp_YYYY.nc`)
- **CCDC**: `data/ccdc_res` (annual files `YYYY.tif`)

## Data Architecture
```
data/                   # input datasets only
cache/chunks/           # chunk NPZ files
results/models/         # model artifacts
tmp/                    # temp scripts
src/inducing_factor/    # Python source code
```

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
1. Run `pytest tests/integration/test_synthetic_pipeline.py` to ensure core logic works
2. Check for `pgmpy` import errors if modifying Bayesian networks
3. Verify discretization logic in `AdaptiveDiscretizer` if changing feature engineering
4. Ensure chunked processing maintains point_id uniqueness across chunks

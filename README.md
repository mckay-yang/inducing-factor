# Inducing Factor Attribution

Rust workflow for CLASS-grid disturbance probability prediction, inducing-factor interaction mining, and geospatial output products.

The current production path is Rust-first:

```text
data-process -> train -> predict
```

The pipeline aligns all source data to the CLASS raster grid, builds cached feature tables, trains a continuous numeric model with interaction terms, then writes GeoTIFF prediction products.

## Repository Layout

```text
configs/              Workflow configuration files
crates/               Rust workspace crates
data/input/           Input datasets only
data/cache/           Long-running intermediate caches
data/result/          Model, table, and prediction outputs
tests/                Unit/integration fixtures and smoke tests
```

`data/*` is ignored by git. Do not put source code, docs, or tests under `data/`.

## Data Layout

The active workflow expects this directory contract:

| Path | Meaning |
|---|---|
| `data/input/` | Raw or prepared input data used by the workflow. |
| `data/cache/` | Generated caches: resampled rasters, distance rasters, feature parquet tiles, mmap point caches, interaction caches. |
| `data/result/` | Final outputs: trained model JSON, factor tables, diagnostics, prediction GeoTIFFs. |
| `tests/data/` | Small test fixtures only. |

Large outputs and temporary test products should go to system temp directories such as `/tmp`, not to the repository root.

## Main Workflow

The active config is:

```text
configs/class_basin_workflow.json
```

Useful commands:

```bash
# Build the CLI
cargo build -p if-cli

# Inspect the plan without running the heavy stages
cargo run -p if-cli -- plan-workflow --config configs/class_basin_workflow.json

# Run the full workflow
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json

# Reuse cached data-process outputs and rerun training/prediction
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json --skip-data-process

# Refresh only climate-derived feature caches and downstream model outputs
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json --refresh-cache climate-features

# Refresh only model outputs, interaction caches, tables, and prediction rasters
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json --refresh-cache model-outputs
```

If using the already-built binary:

```bash
target/debug/if-cli run-workflow --config configs/class_basin_workflow.json
```

## Workflow Stages

### 1. Data Processing

`data-process` reads inputs from `data/input/` and writes reusable artifacts under `data/cache/`.

It performs:

- CLASS-grid alignment for DEM, CCDC, rainfall, temperature, landuse, and resident rasters.
- AOI-aware distance raster generation for river, road, railway, waterway, urban settlement, and rural settlement factors.
- Training point generation from CLASS/CCDC disturbance labels.
- Training feature extraction to `data/cache/features/training_features.parquet`.
- Prediction feature extraction to tiled parquet files under `data/cache/features/prediction_YYYY/`.

Distance factors are produced by rasterizing source geometries or source masks inside a 20 km buffered AOI, running a raster distance transform, then writing distances back on the original CLASS grid. If a source is missing, the distance raster is filled with a large sentinel distance so downstream numeric models can still run.

### 2. Training

`train-workflow` reads `data/cache/features/training_features.parquet` and writes model artifacts under `data/result/models/` plus ranking tables under `data/result/tables/`.

The current occurrence model is continuous numeric logistic regression with continuous interaction terms:

```text
P(y = 1 | x) = sigmoid(b + sum_j w_j z_j + sum_S w_S A_i(S))
```

where:

```text
z_j      = standardized numeric feature
a_j      = sigmoid(gamma * sign(w_baseline_j) * z_j)
A_i(S)   = product of a_j for all features in interaction S
```

Interactions are mined with Apriori/beam-style pruning:

- Order 1 and order 2 statistics are computed directly.
- Strong pairs are selected by support, lift, and score.
- Order 3 candidates are generated only from strong pairs.
- Order 4 candidates are generated only from strong triples when enabled.
- Climate-only and mixed climate combinations are down-weighted so water/heat factors do not dominate the explanation layer by themselves.

The model keeps continuous values for prediction. Learned thresholds and bins are used only for explanation text, legends, and tables.

### 3. Prediction

`predict-workflow` reads:

- model artifact: `data/result/models/basin_model.json`
- prediction feature tiles: `data/cache/features/prediction_YYYY/*.parquet`

It writes requested GeoTIFF products under `data/result/predictions/`.

The probability raster is computed from the full numeric model and interaction terms. The dominant-factor rasters are explanatory products: each pixel stores the ID of the largest positive model contribution. They are not causal attribution maps.

## Core Features

The current feature table includes:

- topography: `elevation`, `slope`, `aspect`
- categorical/land surface: `landuse`, `resident`
- distances: `dist_to_river`, `dist_to_road`, `dist_to_railway`, `dist_to_waterway`, `dist_to_urban_settlement`, `dist_to_rural_settlement`
- climate windows: `rainfall_avg_prev_1m`, `rainfall_avg_prev_3m`, `rainfall_avg_prev_6m`, `rainfall_avg_prev_12m`, `temp_avg_prev_1m`, `temp_avg_prev_3m`, `temp_avg_prev_6m`, `temp_avg_prev_12m`

Training climate windows are computed relative to the CCDC timing field. Prediction does not have observed disturbance timing, so prediction-time climate windows use the configured fallback/reference behavior recorded in `data/result/models/prediction_feature_assumptions.json`.

Temperature source values are stored as tenths of degrees Celsius. A raw value of `253` means `25.3 deg C`.

## Output Files

Key model and table outputs:

| File | Meaning |
|---|---|
| `data/result/models/basin_model.json` | Main model artifact consumed by prediction. |
| `data/result/models/interaction_factors.json` | Ranked mined interaction terms with support/lift/score/model weight. |
| `data/result/tables/interaction_factor_ranking.csv` | CSV version of interaction ranking. |
| `data/result/models/learned_thresholds.json` | Sample-learned per-feature thresholds and bins for explanation. |
| `data/result/tables/learned_thresholds.csv` | CSV threshold summary. |
| `data/result/models/dominant_factor_mapping.json` | ID mapping for dominant-factor rasters. |
| `data/result/models/disturbance_type_diagnostics.json` | Class sample counts, imbalance warnings, and type-probability semantics. |
| `data/result/models/prediction_feature_assumptions.json` | Prediction-time feature assumptions and fallback notes. |

Key prediction outputs:

| File | Meaning |
|---|---|
| `data/result/predictions/probability_YYYY.tif` | Disturbance probability. |
| `data/result/predictions/dominant_factor_YYYY.tif` | Largest positive occurrence-model contribution ID. |
| `data/result/predictions/disturbance_type_YYYY.tif` | Most likely disturbance type, if a type model is trained. |
| `data/result/predictions/disturbance_type_*_probability_YYYY.tif` | Joint probability for each type. |
| `data/result/predictions/disturbance_type_*_dominant_factor_YYYY.tif` | Type-specific largest positive contribution ID. |
| `data/result/predictions/missing_data_mask_YYYY.tif` | Pixels with missing feature values. |

Type probabilities are joint probabilities:

```text
P(type = k | x) = P(disturbance | x) * P(type = k | disturbance, x)
```

They depend on the quality and balance of the `disturbance_class` labels.

## Rust Workspace

| Crate | Role |
|---|---|
| `crates/cli` | `if-cli` command-line interface and workflow orchestration. |
| `crates/workflow-config` | JSON config schema and validation. |
| `crates/data-process` | CLASS-grid preprocessing, distance rasters, and feature parquet generation. |
| `crates/train` | Numeric occurrence model, interaction mining, thresholds, type model, diagnostics. |
| `crates/predict` | Tiled inference and GeoTIFF writers. |
| `crates/model` | Shared model artifact schema. |
| `crates/factor-core` | Core statistical/modeling structures and algorithms. |
| `crates/gdal` | Geospatial grid and GDAL helpers. |
| `crates/loaders` | Lower-level loader utilities. |
| `crates/artifacts`, `crates/golden-tests` | Artifact/test support. |

## Environment

This project links to GDAL, HDF5, and NetCDF. The current local `.cargo/config.toml` points to a macOS miniforge environment:

```text
/opt/homebrew/Caskroom/miniforge/base/envs/geo-science
```

If your GDAL environment differs, update local environment variables or `.cargo/config.toml` accordingly. The repository does not vendor GDAL binaries.

## Verification

Lightweight checks:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test -p data-process --lib
cargo test -p train --lib
cargo test -p predict --lib
cargo test -p if-cli --bins
cargo run -p if-cli -- plan-workflow --config configs/class_basin_workflow.json
```

Some tests and workflow stages require GDAL plus real input data. Full production runs are long and should reuse `data/cache/` whenever possible.

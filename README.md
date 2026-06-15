# Inducing Factor Attribution - Rust Workspace

Rust-native ecosystem disturbance attribution system for CLASS/CCDC-grid
preprocessing, inducing-factor training, and geospatial prediction products.

The current architecture is a Cargo workspace. The production flow is:

```text
data-process -> train -> predict
```

The stages are integrated by `cli`, but the stage crates are kept independent.

## Quick Start

```bash
# Build all crates
cargo build --workspace

# Inspect the CLASS-based workflow plan
cargo run -p if-cli -- plan-workflow --config configs/class_basin_workflow.json

# Run individual stages
cargo run -p if-cli -- data-process --config configs/class_basin_workflow.json
cargo run -p if-cli -- train-workflow --config configs/class_basin_workflow.json
cargo run -p if-cli -- predict-workflow --config configs/class_basin_workflow.json

# Run the full workflow
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json
```

## Command Reference

| Command | Purpose |
|---|---|
| `plan-workflow` | Opens the CLASS raster, expands years, and prints the tile/train/predict plan. |
| `data-process` | Resamples source rasters/vectors to the CLASS grid and writes training/prediction feature parquet files. |
| `train-workflow` | Trains the basin model from cached training features and writes `basin_model.json`. |
| `predict-workflow` | Reads the model and prediction feature tiles and writes configured raster products. |
| `run-workflow` | Runs data-process, train, and predict in sequence. |

`run-workflow` accepts `--skip-data-process` when cached features are already present.

## Configuration

The active workflow config is JSON. See
`configs/class_basin_workflow.json`.

Important fields:

| Field | Meaning |
|---|---|
| `class_grid.raster` | CLASS/CCDC-class raster used as the canonical grid. |
| `class_grid.start_year` | First band year for the CLASS raster. |
| `train.training_years` | Explicit list of years used for training, for example `[2017, ..., 2024]`. |
| `predict.prediction_year` | Future or held-out year to predict. |
| `data_process.temporal_context_years` | Expands data processing years around train/predict years for temporal covariates. |
| `*.workers` | Per-stage worker count. `0` means auto-detect available parallelism. |
| `predict.products` | Output products such as `probability`, `dominant_factor`, and `missing_data_mask`. |

## Crate Structure

| Crate | Role |
|---|---|
| `gdal` | Independent geospatial grid primitives and GDAL boundary helpers. |
| `model` | Model artifact schema and JSON read/write boundary shared by train and predict. |
| `data-process` | CLASS-grid preprocessing, resampling, distance rasters, training points, and feature parquet generation. |
| `train` | Inducing-factor mining and basin model training. |
| `predict` | Model inference and geospatial raster product writers. |
| `cli` | Workflow orchestration and JSON config mapping. |
| `factor-core` | Statistical and inducing-factor algorithms used by `model` and `train/predict`. |
| `loaders` | Low-level data loaders used by preprocessing. |
| `workflow-config` | JSON workflow config types. |
| `artifacts`, `golden-tests` | Test/artifact support. |

Dependency direction is intentional:

```text
gdal        model
  ^          ^
  |          |
data-process train predict
       \      |     /
              cli
```

`data-process`, `train`, and `predict` do not depend on each other.

## Geospatial Rules

- CLASS is the canonical raster grid for preprocessing and prediction output.
- Resampling uses CLASS-derived geospatial bounds, CRS, dimensions, and transform.
- Vector distance rasters are computed from geometry in grid coordinates, not representative points.
- Prediction products inherit the CLASS geotransform and CRS.
- Full extent is processed in tiles, typically `1024x1024`, to control memory.

## Data Units

- Temperature source files `data/input/temperature/tmp_YYYY.nc` store values in
  tenths of degrees Celsius (`0.1 deg C`). A raw value of `253` means
  `25.3 deg C`.
- Settlement rasters use categorical values `1=urban settlement` and
  `2=rural settlement`; the workflow converts them into separate distance
  factors. See `docs/settlement_distance_factors.md`.

## Missing Data

The workflow is designed to keep prediction possible when covariates are missing.
Feature extraction writes missing numeric inputs as `NaN`; training can use the
`native_missing_category` policy so discretization learns an explicit missing
bucket. Prediction also emits `missing_data_mask` when requested.

## Verification

Safe checks that do not require the full production dataset:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test -p data-process --lib
cargo test -p train --lib
cargo test -p predict --lib
cargo test -p if-cli --bins
cargo run -p if-cli -- plan-workflow --config configs/class_basin_workflow.json
```

Some GDAL-backed tests skip themselves when command-line GDAL tools are not
available.

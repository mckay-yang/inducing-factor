# Inducing Factor Attribution — Rust Implementation

A Rust-native implementation of the ecosystem disturbance attribution system. This is a complete rewrite of the Python `inducing_factor` package, providing equivalent functionality through a CLI and library crates.

## Quick Start

```bash
# Build all crates
cargo build --release

# Show available commands
cargo run -p if-cli -- --help

# Run the full pipeline end-to-end with synthetic data
cargo run -p if-cli -- all --config configs/example.yaml

# Run individual stages
cargo run -p if-cli -- pipeline --config configs/example.yaml
cargo run -p if-cli -- train --config configs/example.yaml
cargo run -p if-cli -- predict --config configs/example.yaml --year 2016

# Validate config without processing
cargo run -p if-cli -- pipeline --config configs/example.yaml --dry-run

# Override worker count
cargo run -p if-cli -- pipeline --config configs/example.yaml --workers 4
```

## Command Reference

### `pipeline` — Feature Extraction

Runs the chunked feature extraction pipeline. Reads geospatial data sources, clips to AOI, and writes chunked parquet files to the cache directory.

```bash
if-cli pipeline --config <path> [--workers <n>] [--dry-run]
```

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML configuration file (required) |
| `--workers <n>` | Override worker count from config |
| `--dry-run` | Validate config and show chunk plan without processing |

### `train` — Model Training

Trains the attribution model on pipeline cache output. Fits the AdaptiveDiscretizer, mines inducing factor candidates, selects top factors, and trains occurrence and class probability models. Saves all artifacts to the output directory.

```bash
if-cli train --config <path> [--output-dir <path>]
```

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML configuration file (required) |
| `--output-dir <path>` | Override output directory for model artifacts |

### `predict` — Probability Maps

Generates annual disturbance probability maps. Reads chunked feature tables, runs model inference with learned factors, and writes probability rasters.

```bash
if-cli predict --config <path> [--year <year>] [--workers <n>]
```

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML configuration file (required) |
| `--year <year>` | Target year for prediction (default: latest from config) |
| `--workers <n>` | Override worker count from config |

### `all` — End-to-End

Runs pipeline → train → predict in sequence.

```bash
if-cli all --config <path> [--dry-run]
```

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML configuration file (required) |
| `--dry-run` | Validate config only, no processing |

## Configuration

See `configs/example.yaml` for a complete configuration reference with all available options:

- **paths**: `cache_dir`, `output_dir`, `data_dir`
- **chunks**: `tile_size`, `aoi_shapefile`
- **pipeline**: `workers` (auto or integer), `memory_limit_gb`, `start_year`, `end_year`
- **attribution**: `lookback_years`, `background_sample_rate`, `max_pattern_size`, `min_support`, `min_relative_risk`, `bootstrap_iterations`, `min_stability`, `max_bootstrap_samples`, `max_network_factors`, `discretizer_method`, `discretize_bins`, `random_state`, `dominant_threshold`, `synergy_threshold`
- **data_sources**: paths and file patterns for DEM, landuse, rainfall, temperature, resident, river, road, CCDC, and disturbance class data

## Logging

All commands use structured logging via `tracing` with the format:

```
[stage][phase n/total] description progress/total (percent%) elapsed=Xms eta=Ys
```

Set the `RUST_LOG` environment variable to control verbosity:

```bash
RUST_LOG=if_cli=debug cargo run -p if-cli -- pipeline --config configs/example.yaml
```

Progress bars are displayed during chunk processing via `indicatif`.

## Performance Notes

- **Worker count**: Defaults to all logical CPUs (`workers: "auto"`). Override with `--workers N` or config. Memory scales linearly with workers.
- **Chunk size**: Default 1024×1024 pixels. Larger chunks use more memory; smaller chunks increase overhead.
- **Synthetic mode**: When no real data is available, the pipeline generates synthetic data for testing. This is the default behavior when cache directories are empty.
- **Memory**: Full-extent processing (29k×38k pixels) uses 1024×1024 chunking to manage memory. Each chunk requires ~50-200MB depending on feature count.

## Crate Structure

| Crate | Description |
|-------|-------------|
| `if-cli` | CLI binary with `pipeline`, `train`, `predict`, `all` commands |
| `if-config` | Typed configuration model with YAML deserialization |
| `if-core` | Core algorithms: discretization, factor mining, logistic regression, Bayesian networks |
| `if-pipeline` | Chunked processing pipeline, feature builder, probability mapper |
| `if-loaders` | Data loaders for DEM, vector, climate, land use, CCDC (requires GDAL/netcdf features) |
| `if-io` | Artifact serialization and golden test fixtures |
| `if-tests` | Shared golden baseline test utilities |

## Relation to Python Project

This is a Rust-native rewrite of the Python `inducing_factor` package. The Rust implementation provides:

- Equivalent statistical results (golden baseline tests verify this)
- Better memory management via chunked processing
- Structured logging with progress bars
- Type-safe configuration via serde
- No Python runtime dependency

The Python codebase remains in the main branch and is not modified by this worktree.

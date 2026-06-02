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

- **Worker count**: Defaults to all logical CPUs (`workers: "auto"` or `workers: 0`). Override with `--workers N` or config. Memory scales linearly with workers.
- **Chunk size**: Default 1024×1024 pixels. Larger chunks use more memory; smaller chunks increase overhead.
- **Synthetic mode**: When no real data is available, the pipeline generates synthetic data for testing. This is the default behavior when cache directories are empty.
- **Memory**: Full-extent processing (29k×38k pixels) uses 1024×1024 chunking to manage memory. Each chunk requires ~0.3 GB for 1024×1024, ~2.5 GB for 4096×4096 per worker.
- **Memory guardrail**: When `memory_limit_gb` is set, or system RAM is detected, workers are automatically reduced if the requested count would exceed available memory. Set `estimated_worker_memory_gb` to override auto-estimation.

See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks, scaling charts, and known bottlenecks.

## Production Runs

### Real Data Testing

Real-data loader tests are gated behind the `real-data` cargo feature and the `RUN_REAL_DATA_TESTS` environment variable:

```bash
RUN_REAL_DATA_TESTS=1 cargo test -p if-loaders --features real-data,gdal
```

Tests gracefully skip when data files are not found, allowing them to run in partial-data environments.

### Memory Tuning

| Config parameter | Purpose | Recommended |
|---|---|---|
| `pipeline.memory_limit_gb` | Hard cap on total pipeline RAM | 80% of system RAM |
| `pipeline.memory_fraction` | Fraction of system RAM to use (default 0.7) | 0.6–0.8 |
| `pipeline.estimated_worker_memory_gb` | Override per-worker estimate | Only if auto-estimate is wrong |
| `chunks.tile_size` | Chunk dimensions in pixels | 1024 for most systems, 2048 for >64 GB |

**Example production config excerpt:**

```yaml
pipeline:
  workers: "auto"          # or 0 — auto-select based on CPU + memory
  memory_limit_gb: 28      # hard limit for 32 GB system
  memory_fraction: 0.7     # use ≤ 70% of available RAM
  start_year: 2014
  end_year: 2024

chunks:
  tile_size: 1024          # 1024×1024 per chunk
```

### Running Without Real Data

Default tests never require real data. The full build:

```bash
cargo test --workspace     # 164 tests, 0 failures, no data/ required
cargo build --release      # release binary
```

### Historical Issue: Python OOM at 16 Workers

The Python pipeline OOM'd at 16 workers on a 24 GB system because each worker held a full pandas DataFrame copy. The Rust rewrite addresses this by:

1. **Memory guardrail** — auto-reduces worker count when memory budget is exceeded
2. **Zero-copy reads** — Polars/pyarrow columnar format avoids per-worker DataFrame duplication
3. **Conservative estimation** — `estimate_worker_memory_gb` uses a 2.5× working-copy factor
4. **Graceful degradation** — workers are capped, not killed; a warning is logged

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

## Corrected Prediction Semantics (2026-06)

The original Python code contained a **reference_month leakage bug**: rainfall window
features were computed from `disturbance_month` (a CCDC-derived label column) even at
prediction time. This made the model appear more accurate than it actually was on
held-out data.

**Rust corrections applied** (matching the now-corrected Python reference):

| Component | Correction |
|-----------|-----------|
| `feature_builder.rs` | `predict_month: Option<u32>` parameter; when `Some`, uses explicit month instead of requiring `reference_month` from CCDC labels |
| `ensemble.rs` (new) | `predict_ensemble()` iterates months 1-12, aggregates mean/max/peak_month — leakage-free future-year prediction |
| `probability_mapper.rs` | `predict_chunk_ensemble()` wraps ensemble for raster-map generation |

**Prediction flow:**
1. **Training mode** (`predict_month=None`): uses per-row `reference_month` from CCDC
   `disturbance_month` column (valid for historical labeled data)
2. **Prediction mode** (`predict_month=Some(m)`): uses explicit uniform month, does NOT
   require CCDC columns — pure covariate-based prediction
3. **Ensemble mode**: runs prediction for all 12 months, aggregates into ensemble
   statistics (`disturbance_probability_mean`, `_max`, `_peak_month`)

**Golden baselines** must be regenerated from the corrected Python pipeline
(`scripts/generate_golden_fixtures.py`) after re-running with the updated code.

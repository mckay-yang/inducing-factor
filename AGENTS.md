# Agent Guide

This file is the concise operating guide for AI/code agents working in this repository. Read the code before changing behavior; README-level descriptions may lag behind implementation.

## Current Project Shape

- The active workflow is Rust-first and centered on `if-cli`.
- The main config is `configs/class_basin_workflow.json`.
- Runtime data is organized as:
  - `data/input/` for input datasets.
  - `data/cache/` for generated intermediate caches.
  - `data/result/` for final models, tables, and prediction rasters.
  - `tests/data/` for small test fixtures only.
- `data/*` is ignored by git and may contain expensive outputs. Do not delete or rewrite it unless the user explicitly asks.

## First Files To Read

For workflow behavior:

- `crates/cli/src/main.rs`
- `crates/cli/src/commands.rs`
- `crates/workflow-config/src/workflow.rs`
- `configs/class_basin_workflow.json`

For feature generation:

- `crates/data-process/src/lib.rs`
- `crates/data-process/src/feature_extraction.rs`
- `crates/data-process/src/raster_distance.rs`

For model math:

- `crates/train/src/lib.rs`
- `crates/factor-core/src/model.rs`
- `crates/model/src/lib.rs`

For prediction outputs:

- `crates/predict/src/lib.rs`

## Data And Cache Rules

- Preserve `data/cache/features/training_features.parquet` and `data/cache/features/prediction_*/` unless the user clearly requests a refresh.
- Prefer `--refresh-cache model-outputs` when only retraining or rerendering outputs is needed.
- Use `--refresh-cache climate-features` only when climate-derived feature values must be regenerated.
- Do not put temporary test outputs under the repo root. Use `/tmp` or the OS temp directory.
- Do not commit data, rasters, parquet caches, logs, or model outputs unless the user explicitly changes the repository policy.

## Commands

Build and inspect:

```bash
cargo build -p if-cli
cargo run -p if-cli -- plan-workflow --config configs/class_basin_workflow.json
```

Run stages:

```bash
cargo run -p if-cli -- data-process --config configs/class_basin_workflow.json
cargo run -p if-cli -- train-workflow --config configs/class_basin_workflow.json
cargo run -p if-cli -- predict-workflow --config configs/class_basin_workflow.json
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json
```

Cache-aware reruns:

```bash
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json --skip-data-process
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json --refresh-cache model-outputs
cargo run -p if-cli -- run-workflow --config configs/class_basin_workflow.json --refresh-cache climate-features
```

Verification:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test -p data-process --lib
cargo test -p train --lib
cargo test -p predict --lib
cargo test -p if-cli --bins
```

## Modeling Semantics

- The current occurrence model is continuous numeric logistic regression with continuous interaction terms.
- Interaction mining uses fuzzy risk-side activations and Apriori/beam pruning.
- Learned thresholds and bins are for explanation only; they do not replace continuous model inputs.
- Dominant-factor outputs are largest positive model-contribution IDs. They are not causal claims.
- Disturbance type probability is joint probability:

```text
P(type=k | x) = P(disturbance | x) * P(type=k | disturbance, x)
```

- Type outputs depend on `disturbance_class` label quality. Check `data/result/models/disturbance_type_diagnostics.json` before interpreting them strongly.

## Development Style

- Use `rg` for searches.
- Use `apply_patch` for manual file edits.
- Keep changes scoped to the user request.
- Do not revert user changes or manually deleted files unless explicitly asked.
- If the user asks for an explanation of current behavior, inspect source code first and answer from code.
- When reporting long-running workflow progress, use log counters such as `tasks completed / total`, `tiles`, `pixels`, or batch percentages instead of guessing.

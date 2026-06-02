# Golden Baseline Fixtures

Generated from the Python `inducing_factor` reference implementation using
**synthetic data only** (no real data dependencies).

## Regeneration

```bash
cd /Volumes/T7/inducing_factor
conda activate geo-science
python scripts/generate_golden_fixtures.py
```

All fixtures are **deterministic** — same seed (42), same output every run.

## Source Commit

`f8953b2324700c30748c2a3c17b22420dab1c559`

## Generated At

`2026-06-01T14:39:35.521485+00:00`

## File Index

| File | Description |
|------|-------------|
| `MANIFEST.json` | Generation manifest with commit hash, tolerances, file listing |
| `README.md` | This file |
| `attribution_preview.parquet` | Sample attribution results (top factors, probabilities) |
| `bayesian_network_edges.json` | Consensus DAG edge list from Bayesian network structure learning |
| `discretization_labels.json` | Feature discretization meta: bins, labels, feature kinds |
| `edge_all_nodata.parquet` | Edge case: all-NaN feature values |
| `edge_empty_chunk.parquet` | Edge case: empty feature frame (0 rows) |
| `edge_single_class.parquet` | Edge case: single disturbance class (all class=4) |
| `edge_single_row.parquet` | Edge case: single row input |
| `factor_class_summary.parquet` | Factor-class association table (chi-square, frequencies) |
| `factor_library.json` | Full inducing factor library with statistics and metadata |
| `model_metadata.json` | Model metadata: feature columns, selected factors, backend info |
| `model_predictions.parquet` | Occurrence and class probabilities on sample rows |
| `selected_factors.json` | Selected inducing factors used in the Bayesian network |
| `synthetic_panel_chunk_0.parquet` | Raw synthetic panel data, chunk 0 (first half) |
| `synthetic_panel_chunk_1.parquet` | Raw synthetic panel data, chunk 1 (second half) |
| `synthetic_panel_full.parquet` | Full raw synthetic panel (all rows) |
| `training_samples.parquet` | Training samples used for model fitting (event + background) |

## Tolerance Reference

| Fixture Category | Tolerance |
|-----------------|-----------|
| Discretization labels | Exact match |
| Statistical metrics (RR, CI) | Absolute ≤ 1e-6, Relative ≤ 1e-4 |
| Factor rankings | Spearman ρ ≥ 0.95 |
| Occurrence probabilities | MAE ≤ 0.03, max error ≤ 0.10 |
| Class predictions (top-1) | Match rate ≥ 0.95 |
| Raster metadata | Exact match |

## Notes

- **elevation** is intentionally excluded from factor mining (set as metadata).
- `dist_to_road`, `dist_to_railway`, `dist_to_waterway` are dynamic (yearly) features.
- Factor naming uses semantic labels: `dist_to_road=near`, `slope=high`, etc.

## Regeneration After Corrected Semantics

The golden baselines were originally generated from Python code that contained a
**reference_month leakage bug**: the `build_rainfall_window_features` function derived
per-row reference months from `disturbance_month` (a CCDC-derived label column) even
at prediction time, leaking future information into features.

The corrected Python pipeline now supports **prediction mode** with explicit
`predict_month` that does NOT require CCDC columns and a **12-month ensemble**
(`predict_ensemble`) for future-year prediction. After re-running the corrected
Python pipeline, golden fixtures MUST be regenerated:

```bash
cd /Volumes/T7/inducing_factor
python scripts/generate_golden_fixtures.py
```

Key changes that may shift baseline values:
- **Temporal split**: training/validation now split by year (not random)
- **Feature semantics**: `predict_month=None` for training (uses `reference_month`
  from CCDC labels), `predict_month=Some(m)` for prediction (leakage-free)
- **Ensemble prediction**: `predict_ensemble` runs 12-month iteration, aggregating
  into `disturbance_probability_mean`, `disturbance_probability_max`,
  `disturbance_probability_peak_month`
- The discretizer uses `supervised_tree` method with decision tree binning.
- Background sampling uses stratified sampling on static features.

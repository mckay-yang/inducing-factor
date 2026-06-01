#!/usr/bin/env python
"""
Generate golden baseline fixtures for Rust rewrite verification.

Runs the full synthetic data pipeline with deterministic random_state=42,
captures all intermediate and final outputs, and exports them to
`.worktrees/rust-rewrite/tests/golden/`.

DOES NOT MODIFY cache/, results/, or tmp/ in the main tree.
All generation uses temp directories.

Usage:
    python scripts/generate_golden_fixtures.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

GOLDEN_DIR = PROJECT_ROOT / ".worktrees" / "rust-rewrite" / "tests" / "golden"
SEED = 42
RNG = np.random.default_rng(SEED)


def _serialize(obj):
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_serialize(v) for v in obj.tolist()]
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if pd.isna(obj):
        return None
    return obj


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize(data), f, ensure_ascii=False, indent=2)


TABULAR_FORMAT = "parquet"

def write_tabular(path, df: pd.DataFrame):
    global TABULAR_FORMAT
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path, index=False)
        TABULAR_FORMAT = "parquet"
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        TABULAR_FORMAT = "csv"
        print(f"      (parquet unavailable, wrote CSV: {csv_path.name})")
        return csv_path
    return path


# ── step 1: generate synthetic data ───────────────────────────────
from inducing_factor.utils.synthetic import build_synthetic_panel, save_as_npz_chunks
from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem

print("=" * 60)
print("[1/8] Generating synthetic data panel ...")
panel = build_synthetic_panel(random_state=SEED)
print(f"      panel: {panel.shape[0]} rows x {panel.shape[1]} cols")

# ── step 2: run pipeline with temp dirs ───────────────────────────
print("[2/8] Running EcoDisturbanceAttributionSystem ...")
with tempfile.TemporaryDirectory(prefix="golden_cache_") as cache_dir, \
     tempfile.TemporaryDirectory(prefix="golden_out_") as out_dir:

    save_as_npz_chunks(panel, cache_dir)
    print(f"      wrote NPZ chunks to {cache_dir}")

    system = EcoDisturbanceAttributionSystem(
        cache_dir=cache_dir,
        out_dir=out_dir,
        lookback_years=2,
        background_sample_rate=0.3,
        max_pattern_size=3,
        min_support=0.08,
        min_relative_risk=1.2,
        bootstrap_iterations=5,
        min_stability=0.4,
        max_bootstrap_samples=1000,
        max_network_factors=6,
        random_state=SEED,
    )
    result = system.run()
    print(f"      factors found: {len(result['factors'])}")
    print(f"      selected factors: {len(system.selected_factors)}")
    print(f"      model backend: {system.model_backend}")

    # ── step 3: export core artifacts as golden ───────────────────
    print("[3/8] Exporting core golden artifacts ...")
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    # 3a: discretizer feature_meta_ → JSON
    meta = {}
    for feat in system.discretizer.features_:
        info = system.discretizer.feature_meta_[feat].copy()
        meta[feat] = _serialize(info)
    write_json(GOLDEN_DIR / "discretization_labels.json", meta)
    print(f"      → discretization_labels.json  ({len(meta)} features)")

    # 3b: factor library → JSON (detailed)
    factors_list = []
    for factor in system.factor_library:
        d = {
            "factor_id": factor.factor_id,
            "name": factor.name,
            "semantics": factor.semantics,
            "items": list(factor.items),
            "support_event": factor.support_event,
            "support_background": factor.support_background,
            "relative_risk": factor.relative_risk,
            "stability": factor.stability,
            "score": factor.score,
            "count_event": factor.count_event,
            "count_background": factor.count_background,
            "n_event": factor.n_event,
            "n_background": factor.n_background,
            "rr_ci_lower": factor.rr_ci_lower,
            "rr_ci_upper": factor.rr_ci_upper,
            "p_value": factor.p_value,
            "fdr_significant": factor.fdr_significant,
        }
        factors_list.append(d)
    write_json(GOLDEN_DIR / "factor_library.json", factors_list)
    print(f"      → factor_library.json  ({len(factors_list)} factors)")

    # 3c: selected factors → JSON
    selected = []
    for factor in system.selected_factors:
        d = {
            "factor_id": factor.factor_id,
            "name": factor.name,
            "semantics": factor.semantics,
            "items": list(factor.items),
            "relative_risk": factor.relative_risk,
            "stability": factor.stability,
            "score": factor.score,
        }
        selected.append(d)
    write_json(GOLDEN_DIR / "selected_factors.json", {"factors": selected, "count": len(selected)})
    print(f"      → selected_factors.json  ({len(selected)} selected)")

    # 3d: Bayesian network edges → JSON
    edges_path = os.path.join(out_dir, "bayesian_network_edges.json")
    if os.path.exists(edges_path):
        import shutil
        shutil.copy(edges_path, GOLDEN_DIR / "bayesian_network_edges.json")
        print("      → bayesian_network_edges.json")
    else:
        # Fallback: capture from model
        if system.model is not None:
            edges = getattr(system.model, "edges", [])
            write_json(GOLDEN_DIR / "bayesian_network_edges.json", edges)
            print(f"      → bayesian_network_edges.json  (fallback, {len(edges)} edges)")
        else:
            write_json(GOLDEN_DIR / "bayesian_network_edges.json", [])
            print("      → bayesian_network_edges.json  (empty)")

    # 3e: training samples → Parquet
    train_samples = result["training_samples"]
    write_tabular(GOLDEN_DIR / "training_samples.parquet", train_samples)
    print(f"      → training_samples.parquet  ({len(train_samples)} rows)")

    # 3f: factor-class summary → Parquet
    fc_summary = result["factor_class_summary"]
    if fc_summary is not None and not fc_summary.empty:
        write_tabular(GOLDEN_DIR / "factor_class_summary.parquet", fc_summary)
        print(f"      → factor_class_summary.parquet  ({len(fc_summary)} rows)")
    else:
        print("      → factor_class_summary.parquet  (empty, skipped)")

    # 3g: model metadata → JSON
    meta_json_path = os.path.join(out_dir, "model_metadata.json")
    if os.path.exists(meta_json_path):
        import shutil
        shutil.copy(meta_json_path, GOLDEN_DIR / "model_metadata.json")
        print("      → model_metadata.json")
    else:
        meta = {
            "feature_columns": system.feature_columns,
            "network_feature_columns": system.network_feature_columns,
            "selected_factor_ids": [f.factor_id for f in system.selected_factors],
            "target": system.target,
            "model_backend": system.model_backend,
        }
        write_json(GOLDEN_DIR / "model_metadata.json", meta)
        print("      → model_metadata.json  (generated)")

    # ── step 4: model predictions on training samples ─────────────
    print("[4/8] Generating model predictions ...")
    pred_samples = train_samples.head(min(50, len(train_samples)))
    preds = system.predict_dataframe(pred_samples)
    write_tabular(GOLDEN_DIR / "model_predictions.parquet", preds)
    print(f"      → model_predictions.parquet  ({len(preds)} rows)")

    # ── step 5: attribution preview ───────────────────────────────
    print("[5/8] Generating attribution preview ...")
    attr = system.attribute_dataframe(pred_samples, limit=None)
    write_tabular(GOLDEN_DIR / "attribution_preview.parquet", attr)
    print(f"      → attribution_preview.parquet  ({len(attr)} rows)")

    # ── step 6: edge case data ────────────────────────────────────
    print("[6/8] Generating edge case fixtures ...")

    # Use training_samples schema (includes derived features) for edge cases
    train_columns = list(train_samples.columns)

    # 6a: empty chunk (0 rows, full schema)
    empty_df = pd.DataFrame(columns=train_columns)
    write_tabular(GOLDEN_DIR / "edge_empty_chunk.parquet", empty_df)

    # 6b: single-class (all disturbance_flag = 1)
    single_class = train_samples.copy()
    single_class["disturbance_flag"] = 1
    single_class["disturbance_class"] = 4
    sc_export = single_class.head(100)
    write_tabular(GOLDEN_DIR / "edge_single_class.parquet", sc_export)

    # 6c: all-nodata (all feature columns NaN)
    feature_columns = list(system.feature_columns)
    nodata = train_samples.head(100).copy()
    for col in feature_columns:
        if col in nodata.columns:
            nodata[col] = np.nan
    write_tabular(GOLDEN_DIR / "edge_all_nodata.parquet", nodata)

    # 6d: single row test
    single_row = train_samples.iloc[:1].copy()
    write_tabular(GOLDEN_DIR / "edge_single_row.parquet", single_row)

    print(f"      → 4 edge case fixtures written")

    # ── step 7: synthetic NPZ chunks as Parquet ───────────────────
    print("[7/8] Exporting raw synthetic panel chunks as Parquet ...")
    write_tabular(GOLDEN_DIR / "synthetic_panel_chunk_0.parquet",
                  panel.iloc[:len(panel)//2].copy())
    write_tabular(GOLDEN_DIR / "synthetic_panel_chunk_1.parquet",
                  panel.iloc[len(panel)//2:].copy())
    print(f"      → synthetic_panel_chunk_0.parquet  ({len(panel)//2} rows)")
    print(f"      → synthetic_panel_chunk_1.parquet  ({len(panel) - len(panel)//2} rows)")

    # Also export the full raw panel for completeness
    write_tabular(GOLDEN_DIR / "synthetic_panel_full.parquet", panel)
    print(f"      → synthetic_panel_full.parquet  ({len(panel)} rows)")

# ── step 8: MANIFEST.json ────────────────────────────────────────
print("[8/8] Creating MANIFEST.json ...")

import subprocess
commit_hash = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=PROJECT_ROOT,
    text=True,
).strip()

now_iso = datetime.now(timezone.utc).isoformat()

files_found = sorted([p.name for p in GOLDEN_DIR.glob("*")])
manifest = {
    "source_commit": commit_hash,
    "generation_command": "python scripts/generate_golden_fixtures.py",
    "generated_at": now_iso,
    "random_seed": SEED,
    "description": (
        "Golden baseline corpus from the Python inducing_factor synthetic pipeline. "
        "All fixtures generated deterministically with seed=42. "
        "Used by Rust if-core, if-pipeline, and if-tests crates to verify "
        "behavior equivalence against the Python reference implementation."
    ),
    "tolerances": {
        "discretization_labels": "exact",
        "statistical_metrics": {"absolute": 1e-6, "relative": 1e-4},
        "factor_rankings": {"spearman_min": 0.95},
        "occurrence_probabilities": {"mae_max": 0.03, "max_error": 0.10},
        "class_predictions": {"top1_match_min": 0.95},
        "raster_metadata": "exact",
    },
    "pipeline_parameters": {
        "lookback_years": 2,
        "background_sample_rate": 0.3,
        "max_pattern_size": 3,
        "min_support": 0.08,
        "min_relative_risk": 1.2,
        "bootstrap_iterations": 5,
        "min_stability": 0.4,
        "max_bootstrap_samples": 1000,
        "max_network_factors": 6,
        "discretizer_method": "supervised_tree",
        "discretize_bins": 3,
        "random_state": SEED,
    },
    "files": {name: f".worktrees/rust-rewrite/tests/golden/{name}" for name in files_found},
}
write_json(GOLDEN_DIR / "MANIFEST.json", manifest)
print(f"      → MANIFEST.json  ({len(files_found)} files tracked)")

# ── README.md ─────────────────────────────────────────────────────
readme = f"""# Golden Baseline Fixtures

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

`{commit_hash}`

## Generated At

`{now_iso}`

## File Index

| File | Description |
|------|-------------|
"""

for name in sorted(files_found):
    # add description
    desc_map = {
        "MANIFEST.json": "Generation manifest with commit hash, tolerances, file listing",
        "README.md": "This file",
        "attribution_preview.parquet": "Sample attribution results (top factors, probabilities)",
        "bayesian_network_edges.json": "Consensus DAG edge list from Bayesian network structure learning",
        "discretization_labels.json": "Feature discretization meta: bins, labels, feature kinds",
        "factor_class_summary.parquet": "Factor-class association table (chi-square, frequencies)",
        "factor_library.json": "Full inducing factor library with statistics and metadata",
        "model_metadata.json": "Model metadata: feature columns, selected factors, backend info",
        "model_predictions.parquet": "Occurrence and class probabilities on sample rows",
        "selected_factors.json": "Selected inducing factors used in the Bayesian network",
        "synthetic_panel_chunk_0.parquet": "Raw synthetic panel data, chunk 0 (first half)",
        "synthetic_panel_chunk_1.parquet": "Raw synthetic panel data, chunk 1 (second half)",
        "synthetic_panel_full.parquet": "Full raw synthetic panel (all rows)",
        "training_samples.parquet": "Training samples used for model fitting (event + background)",
        "edge_empty_chunk.parquet": "Edge case: empty feature frame (0 rows)",
        "edge_single_class.parquet": "Edge case: single disturbance class (all class=4)",
        "edge_all_nodata.parquet": "Edge case: all-NaN feature values",
        "edge_single_row.parquet": "Edge case: single row input",
    }
    desc = desc_map.get(name, "")
    readme += f"| `{name}` | {desc} |\n"

readme += """
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
- The discretizer uses `supervised_tree` method with decision tree binning.
- Background sampling uses stratified sampling on static features.
"""

with open(GOLDEN_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("      → README.md")

print()
print("=" * 60)
print("GOLDEN GENERATION COMPLETE")
print(f"Output directory: {GOLDEN_DIR}")
print(f"Files generated:   {len(files_found)}")
print("=" * 60)

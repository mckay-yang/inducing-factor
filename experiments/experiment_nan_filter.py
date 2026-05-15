#!/usr/bin/env python
"""Experiment: validate proposed NaN filter fix for the attribution system."""

import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOOKBACK_YEARS = 3
STATIC_FEATURES = ["slope", "aspect", "dist_to_river"]
DYNAMIC_FEATURES = ["temperature", "resident", "dist_to_road", "dist_to_railway", "dist_to_waterway"]
CATEGORICAL_TEMPORAL_FEATURES = ["landuse"]
DIRECT_FEATURES = ["rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m", "rainfall_prev_total_3m"]
TARGET = "disturbance_flag"
CHUNK_PATH = "data/.cache/chunks_full_aoi/chunk_931.npz"

METADATA_COLUMNS = [
    "point_id", "raster_row", "raster_col", "x", "y", "year",
    "disturbance_time_value", "disturbance_type", "disturbance_class",
    "disturbance_month", "reference_month", TARGET,
]
SEP = "=" * 72
RANDOM_SEED = 42


def _load_chunk_dataframe(path):
    data = np.load(path, allow_pickle=True)
    return pd.DataFrame({k: data[k] for k in data.files})


def _assign_reference_months(df_chunk):
    df = df_chunk.copy()
    disturbance_source = (
        "disturbance_time_value" if "disturbance_time_value" in df.columns else "disturbance_type"
    )
    disturbance_month = pd.to_numeric(df["disturbance_month"], errors="coerce")
    disturbance_flag = (pd.to_numeric(df[disturbance_source], errors="coerce") > 0).astype(int)
    df["disturbance_flag"] = disturbance_flag
    df["disturbance_month"] = disturbance_month
    df["reference_month"] = disturbance_month
    bg_mask = disturbance_flag == 0
    if bg_mask.any():
        observed_months = disturbance_month[disturbance_flag == 1].dropna().astype(int)
        rng = np.random.default_rng(RANDOM_SEED)
        if observed_months.empty:
            sampled_months = rng.integers(1, 13, size=int(bg_mask.sum()))
        else:
            month_values = observed_months.to_numpy()
            sampled_months = rng.choice(month_values, size=int(bg_mask.sum()), replace=True)
        df.loc[bg_mask, "reference_month"] = sampled_months
    return df


def print_step(step, title):
    print(f"\n{SEP}")
    print(f"  STEP {step}: {title}")
    print(SEP)


def build_dynamic_features(df, sorted_idx, point_id_starts, point_id_ends):
    """Vectorized rolling history features (avoids slow groupby.transform(lambda))."""
    N = len(df)
    dynamic_feature_cols = [f for f in DYNAMIC_FEATURES if f in df.columns]
    for feature in dynamic_feature_cols:
        vals = df[feature].values
        for agg_name, agg_func in [("last", None), ("mean", np.mean), ("max", np.max)]:
            col_name = f"{feature}_prev_{agg_name}"
            result = np.full(N, np.nan, dtype=np.float64)
            for pid in range(len(point_id_starts)):
                start = point_id_starts[pid]
                end = point_id_ends[pid]
                if end - start <= LOOKBACK_YEARS:
                    result[start:end] = np.nan
                    continue
                seg = vals[start:end].copy()
                seg_shifted = np.full(len(seg), np.nan)
                seg_shifted[1:] = seg[:-1]
                if agg_name == "last":
                    window_result = seg_shifted
                else:
                    window_result = np.full(len(seg), np.nan)
                    kernel = np.ones(LOOKBACK_YEARS) / LOOKBACK_YEARS if agg_name == "mean" else None
                    for i in range(LOOKBACK_YEARS - 1, len(seg)):
                        w = seg_shifted[i - LOOKBACK_YEARS + 1 : i + 1]
                        if not np.any(np.isnan(w)):
                            window_result[i] = agg_func(w) if agg_name != "mean" else np.mean(w)
                result[start:end] = window_result
            df[col_name] = result
    for feature in CATEGORICAL_TEMPORAL_FEATURES:
        if feature not in df.columns:
            continue
        col_name = f"{feature}_prev_last"
        result = np.full(N, np.nan, dtype=np.float64)
        vals = df[feature].values
        for pid in range(len(point_id_starts)):
            start = point_id_starts[pid]
            end = point_id_ends[pid]
            seg = vals[start:end].astype(float)
            seg_shifted = np.full(len(seg), np.nan)
            seg_shifted[1:] = seg[:-1]
            result[start:end] = seg_shifted
        df[col_name] = result


def build_rainfall_features(df, sorted_idx, point_id_starts, point_id_ends):
    """Build rainfall_prev_Nm and rainfall_prev_total_3m."""
    if "reference_month" not in df.columns:
        return
    ref_month = pd.to_numeric(df["reference_month"], errors="coerce").values
    N = len(df)
    month_cols = {m: df[f"rainfall_month_{m:02d}"].values for m in range(1, 13) if f"rainfall_month_{m:02d}" in df.columns}
    if not month_cols:
        return
    prev_year = {}
    for m in month_cols:
        pv = np.full(N, np.nan, dtype=np.float64)
        vals = month_cols[m]
        for pid in range(len(point_id_starts)):
            start = point_id_starts[pid]
            end = point_id_ends[pid]
            seg = vals[start:end].copy()
            shifted = np.full(len(seg), np.nan)
            shifted[1:] = seg[:-1]
            pv[start:end] = shifted
        prev_year[m] = pv
    for offset in range(1, 4):
        col_name = f"rainfall_prev_{offset}m"
        result = np.full(N, np.nan, dtype=np.float64)
        for src_month in range(1, 13):
            target_month = src_month + offset
            if target_month <= 12:
                mask = ref_month == target_month
                result[mask] = month_cols[src_month][mask]
            else:
                mask = ref_month == (target_month - 12)
                result[mask] = prev_year[src_month][mask]
        df[col_name] = result
    total = np.full(N, np.nan, dtype=np.float64)
    parts = np.stack([df[f"rainfall_prev_{o}m"].values for o in range(1, 4)], axis=-1)
    valid = np.all(~np.isnan(parts), axis=-1)
    total[valid] = np.sum(parts[valid], axis=-1)
    df["rainfall_prev_total_3m"] = total


def main():
    print_step("0", "Load chunk_931.npz and report baseline stats")
    df_raw = _load_chunk_dataframe(CHUNK_PATH)
    total_rows_all = len(df_raw)
    raw_events_all = (df_raw["disturbance_time_value"] > 0).sum()
    print(f"  Total rows            : {total_rows_all:>12,}")
    print(f"  Raw events (TV > 0)   : {raw_events_all:>12,} ({100*raw_events_all/max(1,total_rows_all):.2f}%)")
    if "disturbance_class" in df_raw.columns:
        cd = df_raw.loc[df_raw["disturbance_time_value"] > 0, "disturbance_class"].value_counts()
        print("  Disturbance class dist:")
        for cls_val, cnt in sorted(cd.items()):
            print(f"    class={cls_val}: {cnt:>10,}")
    print("  Year distribution of raw events:")
    for yr, cnt in sorted(df_raw.loc[df_raw["disturbance_time_value"] > 0, "year"].value_counts().sort_index().items()):
        print(f"    {yr}: {cnt:>10,}")

    print_step("0b", "Sample point_ids for efficient feature building")
    point_ids = df_raw["point_id"].unique()
    event_pids = np.unique(df_raw.loc[df_raw["disturbance_time_value"] > 0, "point_id"].values)
    bg_pids = np.setdiff1d(point_ids, event_pids)
    rng = np.random.default_rng(RANDOM_SEED)
    n_sample_target = min(15000, len(point_ids))
    n_event_take = min(len(event_pids), n_sample_target)
    n_bg_take = max(0, n_sample_target - n_event_take)
    n_bg_take = min(n_bg_take, len(bg_pids))
    n_sample = n_event_take + n_bg_take
    sampled_event_pids = event_pids[:n_event_take].tolist()
    sampled_bg_pids = rng.choice(bg_pids, size=n_bg_take, replace=False).tolist()
    sampled_pids = sampled_event_pids + sampled_bg_pids
    df = df_raw[df_raw["point_id"].isin(sampled_pids)].copy()
    del df_raw
    total_rows = len(df)
    raw_events = (df["disturbance_time_value"] > 0).sum()
    print(f"  Sampled point_ids : {n_sample:>12,} ({n_event_take} event + {n_bg_take} bg)")
    print(f"  Sampled rows      : {total_rows:>12,} ({100*total_rows/total_rows_all:.1f}% of full)")
    print(f"  Sampled events    : {raw_events:>12,} ({100*raw_events/raw_events_all:.1f}% of all events)")

    print_step("0c", "Assign reference months and sort by (point_id, year)")
    df = _assign_reference_months(df)
    df = df.sort_values(["point_id", "year"]).reset_index(drop=True)
    N = len(df)

    point_id_arr = df["point_id"].values
    boundaries = np.where(point_id_arr[1:] != point_id_arr[:-1])[0] + 1
    point_id_starts = np.concatenate([[0], boundaries])
    point_id_ends = np.concatenate([boundaries, [N]])
    sorted_idx = df.index.values
    n_point_ids = len(point_id_starts)
    print(f"  Unique point_ids: {n_point_ids:,}")
    print(f"  Years per point_id: min={np.min(point_id_ends - point_id_starts)}, max={np.max(point_id_ends - point_id_starts)}")

    print_step("1", "Build features (vectorized)")
    for f in STATIC_FEATURES:
        if f in df.columns:
            df[f] = df[f]
    build_rainfall_features(df, sorted_idx, point_id_starts, point_id_ends)
    build_dynamic_features(df, sorted_idx, point_id_starts, point_id_ends)

    static_cols = [f for f in STATIC_FEATURES if f in df.columns]
    direct_cols = [f for f in DIRECT_FEATURES if f in df.columns]
    dyn_cols = []
    for f in DYNAMIC_FEATURES:
        if f not in df.columns:
            continue
        for agg in ["last", "mean", "max"]:
            col = f"{f}_prev_{agg}"
            if col in df.columns:
                dyn_cols.append(col)
    for f in CATEGORICAL_TEMPORAL_FEATURES:
        col = f"{f}_prev_last"
        if col in df.columns:
            dyn_cols.append(col)
    required_cols = static_cols + direct_cols + dyn_cols
    required_cols = list(dict.fromkeys(required_cols))

    print(f"\n  All required_cols ({len(required_cols)}): {required_cols}")
    print(f"  STATIC    ({len(static_cols)}): {static_cols}")
    print(f"  DIRECT    ({len(direct_cols)}): {direct_cols}")
    print(f"  DYNAMIC   ({len(dyn_cols)}): {dyn_cols}")

    event_mask = df[TARGET] == 1
    print(f"\n  NaN count per column (EVENT rows only, total events={event_mask.sum():,}):")
    print(f"  {'Column':<32s}  {'NaN count':>10s}  {'NaN %':>8s}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*8}")
    for col in required_cols:
        nan_cnt = df.loc[event_mask, col].isna().sum()
        pct = 100 * nan_cnt / max(1, event_mask.sum())
        print(f"  {col:<32s}  {nan_cnt:>10,d}  {pct:>7.2f}%")

    core_cols = static_cols + direct_cols

    print_step("2", "CURRENT filter (reproduce the bug)")
    current_cols = [col for col in required_cols if col in df.columns and df[col].notna().any()]
    dropped = set(required_cols) - set(current_cols)
    print(f"  Per-column notna().any() filter:")
    print(f"    Columns before: {len(required_cols)}, after: {len(current_cols)}")
    print(f"    Dropped cols : {sorted(dropped) if dropped else '(none)'}")

    history_mask = df[current_cols].notna().all(axis=1)
    events_before = event_mask.sum()
    rows_after_mask = history_mask.sum()
    events_after_mask = (event_mask & history_mask).sum()
    events_lost = events_before - events_after_mask
    print(f"\n  After notna().all(axis=1):")
    print(f"    Rows  after : {rows_after_mask:>12,} ({100*rows_after_mask/max(1,N):.1f}%)")
    print(f"    Events after: {events_after_mask:>12,} ({100*events_after_mask/max(1,events_before):.1f}%)")
    print(f"    Events LOST : {events_lost:>12,} ({100*events_lost/max(1,events_before):.1f}%)")

    meta_in = [c for c in METADATA_COLUMNS if c in df.columns]
    df_current = df.loc[history_mask, meta_in + current_cols].copy()
    df_current = df_current.replace([float("inf"), float("-inf")], pd.NA)
    df_current = df_current.dropna(subset=current_cols)
    print(f"\n  After dropna(subset=cols):")
    print(f"    Final rows  : {len(df_current):>12,}")
    print(f"    Final events: {(df_current[TARGET]==1).sum():>12,}")

    current_kept = history_mask
    final_cur = len(df_current)
    final_cur_events = (df_current[TARGET] == 1).sum()

    print_step("3", "PROPOSED filter (only static+direct)")
    proposed_core = [c for c in core_cols if c in df.columns and df[c].notna().any()]
    print(f"  Core columns: {proposed_core}")
    prop_mask = df[proposed_core].notna().all(axis=1)
    prop_rows = prop_mask.sum()
    prop_events = (event_mask & prop_mask).sum()
    prop_lost = events_before - prop_events
    print(f"\n  After notna().all(axis=1) on core only:")
    print(f"    Rows  after : {prop_rows:>12,} ({100*prop_rows/max(1,N):.1f}%)")
    print(f"    Events after: {prop_events:>12,} ({100*prop_events/max(1,events_before):.1f}%)")
    print(f"    Events LOST : {prop_lost:>12,} ({100*prop_lost/max(1,events_before):.1f}%)")

    df_prop = df.loc[prop_mask, meta_in + required_cols].copy()
    df_prop = df_prop.replace([float("inf"), float("-inf")], pd.NA)
    df_prop = df_prop.dropna(subset=proposed_core)
    final_prop = len(df_prop)
    final_prop_events = (df_prop[TARGET] == 1).sum()
    print(f"\n  After dropna(core only):")
    print(f"    Final rows  : {final_prop:>12,}")
    print(f"    Final events: {final_prop_events:>12,}")

    addl_events = final_prop_events - final_cur_events
    addl_rows = final_prop - final_cur
    pct_inc = 100 * addl_events / max(1, final_cur_events)
    print(f"\n  ADDITIONAL events retained: +{addl_events:,} ({pct_inc:.1f}% increase)")

    print_step("4", "Recovered rows analysis")
    recovered = prop_mask & ~current_kept
    rec_total = recovered.sum()
    rec_events = (event_mask & recovered).sum()
    print(f"  Recovered total : {rec_total:>12,}")
    print(f"  Recovered events: {rec_events:>12,}")
    if rec_total:
        print(f"\n  Columns with NaN causing rejection in CURRENT (for recovered rows):")
        for col in dyn_cols:
            nc = df.loc[recovered, col].isna().sum()
            if nc:
                print(f"    {col:<35s}: {nc:>10,d} NaN ({100*nc/max(1,rec_total):.1f}%)")
        print(f"\n  Year distribution of recovered events:")
        for yr, cnt in sorted(df.loc[recovered & event_mask, "year"].value_counts().sort_index().items()):
            print(f"    {yr}: {cnt:>10,d}")

    print_step("5", "NaN downstream handling — discretizer + factor mining")
    from inducing_factor.core.discretization import AdaptiveDiscretizer
    from inducing_factor.core.factor_mining import FactorMiner

    fcols = [c for c in required_cols if c in df_prop.columns]
    df_feat = df_prop[fcols].copy()
    y = df_prop[TARGET].astype(int)
    print(f"  Feature columns ({len(fcols)}):")
    for col in fcols:
        nan_pct = 100 * df_feat[col].isna().mean()
        print(f"    {col:<32s} NaN={nan_pct:.1f}%")

    discretizer = AdaptiveDiscretizer(n_bins=3, min_samples_leaf=0.05)
    discretizer.fit(df_feat, y)
    print(f"\n  Fitted. Meta:")
    for feat, meta in discretizer.feature_meta_.items():
        info = f"kind={meta['kind']}"
        if meta["kind"] == "continuous":
            info += f" labels={meta.get('labels','N/A')}"
        else:
            info += f" categories={meta.get('categories','N/A')}"
        print(f"    {feat}: {info}")

    df_labels = discretizer.transform(df_feat, as_labels=True)
    print(f"\n  Transformed (as_labels=True): shape={df_labels.shape}")
    for col in fcols:
        missing_cnt = (df_labels[col] == "missing").sum()
        print(f"    {col:<32s} 'missing' count={missing_cnt:>10,d}")

    fm = FactorMiner(max_pattern_size=3, min_support=0.05, min_relative_risk=1.5,
                     min_stability=0.6, bootstrap_iterations=100,
                     max_bootstrap_samples=10000, random_state=RANDOM_SEED,
                     score_method="log_rr")
    transactions = fm.transactions_from_labels(df_labels)
    print(f"\n  Total transactions: {len(transactions):,}")
    missing_txn = sum(1 for t in transactions if any("missing" in item for item in t))
    print(f"  Transactions with 'missing': {missing_txn:,} ({100*missing_txn/max(1,len(transactions)):.1f}%)")

    print(f"\n  FactorMiner.count_patterns (first 2000 txn):")
    pc = fm.count_patterns(transactions[:2000])
    print(f"    Unique patterns: {len(pc)}")
    for pat, cnt in pc.most_common(5):
        print(f"    {cnt:>6d}  {pat}")

    print(f"\n  FactorMiner.update_pattern_counts (first 2000):")
    ec, bc = Counter(), Counter()
    ne, nb = fm.update_pattern_counts(df_labels.iloc[:2000], y.iloc[:2000], ec, bc)
    print(f"    Event patterns: {len(ec)}, Bg patterns: {len(bc)}")
    print(f"\n  No crashes — 'missing' is handled safely.")

    print_step("6", "Summary: CURRENT vs PROPOSED")
    print(f"  {'Metric':<45s}  {'CURRENT':>12s}  {'PROPOSED':>12s}  {'DIFF':>12s}")
    print(f"  {'-'*45}  {'-'*12}  {'-'*12}  {'-'*12}")
    rows = [
        ("Total rows retained", final_cur, final_prop),
        ("Total events retained", final_cur_events, final_prop_events),
        ("Event retention rate",
         f"{100*final_cur_events/max(1,events_before):.1f}%",
         f"{100*final_prop_events/max(1,events_before):.1f}%",
         f"{100*addl_events/max(1,events_before):+.1f}pp"),
        ("Rows lost to NaN filter", N - rows_after_mask, N - prop_rows),
        ("Background rows retained",
         final_cur - final_cur_events,
         final_prop - final_prop_events),
    ]
    for label, cur, prop, *rest in rows:
        diff = prop - cur if isinstance(cur, (int, float)) and isinstance(prop, (int, float)) else "—"
        if isinstance(diff, (int, float)):
            diff = f"{diff:+,}"
        print(f"  {label:<45s}  {str(cur):>12s}  {str(prop):>12s}  {diff:>12s}")
    print(f"\n  KEY: Proposed retains {addl_events:,} more events ({pct_inc:.1f}% increase)")
    print(f"  NaN in dynamic features safely handled as 'missing'.")


if __name__ == "__main__":
    main()

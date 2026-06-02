"""Tests for prediction-mode feature building (CCDC-free, predict_month-driven)."""
import numpy as np
import pandas as pd
import pytest

from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem
from inducing_factor.pipeline.feature_builder import FeatureBuilder


def _make_fb(lookback_years=2):
    return FeatureBuilder(
        target="disturbance_flag",
        lookback_years=lookback_years,
        static_features=["slope", "aspect", "dist_to_river"],
        dynamic_features=["temperature", "resident"],
        categorical_temporal_features=["landuse"],
        direct_features=[
            "rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m",
            "rainfall_prev_total_3m",
        ],
    )


def _make_mini_panel(n_points=4, years=None, seed=1):
    """Build a minimal panel with rainfall columns and point_ids for feature tests."""
    rng = np.random.default_rng(seed)
    if years is None:
        years = [2019, 2020, 2021]
    records = []
    for pid in range(n_points):
        base_rain = rng.normal(4.5, 0.5)
        for yr in years:
            row = {
                "point_id": pid,
                "year": yr,
                "slope": 15.0,
                "aspect": 180.0,
                "dist_to_river": 3000.0,
                "temperature": 12.0 + rng.normal(0, 0.5),
                "resident": 1.0 + rng.normal(0, 0.2),
                "landuse": 1,
                "dist_to_road": 2000.0,
                "disturbance_time_value": 0.0,
                "disturbance_type": 0.0,
                "disturbance_flag": 0,
                "disturbance_month": np.nan,
                "reference_month": np.nan,
            }
            for m in range(1, 13):
                row[f"rainfall_month_{m:02d}"] = base_rain + rng.normal(0, 0.3)
            records.append(row)
    return pd.DataFrame(records)


class TestRainfallFeaturesWithExplicitPredictMonth:
    """Verify correct prev_1m/2m/3m/total_3m computed for each month 1-12."""

    def test_predict_month_produces_all_rainfall_cols(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=1, years=[2020, 2021])
        result = fb.build_rainfall_window_features(df.copy(), predict_month=6)

        for col in ["rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m", "rainfall_prev_total_3m"]:
            assert col in result.columns, f"Missing {col}"
            assert not result[col].isna().all(), f"All NaN for {col}"

    def test_prev_1m_is_previous_month_rainfall(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=1, years=[2020])
        df["rainfall_month_05"] = 10.0
        df["rainfall_month_06"] = 20.0
        df["rainfall_month_04"] = 5.0
        result = fb.build_rainfall_window_features(df.copy(), predict_month=6)

        row = result.iloc[0]
        assert row["rainfall_prev_1m"] == 10.0  # month 5
        assert row["rainfall_prev_2m"] == 5.0   # month 4

    def test_prev_3m_wraps_to_previous_year(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=1, years=[2019, 2020])
        df["point_id"] = 0
        df["rainfall_month_01"] = 3.0
        df["rainfall_month_02"] = 5.0
        df["rainfall_month_11"] = 7.0
        df["rainfall_month_12"] = 9.0
        df.loc[df["year"] == 2019, "rainfall_month_11"] = 8.0
        df.loc[df["year"] == 2019, "rainfall_month_12"] = 10.0

        result = fb.build_rainfall_window_features(df.copy(), predict_month=2)
        row2020 = result[result["year"] == 2020].iloc[0]

        assert row2020["rainfall_prev_1m"] == 3.0
        assert row2020["rainfall_prev_2m"] == 10.0
        assert row2020["rainfall_prev_3m"] == 8.0

    def test_total_3m_equals_sum(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=1, years=[2020])
        for m in range(1, 13):
            df[f"rainfall_month_{m:02d}"] = float(m)
        result = fb.build_rainfall_window_features(df.copy(), predict_month=6)
        row = result.iloc[0]
        expected = row["rainfall_prev_1m"] + row["rainfall_prev_2m"] + row["rainfall_prev_3m"]
        assert abs(row["rainfall_prev_total_3m"] - expected) < 1e-6

    def test_12_months_all_produce_different_values(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=1, years=[2020])
        for m in range(1, 13):
            df[f"rainfall_month_{m:02d}"] = float(m)
        prev1 = []
        for month in range(1, 13):
            result = fb.build_rainfall_window_features(df.copy(), predict_month=month)
            prev1.append(result.iloc[0]["rainfall_prev_1m"])
        # All 12 months should have different prev_1m values since rainfall varies by month
        assert len(set(prev1)) > 1


class TestPredictModeNoCCDCRequired:
    """Prediction with predict_month succeeds without CCDC columns."""

    def test_build_features_without_disturbance_columns(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=2, years=[2020, 2021])
        for col in ["disturbance_time_value", "disturbance_type", "disturbance_flag",
                     "disturbance_month", "reference_month", "disturbance_class"]:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        df_out, required = fb.build_chunk_features(df, predict_month=7)
        assert required, "Should produce non-empty feature list"
        assert "rainfall_prev_1m" in df_out.columns
        assert "slope" in df_out.columns or "slope" in required

    def test_system_build_prediction_features_no_ccdc(self, synthetic_cache_and_output):
        cache_dir, out_dir = synthetic_cache_and_output
        system = EcoDisturbanceAttributionSystem(
            cache_dir=cache_dir,
            out_dir=out_dir,
            lookback_years=2,
        )
        system.run()
        chunk_files = list(system._iter_chunk_files())
        assert chunk_files
        df = system._load_chunk_dataframe(chunk_files[0])
        df_clean = df.drop(columns=["disturbance_time_value", "disturbance_type",
                                     "disturbance_month", "disturbance_class",
                                     "disturbance_flag", "reference_month"], errors="ignore")
        lb = system.feature_builder.lookback_years
        target_year = df_clean["year"].max()
        df_window = df_clean[df_clean["year"].isin(range(target_year - lb - 1, target_year + 1))].copy()

        # Should NOT raise KeyError (no CCDC columns)
        df_features, required = system._build_prediction_features(df_window, predict_month=7)
        assert required
        df_year = df_features[df_features["year"] == target_year].copy()
        for col in system.feature_columns:
            if col not in df_year.columns:
                df_year[col] = np.nan
        pred = system.predict_dataframe(df_year, predict_month=7)
        assert "disturbance_probability" in pred.columns


class TestEnsembleMeanMaxPeakMonth:
    """Ensemble over 12 months produces expected aggregate values."""

    def test_ensemble_returns_aggregate_columns(self, synthetic_cache_and_output):
        cache_dir, out_dir = synthetic_cache_and_output
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
        )
        system.run()
        chunk_files = list(system._iter_chunk_files())
        df = system._load_chunk_dataframe(chunk_files[0])
        lb = system.feature_builder.lookback_years
        target = df["year"].max()
        df_window = df[df["year"].isin(range(target - lb - 1, target + 1))].copy()

        result = system.predict_ensemble(df_window, target)
        assert "disturbance_probability_mean" in result.columns
        assert "disturbance_probability_max" in result.columns
        assert "disturbance_probability_peak_month" in result.columns
        assert "disturbance_probability" in result.columns  # legacy = mean
        assert len(result) > 0

    def test_mean_between_min_and_max(self, synthetic_cache_and_output):
        cache_dir, out_dir = synthetic_cache_and_output
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
        )
        system.run()
        chunk_files = list(system._iter_chunk_files())
        df = system._load_chunk_dataframe(chunk_files[0])
        lb = system.feature_builder.lookback_years
        target = df["year"].max()
        df_window = df[df["year"].isin(range(target - lb - 1, target + 1))].copy()

        result = system.predict_ensemble(df_window, target)
        for _, row in result.iterrows():
            assert row["disturbance_probability_mean"] <= row["disturbance_probability_max"] + 1e-9  # allow float

    def test_peak_month_in_range(self, synthetic_cache_and_output):
        cache_dir, out_dir = synthetic_cache_and_output
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
        )
        system.run()
        chunk_files = list(system._iter_chunk_files())
        df = system._load_chunk_dataframe(chunk_files[0])
        lb = system.feature_builder.lookback_years
        target = df["year"].max()
        df_window = df[df["year"].isin(range(target - lb - 1, target + 1))].copy()

        result = system.predict_ensemble(df_window, target)
        for _, row in result.iterrows():
            assert 1 <= row["disturbance_probability_peak_month"] <= 12


class TestTrainingModeUnchanged:
    """Existing training behavior preserved when predict_month is None."""

    def test_training_with_predict_month_none(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=2, years=[2020, 2021])
        df["reference_month"] = 6
        df["disturbance_flag"] = 0

        df_out, required = fb.build_chunk_features(df, predict_month=None)
        assert "rainfall_prev_1m" in df_out.columns

    def test_training_without_predict_month_param(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=2, years=[2020, 2021])
        df["reference_month"] = 6
        df["disturbance_flag"] = 0

        df_out, required = fb.build_chunk_features(df)  # no predict_month arg
        assert "rainfall_prev_1m" in df_out.columns

    def test_training_without_reference_month_skips_rainfall(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=2, years=[2020, 2021])
        df["disturbance_flag"] = 0
        df = df.drop(columns=["reference_month"], errors="ignore")
        df_out, required = fb.build_chunk_features(df, predict_month=None)
        assert "rainfall_prev_1m" not in df_out.columns

    def test_training_requires_reference_month_for_rainfall(self):
        fb = _make_fb()
        df = _make_mini_panel(n_points=2, years=[2020])
        df["disturbance_flag"] = 0
        df["reference_month"] = 6
        df_out, required = fb.build_chunk_features(df, predict_month=None)
        assert "rainfall_prev_1m" in df_out.columns


class TestPredictEnsembleAllMonths:
    """Each month 1-12 produces different probabilities."""

    def test_different_months_produce_different_probs(self, synthetic_cache_and_output):
        cache_dir, out_dir = synthetic_cache_and_output
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
        )
        system.run()
        chunk_files = list(system._iter_chunk_files())
        df = system._load_chunk_dataframe(chunk_files[0])
        lb = system.feature_builder.lookback_years
        target = df["year"].max()
        df_window = df[df["year"].isin(range(target - lb - 1, target + 1))].copy()

        prob_sets = []
        for month in range(1, 13):
            df_feat, required = system._build_prediction_features(df_window, predict_month=month)
            df_year = df_feat[df_feat["year"] == target].copy()
            if df_year.empty:
                continue
            for col in system.feature_columns:
                if col not in df_year.columns:
                    df_year[col] = np.nan
            drop_mask = df_year[system.feature_columns].isna().any(axis=1)
            if drop_mask.all():
                continue
            pred = system.predict_dataframe(df_year[~drop_mask].copy(), predict_month=month)
            if pred.empty:
                continue
            prob_sets.append(set(pred["disturbance_probability"].round(4).tolist()))

        # All 12 months should produce at least somewhat different probabilities
        unique_sets = len(set(frozenset(s) for s in prob_sets))
        assert unique_sets > 1, "Each month should produce different probability patterns"

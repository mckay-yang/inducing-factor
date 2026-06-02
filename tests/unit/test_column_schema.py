import json
import os
from pathlib import Path

import pandas as pd
import pytest

from inducing_factor.pipeline.column_schema import (
    ALLOWED_PREDICTORS,
    HISTORICAL_PREDICTORS,
    LABEL_COLUMNS,
    LEAKAGE_FORBIDDEN,
    METADATA_COLUMNS,
    STATIC_PREDICTORS,
    get_predictor_columns_only,
    validate_no_leakage,
    validate_predictor_columns,
)


class TestSchemaInvariants:
    """Tests that the column schema definitions are internally consistent."""

    def test_allowed_predictors_no_overlap_with_labels(self):
        overlap = ALLOWED_PREDICTORS & LABEL_COLUMNS
        assert not overlap, (
            f"ALLOWED_PREDICTORS and LABEL_COLUMNS must be disjoint, "
            f"found: {overlap}"
        )

    def test_allowed_predictors_no_overlap_with_leakage(self):
        overlap = ALLOWED_PREDICTORS & LEAKAGE_FORBIDDEN
        assert not overlap, (
            f"ALLOWED_PREDICTORS and LEAKAGE_FORBIDDEN must be disjoint, "
            f"found: {overlap}"
        )

    def test_leakage_forbidden_not_in_allowed(self):
        for col in LEAKAGE_FORBIDDEN:
            assert col not in ALLOWED_PREDICTORS, (
                f"'{col}' is in LEAKAGE_FORBIDDEN but also in ALLOWED_PREDICTORS"
            )

    def test_static_and_historical_disjoint(self):
        overlap = STATIC_PREDICTORS & HISTORICAL_PREDICTORS
        assert not overlap, (
            f"STATIC_PREDICTORS and HISTORICAL_PREDICTORS must be disjoint, "
            f"found: {overlap}"
        )

    def test_allowed_predictors_equals_union(self):
        assert ALLOWED_PREDICTORS == STATIC_PREDICTORS | HISTORICAL_PREDICTORS, (
            "ALLOWED_PREDICTORS must be exactly STATIC_PREDICTORS | HISTORICAL_PREDICTORS"
        )

    def test_labels_not_in_predictors(self):
        for col in LABEL_COLUMNS:
            assert col not in STATIC_PREDICTORS, f"'{col}' must not be in STATIC_PREDICTORS"
            assert col not in HISTORICAL_PREDICTORS, f"'{col}' must not be in HISTORICAL_PREDICTORS"

    def test_metadata_not_in_predictors(self):
        for col in METADATA_COLUMNS:
            assert col not in ALLOWED_PREDICTORS, (
                f"Metadata column '{col}' must not be in ALLOWED_PREDICTORS"
            )

    def test_known_forbidden_columns_are_listed(self):
        """Ensure critical CCDC-derived columns are in LEAKAGE_FORBIDDEN."""
        required = {
            "disturbance_time_value",
            "disturbance_type",
            "disturbance_month",
            "disturbance_class",
            "disturbance_flag",
            "reference_month",
        }
        missing = required - LEAKAGE_FORBIDDEN
        assert not missing, f"Missing from LEAKAGE_FORBIDDEN: {missing}"

    def test_known_predictors_in_allowed(self):
        expected = {
            "slope", "aspect", "dist_to_river",
            "temperature_prev_last", "temperature_prev_mean", "temperature_prev_max",
            "resident_prev_last", "resident_prev_mean", "resident_prev_max",
            "dist_to_road_prev_last", "dist_to_road_prev_mean", "dist_to_road_prev_max",
            "dist_to_railway_prev_last", "dist_to_railway_prev_mean", "dist_to_railway_prev_max",
            "dist_to_waterway_prev_last", "dist_to_waterway_prev_mean", "dist_to_waterway_prev_max",
            "landuse_prev_last",
            "rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m",
            "rainfall_prev_total_3m",
        }
        missing = expected - ALLOWED_PREDICTORS
        assert not missing, f"Expected predictors missing from ALLOWED_PREDICTORS: {missing}"


class TestValidationHelpers:
    """Tests for the column validation helper functions."""

    def test_validate_no_leakage_detects_violation(self):
        cols = ["slope", "aspect", "disturbance_time_value", "reference_month", "point_id"]
        violations = validate_no_leakage(cols)
        assert sorted(violations) == ["disturbance_time_value", "reference_month"]

    def test_validate_no_leakage_clean_returns_empty(self):
        cols = ["slope", "aspect", "dist_to_river", "rainfall_prev_1m"]
        violations = validate_no_leakage(cols)
        assert violations == []

    def test_validate_predictor_columns_rejects_labels(self):
        cols = ["slope", "disturbance_flag"]
        violations = validate_predictor_columns(cols)
        assert "disturbance_flag" in violations

    def test_validate_predictor_columns_rejects_metadata(self):
        cols = ["slope", "year", "point_id"]
        violations = validate_predictor_columns(cols)
        assert "year" in violations
        assert "point_id" in violations

    def test_validate_predictor_columns_rejects_forbidden(self):
        cols = ["slope", "reference_month", "disturbance_time_value"]
        violations = validate_predictor_columns(cols)
        assert "reference_month" in violations
        assert "disturbance_time_value" in violations

    def test_validate_predictor_columns_all_valid(self):
        cols = ["slope", "aspect", "rainfall_prev_1m", "dist_to_road_prev_last"]
        violations = validate_predictor_columns(cols)
        assert violations == []

    def test_validate_predictor_columns_empty_input(self):
        assert validate_predictor_columns([]) == []

    def test_validate_no_leakage_empty_input(self):
        assert validate_no_leakage([]) == []

    def test_get_predictor_columns_only_filters_correctly(self):
        cols = ["slope", "disturbance_flag", "year", "reference_month",
                "rainfall_prev_1m", "disturbance_time_value", "point_id"]
        result = get_predictor_columns_only(cols)
        assert sorted(result) == ["rainfall_prev_1m", "slope"]


class TestBadRunTrainingSamples:
    """Tests against the archived bad run to document current data state."""

    @pytest.fixture
    def bad_run_dir(self):
        p = Path("results/bad_runs/20260602T010327Z")
        if not p.is_dir():
            pytest.skip("Bad run archive not available")
        return p

    @pytest.fixture
    def bad_run_training_samples(self, bad_run_dir):
        csv_path = bad_run_dir / "models" / "training_samples.csv"
        if not csv_path.is_file():
            pytest.skip("training_samples.csv not found in bad run")
        return pd.read_csv(csv_path, nrows=1)

    @pytest.fixture
    def bad_run_metadata(self, bad_run_dir):
        meta_path = bad_run_dir / "models" / "model_metadata.json"
        if not meta_path.is_file():
            pytest.skip("model_metadata.json not found in bad run")
        with open(meta_path) as f:
            return json.load(f)

    def test_bad_run_feature_columns_no_leakage(self, bad_run_metadata):
        """Feature columns in metadata must not include LEAKAGE_FORBIDDEN columns."""
        feature_cols = bad_run_metadata.get("feature_columns", [])
        violations = {c for c in feature_cols if c in LEAKAGE_FORBIDDEN}
        assert not violations, (
            f"Bad run feature_columns contain LEAKAGE_FORBIDDEN columns: {violations}. "
            f"feature_columns={feature_cols}"
        )

    def test_bad_run_feature_columns_all_allowed(self, bad_run_metadata):
        """All feature columns should be in ALLOWED_PREDICTORS."""
        feature_cols = bad_run_metadata.get("feature_columns", [])
        violations = validate_predictor_columns(feature_cols)
        assert not violations, (
            f"Bad run feature_columns contain non-allowed columns: {violations}"
        )

    def test_bad_run_training_columns_have_leakage(self, bad_run_training_samples):
        """Document current state: forbidden columns are present in training_samples.

        This documents the structural problem: CCDC-derived columns exist in the
        training DataFrame even though they may not be directly used as features.
        The presence of reference_month in particular enables leakage through
        rainfall window computation.
        """
        cols = set(bad_run_training_samples.columns)
        leaked = cols & LEAKAGE_FORBIDDEN
        # Currently these ARE present — this test documents that reality.
        # After fixes: this assertion should be flipped to assert not leaked.
        assert leaked, (
            "Expected to find LEAKAGE_FORBIDDEN columns in bad run training_samples, "
            "but none were present. This may mean the data has already been cleaned."
        )
        # List what's actually there for documentation
        critical = {"reference_month", "disturbance_time_value", "disturbance_month"}
        found_critical = leaked & critical
        assert found_critical, (
            f"Critical leakage columns not found in training_samples: "
            f"expected {critical}, found {leaked}"
        )


class TestSyntheticPipeline:
    """Tests against the synthetic pipeline output to verify predictor columns."""

    def test_synthetic_pipeline_predictor_columns(self, synthetic_cache_and_output):
        """Verify synthetic pipeline doesn't use forbidden columns in features."""
        from inducing_factor import EcoDisturbanceAttributionSystem

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
        result = system.run()

        feature_cols = system.feature_columns
        assert feature_cols, "feature_columns must not be empty"

        violations = validate_predictor_columns(feature_cols)
        assert not violations, (
            f"feature_columns contain non-allowed columns: {violations}"
        )

        leakage = validate_no_leakage(feature_cols)
        assert not leakage, (
            f"feature_columns contain LEAKAGE_FORBIDDEN columns: {leakage}"
        )

        # training_samples may contain metadata/labels but feature_columns must be clean
        ts_cols = set(result["training_samples"].columns)
        ts_leakage = ts_cols & LEAKAGE_FORBIDDEN
        # reference_month is expected in training_samples (used for window calc)
        # but should NOT be in feature_columns
        assert "reference_month" not in feature_cols, (
            "reference_month must not be in feature_columns"
        )

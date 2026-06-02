"""Tests for temporal validation split, metrics, and gating."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from inducing_factor.core.validation import (
    check_validation_gate,
    compute_validation_metrics,
    temporal_holdout_split,
)
from inducing_factor.config.models import AppConfig
from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem


class TestTemporalHoldoutSplit:
    def test_train_years_le_train_end_year(self):
        df = pd.DataFrame({
            "year": [2015, 2016, 2017, 2018, 2019, 2020],
            "value": range(6),
        })
        train_df, val_df = temporal_holdout_split(df, train_end_year=2017)
        assert (train_df["year"] <= 2017).all()
        assert (val_df["year"] > 2017).all()
        assert len(train_df) + len(val_df) == len(df)

    def test_explicit_validation_years(self):
        df = pd.DataFrame({
            "year": [2015, 2016, 2017, 2018, 2019, 2020],
            "value": range(6),
        })
        train_df, val_df = temporal_holdout_split(df, train_end_year=2016, val_years=[2019, 2020])
        assert (train_df["year"] <= 2016).all()
        assert set(val_df["year"].unique()) == {2019, 2020}

    def test_no_validation_years_produces_empty_val(self):
        df = pd.DataFrame({
            "year": [2015, 2016, 2017],
            "value": range(3),
        })
        train_df, val_df = temporal_holdout_split(df, train_end_year=2018)
        assert len(val_df) == 0
        assert len(train_df) == len(df)

    def test_train_end_year_none_uses_empty_val(self):
        df = pd.DataFrame({
            "year": [2015, 2016, 2017],
            "value": range(3),
        })
        train_df, val_df = temporal_holdout_split(df, train_end_year=2015, val_years=[])
        assert len(val_df) == 0
        assert (train_df["year"] <= 2015).all()

    def test_works_with_custom_year_column_name(self):
        df = pd.DataFrame({
            "yr": [2015, 2016, 2017, 2018],
            "value": range(4),
        })
        train_df, val_df = temporal_holdout_split(df, train_end_year=2016, year_col="yr")
        assert (train_df["yr"] <= 2016).all()
        assert (val_df["yr"] > 2016).all()


class TestValidationMetrics:
    def test_output_includes_auc_pr_auc_brier(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred_prob = np.array([0.1, 0.2, 0.15, 0.3, 0.1, 0.6, 0.7, 0.8, 0.9, 0.55])
        metrics = compute_validation_metrics(y_true, y_pred_prob)
        assert "auc" in metrics
        assert "pr_auc" in metrics
        assert "brier_score" in metrics
        assert "calibration_slope" in metrics
        assert "n_val" in metrics
        assert "prevalence" in metrics
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["brier_score"] <= 1

    def test_perfect_prediction_gets_high_auc(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred_prob = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 1.0, 0.9, 1.0, 1.0])
        metrics = compute_validation_metrics(y_true, y_pred_prob)
        assert metrics["auc"] >= 0.95

    def test_per_class_metrics_when_provided(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        y_pred_prob = np.array([0.2, 0.1, 0.7, 0.8, 0.3, 0.9, 0.2, 0.6, 0.1, 0.85])
        y_pred_class = np.array([0, 0, 1, 1, 0, 1, 0, 4, 0, 2])
        class_labels = {1: "fire", 2: "logging", 4: "road_construction"}
        metrics = compute_validation_metrics(
            y_true, y_pred_prob, y_pred_class=y_pred_class,
            class_labels=class_labels,
        )
        assert "per_class" in metrics
        assert "fire" in metrics["per_class"]
        assert "count" in metrics["per_class"]["fire"]
        assert "prevalence" in metrics["per_class"]["fire"]

    def test_single_class_returns_nan_metrics(self):
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        metrics = compute_validation_metrics(y_true, y_pred_prob)
        assert np.isnan(metrics["auc"])
        assert np.isnan(metrics["pr_auc"])
        assert not np.isnan(metrics["n_val"])
        assert metrics["prevalence"] == 1.0


class TestValidationGate:
    def test_passes_when_metrics_above_thresholds(self):
        metrics = {"auc": 0.75, "pr_auc": 0.15, "n_val": 100}
        passed, reason = check_validation_gate(metrics, {"auc_min": 0.60, "pr_auc_min": 0.05})
        assert passed
        assert "passed" in reason.lower()

    def test_fails_when_auc_below_threshold(self):
        metrics = {"auc": 0.45, "pr_auc": 0.15, "n_val": 100}
        passed, reason = check_validation_gate(metrics, {"auc_min": 0.60, "pr_auc_min": 0.05})
        assert not passed
        assert "failed" in reason.lower()

    def test_fails_when_pr_auc_below_threshold(self):
        metrics = {"auc": 0.75, "pr_auc": 0.02, "n_val": 100}
        passed, reason = check_validation_gate(metrics, {"auc_min": 0.60, "pr_auc_min": 0.05})
        assert not passed
        assert "failed" in reason.lower()

    def test_passes_when_no_thresholds(self):
        metrics = {"auc": 0.45, "pr_auc": 0.02, "n_val": 100}
        passed, reason = check_validation_gate(metrics, None)
        assert passed

    def test_nan_metrics_fail_gate(self):
        metrics = {"auc": float("nan"), "pr_auc": float("nan"), "n_val": 100}
        passed, reason = check_validation_gate(metrics, {"auc_min": 0.60})
        assert not passed


class TestTemporalPipelineIntegration:
    def test_train_end_year_filters_training_data(self, synthetic_cache_and_output):
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
            train_end_year=2018,
        )
        assert system.train_end_year == 2018

    def test_temporal_split_excludes_holdout_from_training(self, synthetic_cache_and_output):
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
            train_end_year=2018,
        )
        result = system.run()
        training_samples = result["training_samples"]
        assert not training_samples.empty
        assert (training_samples["year"] <= 2018).all()

    def test_validation_report_saved_when_split_configured(self, synthetic_cache_and_output):
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
            train_end_year=2019,
        )
        result = system.run()
        report_path = os.path.join(out_dir, "validation_report.json")
        assert os.path.exists(report_path), f"validation_report.json not found at {report_path}"
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert "auc" in report or "gate_passed" in report
        assert "train_end_year" in report
        assert report["train_end_year"] == 2019

    def test_no_validation_when_train_end_year_none(self, synthetic_cache_and_output):
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
            train_end_year=None,
        )
        result = system.run()
        assert not result["training_samples"].empty
        report_path = os.path.join(out_dir, "validation_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                json.load(f)


class TestValidationGateBlocksPrediction:
    def test_skip_validation_flag_allows_prediction_without_report(self):
        from inducing_factor.cli import _check_validation_before_predict
        from inducing_factor.pipeline.probability_mapper import AnnualProbabilityMapper

        with tempfile.TemporaryDirectory() as out_dir:
            config = AppConfig()
            config.paths.out_dir = out_dir
            mapper = AnnualProbabilityMapper(config=config)
            mapper.system.validation_required = True

            _check_validation_before_predict(mapper, skip_validation=True)

    def test_missing_report_raises_when_required(self):
        from inducing_factor.cli import _check_validation_before_predict
        from inducing_factor.pipeline.probability_mapper import AnnualProbabilityMapper

        with tempfile.TemporaryDirectory() as out_dir:
            config = AppConfig()
            config.paths.out_dir = out_dir
            mapper = AnnualProbabilityMapper(config=config)
            mapper.system.validation_required = True

            with pytest.raises(RuntimeError, match="No validation report"):
                _check_validation_before_predict(mapper, skip_validation=False)

    def test_failed_gate_raises_when_required(self):
        from inducing_factor.cli import _check_validation_before_predict
        from inducing_factor.pipeline.probability_mapper import AnnualProbabilityMapper

        with tempfile.TemporaryDirectory() as out_dir:
            config = AppConfig()
            config.paths.out_dir = out_dir
            report_path = os.path.join(out_dir, "validation_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"gate_passed": False, "gate_reason": "auc=0.45 < 0.60"}, f)

            mapper = AnnualProbabilityMapper(config=config)
            mapper.system.validation_required = True

            with pytest.raises(RuntimeError, match="Validation gate failed"):
                _check_validation_before_predict(mapper, skip_validation=False)

    def test_failed_gate_allowed_with_skip_flag(self):
        from inducing_factor.cli import _check_validation_before_predict
        from inducing_factor.pipeline.probability_mapper import AnnualProbabilityMapper

        with tempfile.TemporaryDirectory() as out_dir:
            config = AppConfig()
            config.paths.out_dir = out_dir
            report_path = os.path.join(out_dir, "validation_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"gate_passed": False, "gate_reason": "auc=0.45 < 0.60"}, f)

            mapper = AnnualProbabilityMapper(config=config)

            _check_validation_before_predict(mapper, skip_validation=True)

    def test_passed_gate_proceeds_normally(self):
        from inducing_factor.cli import _check_validation_before_predict
        from inducing_factor.pipeline.probability_mapper import AnnualProbabilityMapper

        with tempfile.TemporaryDirectory() as out_dir:
            config = AppConfig()
            config.paths.out_dir = out_dir
            report_path = os.path.join(out_dir, "validation_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"gate_passed": True, "gate_reason": "validation gate passed"}, f)

            mapper = AnnualProbabilityMapper(config=config)

            _check_validation_before_predict(mapper, skip_validation=False)

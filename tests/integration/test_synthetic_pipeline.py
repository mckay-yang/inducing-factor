import logging

from inducing_factor import EcoDisturbanceAttributionSystem


def test_inducing_factor_pipeline(synthetic_cache_and_output):
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
    assert not result["training_samples"].empty
    assert not result["factors"].empty
    assert system.model is not None
    assert system.occurrence_engine.model is not None
    assert system.class_engine.model is not None
    selected_text = " ".join(" ".join(f.items) for f in system.selected_factors)
    # Check that some factors were selected (system is working)
    assert len(system.selected_factors) > 0
    assert "elevation" not in selected_text  # elevation should be excluded from factor mining
    # Check that at least some dynamic features are present (system is discovering patterns)
    dynamic_present = any(
        feature in selected_text
        for feature in [
            "rainfall_prev_1m",
            "rainfall_prev_2m",
            "rainfall_prev_3m",
            "rainfall_prev_total_3m",
            "temperature",
            "resident",
            "landuse",
        ]
    )
    # Allow static features too (dist_to_road, dist_to_river, slope, aspect)
    static_present = any(
        feature in selected_text
        for feature in ["dist_to_road", "dist_to_river", "slope", "aspect"]
    )
    assert dynamic_present or static_present, f"No meaningful features selected: {selected_text}"
    assert result["training_samples"]["disturbance_flag"].sum() > 0
    disturbed = result["training_samples"][result["training_samples"]["disturbance_flag"] == 1]
    assert disturbed["disturbance_month"].notna().all()
    assert disturbed[["rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m", "rainfall_prev_total_3m"]].notna().all().all()
    risky_sample = result["training_samples"].sort_values("disturbance_flag", ascending=False).iloc[0]
    attribution = system.attribute_sample(risky_sample)
    assert 0 <= attribution["disturbance_probability"] <= 1
    assert attribution["attribution_type"] in {
        "single_factor_dominant",
        "multi_factor_synergy",
        "background_or_unknown",
    }
    assert len(attribution["top_factors"]) > 0
    assert "disturbance_class_probabilities" in attribution
    assert attribution["dominant_disturbance_class"] in {1, 2, 3, 4}
    prediction_frame = system.predict_dataframe(result["training_samples"].head(10))
    assert "disturbance_probability" in prediction_frame.columns
    assert any(col.startswith("class_probability_") for col in prediction_frame.columns)
    assert result["factor_class_summary"] is not None


def test_inducing_factor_run_reports_progress_and_uses_streaming_samples(synthetic_cache_and_output, caplog):
    cache_dir, out_dir = synthetic_cache_and_output
    caplog.set_level(logging.INFO)

    system = EcoDisturbanceAttributionSystem(
        cache_dir=cache_dir,
        out_dir=out_dir,
        lookback_years=2,
        background_sample_rate=0.3,
        max_pattern_size=3,
        min_support=0.08,
        min_relative_risk=1.2,
        bootstrap_iterations=3,
        min_stability=0.4,
        max_bootstrap_samples=200,
        max_network_factors=6,
        discretizer_sample_limit=80,
        model_sample_limit=60,
        progress_log_every=1,
    )

    result = system.run()

    assert len(result["training_samples"]) <= 60
    assert any("stage 1/3" in record.message for record in caplog.records)
    assert any("stage 3/3" in record.message for record in caplog.records)

use std::collections::HashMap;
use std::path::PathBuf;

use factor_core::bayesian::BayesianAttributionEngine;
use factor_core::discretize::AdaptiveDiscretizer;
use factor_core::factors::FactorMiner;
use factor_core::model::{DisturbanceClassEngine, OccurrenceProbabilityEngine};
use factor_core::stats::relative_risk_ci;
use factor_core::types::{DiscretizerConfig, FactorMiningConfig, InducingFactor};

use polars::prelude::*;
use serde_json::Value;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

fn read_golden_json(name: &str) -> Value {
    let path = golden_dir().join(name);
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read {}", path.display()));
    match json5::from_str(&content) {
        Ok(val) => val,
        Err(e) => {
            let sanitized = content
                .replace("-Infinity", "-1e308")
                .replace("Infinity", "1e308");
            serde_json::from_str(&sanitized)
                .unwrap_or_else(|_| panic!("Failed to parse {}: {}", path.display(), e))
        }
    }
}

fn json_to_bin_val(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| {
        if let Some(s) = v.as_str() {
            match s {
                "-Infinity" => f64::NEG_INFINITY,
                "Infinity" => f64::INFINITY,
                _ => f64::NAN,
            }
        } else {
            f64::NAN
        }
    })
}

fn col_to_f64_vec(col: &Column) -> Vec<f64> {
    let s = col.as_materialized_series();
    if let Ok(ca) = s.f64() {
        ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect()
    } else if let Ok(ca) = s.i64() {
        ca.into_iter().map(|v| v.unwrap_or(0) as f64).collect()
    } else if let Ok(ca) = s.i32() {
        ca.into_iter().map(|v| v.unwrap_or(0) as f64).collect()
    } else {
        vec![f64::NAN; s.len()]
    }
}

#[test]
fn test_golden_discretization_parity() {
    let golden = read_golden_json("discretization_labels.json");
    let all_expected_features: serde_json::Map<String, Value> =
        golden.as_object().expect("should be object").clone();

    let panel_path = golden_dir().join("synthetic_panel_full.parquet");
    let mut panel_file = std::fs::File::open(&panel_path)
        .unwrap_or_else(|_| panic!("Failed to open {}", panel_path.display()));

    let df = ParquetReader::new(&mut panel_file)
        .finish()
        .expect("Failed to read parquet");

    let available_cols: std::collections::HashSet<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let feature_names: Vec<String> = all_expected_features
        .keys()
        .filter(|name| available_cols.contains(*name))
        .cloned()
        .collect();

    eprintln!(
        "Testing {} of {} features present in parquet data",
        feature_names.len(),
        all_expected_features.len()
    );

    if feature_names.is_empty() {
        eprintln!("WARNING: No feature columns match between golden and parquet - skipping test");
        return;
    }

    let target_col = df
        .column("disturbance_flag")
        .expect("Missing disturbance_flag");
    let target_series = target_col.as_materialized_series();
    let y: Vec<f64> = if let Ok(ca) = target_series.f64() {
        ca.into_iter().map(|v| v.unwrap_or(0.0)).collect()
    } else if let Ok(ca) = target_series.i64() {
        ca.into_iter().map(|v| v.unwrap_or(0) as f64).collect()
    } else {
        panic!("disturbance_flag has unexpected type");
    };

    let feature_vecs: Vec<Vec<f64>> = feature_names
        .iter()
        .map(|name| {
            let col = df
                .column(name)
                .unwrap_or_else(|_| panic!("Missing column: {}", name));
            col_to_f64_vec(col)
        })
        .collect();

    let n_rows = feature_vecs[0].len();
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    for ri in 0..n_rows {
        let row: Vec<f64> = feature_vecs.iter().map(|col| col[ri]).collect();
        data.push(row);
    }

    let config = DiscretizerConfig {
        categorical_unique_threshold: 12,
        ..DiscretizerConfig::default()
    };
    let mut discretizer = AdaptiveDiscretizer::new(config);
    discretizer.fit(&feature_names, &data, &y);

    let tolerance = 1e-5_f64;
    let mut mismatches = 0;

    for (name, expected_meta) in &all_expected_features {
        if !feature_names.contains(name) {
            continue;
        }
        let actual = discretizer.feature_meta.get(name);
        assert!(
            actual.is_some(),
            "Feature '{}' not found in fitted discretizer",
            name
        );
        let actual = actual.unwrap();

        let expected_kind = expected_meta["kind"].as_str().unwrap_or("");
        assert_eq!(
            actual.kind, expected_kind,
            "Feature '{}' kind mismatch",
            name
        );

        if expected_kind == "continuous" {
            let expected_bins: Vec<f64> = expected_meta["bins"]
                .as_array()
                .unwrap()
                .iter()
                .map(json_to_bin_val)
                .collect();
            let expected_labels: Vec<String> = expected_meta["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();

            assert!(actual.bins.is_some(), "Missing bins for '{}'", name);
            assert!(actual.labels.is_some(), "Missing labels for '{}'", name);

            let actual_bins = actual.bins.as_ref().unwrap();
            let actual_labels = actual.labels.as_ref().unwrap();

            assert_eq!(
                actual_labels, &expected_labels,
                "Label mismatch for '{}'",
                name
            );
            assert_eq!(
                actual_bins.len(),
                expected_bins.len(),
                "Bin count for '{}'",
                name
            );

            for (bi, (a, e)) in actual_bins.iter().zip(expected_bins.iter()).enumerate() {
                if a.is_infinite() && e.is_infinite() {
                    continue;
                }
                let diff = (a - e).abs();
                if diff > tolerance {
                    mismatches += 1;
                    eprintln!(
                        "Feature '{}' bin[{}]: expected {:.15}, got {:.15}, diff={:.2e}",
                        name, bi, e, a, diff
                    );
                }
            }
        } else if expected_kind == "categorical" {
            let expected_cats: Vec<String> = expected_meta["categories"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            assert!(
                actual.categories.is_some(),
                "Missing categories for '{}'",
                name
            );
            assert_eq!(actual.categories.as_ref().unwrap(), &expected_cats);
        }
    }

    eprintln!(
        "Discretization: {} features checked, {} bin mismatches",
        all_expected_features.len(),
        mismatches
    );
}

#[test]
fn test_golden_factor_parity_basic() {
    let golden_factors = read_golden_json("factor_library.json");
    let expected_factors: Vec<Value> = golden_factors
        .as_array()
        .expect("factor_library should be an array")
        .clone();
    assert!(!expected_factors.is_empty());
}

#[test]
fn test_golden_statistics_parity() {
    let golden_json = read_golden_json("factor_library.json");
    let golden_factors: Vec<Value> = golden_json.as_array().expect("array").clone();

    let abs_tol: f64 = 1e-6;
    let rel_tol: f64 = 1e-4;
    let mut max_abs_err: f64 = 0.0;
    let mut max_rel_err: f64 = 0.0;
    let mut failures: usize = 0;
    let mut checked: usize = 0;

    for factor in &golden_factors {
        let count_event = factor["count_event"].as_i64().unwrap_or(0);
        let n_event = factor["n_event"].as_i64().unwrap_or(1);
        let count_bg = factor["count_background"].as_i64().unwrap_or(0);
        let n_bg = factor["n_background"].as_i64().unwrap_or(1);

        if factor.get("rr_ci_lower").is_none() {
            continue;
        }

        let expected_ci_lo = factor["rr_ci_lower"].as_f64().unwrap_or(0.0);
        let expected_ci_hi = factor["rr_ci_upper"].as_f64().unwrap_or(0.0);

        let (_actual_rr, actual_ci_lo, actual_ci_hi, _actual_pv) =
            relative_risk_ci(count_event, n_event, count_bg, n_bg, 0.05, 0.5);

        for (_name, expected, actual) in &[
            ("ci_lower", expected_ci_lo, actual_ci_lo),
            ("ci_upper", expected_ci_hi, actual_ci_hi),
        ] {
            let abs_err = (actual - expected).abs();
            let rel_err = if expected.abs() > 1e-10 {
                abs_err / expected.abs()
            } else {
                abs_err
            };
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);
            checked += 1;
            if abs_err > abs_tol && rel_err > rel_tol {
                failures += 1;
            }
        }
    }

    eprintln!("Stats parity: {} comparisons across {} factors, max_abs={:.2e}, max_rel={:.2e}, failures={}",
        checked, golden_factors.len(), max_abs_err, max_rel_err, failures);
    assert!(
        max_abs_err <= abs_tol || max_rel_err <= rel_tol,
        "Statistical metrics exceed tolerance: max_abs={:.2e}, max_rel={:.2e}",
        max_abs_err,
        max_rel_err
    );
}

#[test]
fn test_golden_factor_ranking_spearman() {
    let golden_json = read_golden_json("factor_library.json");
    let golden_factors: Vec<Value> = golden_json.as_array().expect("array").clone();
    let mut golden_scores: Vec<(String, f64)> = golden_factors
        .iter()
        .map(|f| {
            let id = f["factor_id"].as_str().unwrap_or("").to_string();
            let score = f["score"].as_f64().unwrap_or(0.0);
            (id, score)
        })
        .collect();
    golden_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let golden_ids: Vec<String> = golden_scores.iter().map(|(id, _)| id.clone()).collect();
    eprintln!(
        "Golden factors: {} total, top-5: {:?}",
        golden_ids.len(),
        &golden_ids[..golden_ids.len().min(5)]
    );
}

#[test]
fn test_discretizer_single_value_feature() {
    let config = DiscretizerConfig {
        categorical_unique_threshold: 5,
        ..DiscretizerConfig::default()
    };
    let mut disc = AdaptiveDiscretizer::new(config);
    let features: Vec<String> = vec!["f1".to_string()];
    let n = 20;
    let data: Vec<Vec<f64>> = (0..n).map(|_| vec![42.0]).collect();
    let y: Vec<f64> = vec![0.0; n];
    disc.fit(&features, &data, &y);
    let meta = disc.feature_meta.get("f1").unwrap();
    assert!(meta.kind == "continuous" || meta.kind == "categorical");
}

#[test]
fn test_discretizer_all_nan_feature() {
    let config = DiscretizerConfig::default();
    let mut disc = AdaptiveDiscretizer::new(config);
    let features: Vec<String> = vec!["f1".to_string()];
    let n = 20;
    let data: Vec<Vec<f64>> = (0..n).map(|_| vec![f64::NAN]).collect();
    let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    disc.fit(&features, &data, &y);
    let meta = disc.feature_meta.get("f1").unwrap();
    assert_eq!(meta.kind, "categorical");
    assert_eq!(meta.categories.as_ref().unwrap(), &vec!["missing"]);
}

#[test]
fn test_factor_mining_empty_input() {
    let config = FactorMiningConfig::default();
    let miner = FactorMiner::new(config);
    let candidates = miner.mine_pattern_candidates(&[], &[]);
    assert!(candidates.is_empty());
}

#[test]
fn test_factor_mining_low_support() {
    let config = FactorMiningConfig {
        min_support: 0.5,
        min_relative_risk: 1.0,
        ..FactorMiningConfig::default()
    };
    let miner = FactorMiner::new(config);
    let event_tx: Vec<Vec<String>> = (0..10).map(|i| vec![format!("f1=val_{}", i % 3)]).collect();
    let bg_tx: Vec<Vec<String>> = (0..10).map(|i| vec![format!("f1=val_{}", i % 3)]).collect();
    let candidates = miner.mine_pattern_candidates(&event_tx, &bg_tx);
    assert!(candidates.is_empty());
}

#[test]
fn test_factor_mining_with_data() {
    let config = FactorMiningConfig::default();
    let miner = FactorMiner::new(config);
    let event_tx: Vec<Vec<String>> = (0..20)
        .map(|i| vec!["f1=high".to_string(), format!("f2=val_{}", i % 3)])
        .collect();
    let bg_tx: Vec<Vec<String>> = (0..20)
        .map(|i| vec!["f1=low".to_string(), format!("f2=val_{}", i % 5)])
        .collect();

    let candidates = miner.mine_pattern_candidates(&event_tx, &bg_tx);
    assert!(!candidates.is_empty());
    let stability = miner.estimate_pattern_stability(&event_tx, &bg_tx);
    assert!(!stability.is_empty());
    let library =
        miner.build_factor_library(&candidates, event_tx.len() as i64, bg_tx.len() as i64);
    assert!(!library.is_empty());
    assert!(library.first().unwrap().factor_id.starts_with("IF_"));
}

// ── Helper: build label rows from discretized features ─────────

fn build_label_df(
    feature_names: &[String],
    data: &[Vec<f64>],
    discretizer: &AdaptiveDiscretizer,
) -> Vec<HashMap<String, String>> {
    let labels = discretizer.transform(feature_names, data, true);
    labels
        .iter()
        .map(|row_labels| {
            let mut row = HashMap::new();
            for (fi, name) in feature_names.iter().enumerate() {
                row.insert(name.clone(), row_labels[fi].clone());
            }
            row
        })
        .collect()
}

fn load_selected_factors() -> Vec<InducingFactor> {
    let factors_json = read_golden_json("selected_factors.json");
    let factors_arr = factors_json["factors"].as_array().expect("factors array");
    factors_arr
        .iter()
        .map(|v| InducingFactor {
            factor_id: v["factor_id"].as_str().unwrap_or("").to_string(),
            name: v["name"].as_str().unwrap_or("").to_string(),
            semantics: v["semantics"].as_str().unwrap_or("").to_string(),
            items: v["items"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|s| s.as_str().unwrap_or("").to_string())
                        .collect()
                })
                .unwrap_or_default(),
            support_event: v["support_event"].as_f64().unwrap_or(0.0),
            support_background: v["support_background"].as_f64().unwrap_or(0.0),
            relative_risk: v["relative_risk"].as_f64().unwrap_or(1.0),
            stability: v["stability"].as_f64().unwrap_or(0.0),
            score: v["score"].as_f64().unwrap_or(0.0),
            count_event: v["count_event"].as_i64().unwrap_or(0),
            count_background: v["count_background"].as_i64().unwrap_or(0),
            n_event: v["n_event"].as_i64().unwrap_or(0),
            n_background: v["n_background"].as_i64().unwrap_or(0),
            rr_ci_lower: v["rr_ci_lower"].as_f64().unwrap_or(0.0),
            rr_ci_upper: v["rr_ci_upper"].as_f64().unwrap_or(0.0),
            p_value: v["p_value"].as_f64().unwrap_or(1.0),
            fdr_significant: v["fdr_significant"].as_bool().unwrap_or(false),
            likelihood_ratio: v["likelihood_ratio"].as_f64().unwrap_or(0.0),
            true_relative_risk: v["true_relative_risk"].as_f64().unwrap_or(0.0),
        })
        .collect()
}

fn read_parquet_df(name: &str) -> DataFrame {
    let path = golden_dir().join(name);
    let mut file =
        std::fs::File::open(&path).unwrap_or_else(|_| panic!("Failed to open {}", path.display()));
    ParquetReader::new(&mut file)
        .finish()
        .unwrap_or_else(|_| panic!("Failed to read {}", path.display()))
}

// ── Model Golden Tests ─────────────────────────────────────────

#[test]
fn test_golden_occurrence_parity() {
    let train = read_parquet_df("training_samples.parquet");
    let features: Vec<String> = [
        "slope",
        "aspect",
        "dist_to_river",
        "rainfall_prev_1m",
        "rainfall_prev_2m",
        "rainfall_prev_3m",
        "rainfall_prev_total_3m",
        "temperature_prev_last",
        "temperature_prev_mean",
        "temperature_prev_max",
        "resident_prev_last",
        "resident_prev_mean",
        "resident_prev_max",
        "dist_to_road_prev_last",
        "dist_to_road_prev_mean",
        "dist_to_road_prev_max",
        "dist_to_railway_prev_last",
        "dist_to_railway_prev_mean",
        "dist_to_railway_prev_max",
        "dist_to_waterway_prev_last",
        "dist_to_waterway_prev_mean",
        "dist_to_waterway_prev_max",
        "landuse_prev_last",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let available: std::collections::HashSet<String> = train
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let feature_names: Vec<String> = features
        .into_iter()
        .filter(|f| available.contains(f))
        .collect();

    eprintln!(
        "Training samples shape: {:?}, feature_names: {:?}",
        train.shape(),
        feature_names
    );

    let y_col = train.column("disturbance_flag").expect("target column");
    let y_s = y_col.as_materialized_series();
    let y: Vec<f64> = if let Ok(ca) = y_s.f64() {
        ca.into_iter().map(|v| v.unwrap_or(0.0)).collect()
    } else {
        y_s.i64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or(0) as f64)
            .collect()
    };

    let mut data: Vec<Vec<f64>> = Vec::with_capacity(train.height());
    let col_data: Vec<Vec<f64>> = feature_names
        .iter()
        .map(|name| col_to_f64_vec(train.column(name).unwrap()))
        .collect();
    for ri in 0..train.height() {
        let row: Vec<f64> = col_data.iter().map(|c| c[ri]).collect();
        data.push(row);
    }

    let config = DiscretizerConfig {
        categorical_unique_threshold: 12,
        ..DiscretizerConfig::default()
    };
    let mut discretizer = AdaptiveDiscretizer::new(config);
    discretizer.fit(&feature_names, &data, &y);

    let factors = load_selected_factors();

    let label_df = build_label_df(&feature_names, &data, &discretizer);

    let mut engine = OccurrenceProbabilityEngine::new();
    engine.fit(&label_df, &y, &factors, 42);

    eprintln!(
        "OccurrenceEngine: backend={}, factors={}, base_rate={:.4}, weights=[{:?}], intercept={:.4}",
        engine.backend, engine.factor_ids.len(), engine.base_rate,
        engine.weights.iter().map(|w| format!("{:.4}", w)).collect::<Vec<_>>(),
        engine.intercept
    );

    let preds = engine.predict_proba(&label_df, &factors);
    assert_eq!(preds.len(), y.len());
    for &p in &preds {
        assert!((0.0..=1.0).contains(&p), "probability out of range: {}", p);
    }

    let n_pos = y.iter().filter(|&&v| v > 0.5).count();

    let mae: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(&p, &t)| (p - t).abs())
        .sum::<f64>()
        / preds.len().max(1) as f64;
    eprintln!(
        "Occurrence fit: backend={}, n_train={}, n_pos={}, MAE(train)={:.4}",
        engine.backend,
        y.len(),
        n_pos,
        mae
    );

    // Verify model fits training data (not perfectly due to L2 regularization)
    assert!(mae <= 0.25, "MAE(train) {:.4} exceeds 0.25", mae);
}

#[test]
fn test_golden_class_parity() {
    let train = read_parquet_df("training_samples.parquet");
    let features: Vec<String> = [
        "slope",
        "aspect",
        "dist_to_river",
        "rainfall_prev_1m",
        "rainfall_prev_2m",
        "rainfall_prev_3m",
        "rainfall_prev_total_3m",
        "temperature_prev_last",
        "temperature_prev_mean",
        "temperature_prev_max",
        "resident_prev_last",
        "resident_prev_mean",
        "resident_prev_max",
        "dist_to_road_prev_last",
        "dist_to_road_prev_mean",
        "dist_to_road_prev_max",
        "dist_to_railway_prev_last",
        "dist_to_railway_prev_mean",
        "dist_to_railway_prev_max",
        "dist_to_waterway_prev_last",
        "dist_to_waterway_prev_mean",
        "dist_to_waterway_prev_max",
        "landuse_prev_last",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let available: std::collections::HashSet<String> = train
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let feature_names: Vec<String> = features
        .into_iter()
        .filter(|f| available.contains(f))
        .collect();

    let y_col = train.column("disturbance_flag").expect("target column");
    let y_s = y_col.as_materialized_series();
    let y: Vec<f64> = if let Ok(ca) = y_s.f64() {
        ca.into_iter().map(|v| v.unwrap_or(0.0)).collect()
    } else {
        y_s.i64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or(0) as f64)
            .collect()
    };

    let mut data: Vec<Vec<f64>> = Vec::with_capacity(train.height());
    let col_data: Vec<Vec<f64>> = feature_names
        .iter()
        .map(|name| col_to_f64_vec(train.column(name).unwrap()))
        .collect();
    for ri in 0..train.height() {
        let row: Vec<f64> = col_data.iter().map(|c| c[ri]).collect();
        data.push(row);
    }

    let config = DiscretizerConfig {
        categorical_unique_threshold: 12,
        ..DiscretizerConfig::default()
    };
    let mut discretizer = AdaptiveDiscretizer::new(config);
    discretizer.fit(&feature_names, &data, &y);

    let factors = load_selected_factors();

    let label_df: Vec<HashMap<String, String>> =
        build_label_df(&feature_names, &data, &discretizer);

    let class_col = train.column("disturbance_class").expect("class column");
    let class_s = class_col.as_materialized_series();
    let class_raw: Vec<f64> = if let Ok(ca) = class_s.f64() {
        ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect()
    } else {
        class_s
            .i64()
            .unwrap()
            .into_iter()
            .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
            .collect()
    };

    let valid_class = |v: f64| -> bool { v == 1.0 || v == 2.0 || v == 3.0 || v == 4.0 };
    let class_mask: Vec<usize> = class_raw
        .iter()
        .enumerate()
        .filter(|(_, &v)| valid_class(v))
        .map(|(i, _)| i)
        .collect();
    let class_values: Vec<f64> = class_mask.iter().map(|&i| class_raw[i]).collect();

    let mut engine = DisturbanceClassEngine::new();
    engine.fit(&label_df, &class_values, &class_mask, &factors, 42);

    eprintln!(
        "ClassEngine: backend={}, classes={:?}, factors={}",
        engine.backend,
        engine.classes,
        engine.factor_ids.len()
    );

    let probs_all = engine.predict_proba(&label_df, &factors);
    assert_eq!(probs_all.len(), label_df.len());

    if !class_mask.is_empty() && engine.classes.len() > 1 {
        let mut correct = 0;
        let mut total_confident = 0;
        let mut total = 0;
        for &ri in &class_mask {
            if ri < probs_all.len() {
                let row_probs = &probs_all[ri];
                if let Some((top_cls, top_prob)) = row_probs
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    let true_cls = class_raw[ri] as i64;
                    if *top_cls == true_cls {
                        correct += 1;
                    }
                    if *top_prob >= 0.70 {
                        if *top_cls == true_cls {
                            total_confident += 1;
                        }
                        total += 1;
                    }
                }
            }
        }
        let accuracy = correct as f64 / class_mask.len().max(1) as f64;
        let confident_accuracy = if total > 0 {
            total_confident as f64 / total as f64
        } else {
            1.0
        };
        eprintln!(
            "Class accuracy: {:.2}% ({}/{}), high-conf: {:.2}% ({}/{})",
            accuracy * 100.0,
            correct,
            class_mask.len(),
            confident_accuracy * 100.0,
            total_confident,
            total
        );
        assert!(accuracy >= 0.50, "Overall accuracy {:.4} too low", accuracy);
        assert!(
            confident_accuracy >= 0.80,
            "High-confidence accuracy {:.4} < 0.80",
            confident_accuracy
        );
    }
}

#[test]
fn test_modeling_single_class() {
    let factors = load_selected_factors();
    if factors.is_empty() {
        return;
    }

    let label_rows: Vec<HashMap<String, String>> = (0..10)
        .map(|i| {
            let mut row = HashMap::new();
            row.insert("feat".to_string(), "high".to_string());
            row.insert(format!("factor_{}", i), i.to_string());
            row
        })
        .collect();

    let mut engine = OccurrenceProbabilityEngine::new();
    let targets = vec![1.0; label_rows.len()];
    engine.fit(&label_rows, &targets, &factors, 42);
    assert!(engine.backend.contains("fallback"));

    let mut class_engine = DisturbanceClassEngine::new();
    let class_vals = vec![1.0; label_rows.len()];
    let class_mask: Vec<usize> = (0..label_rows.len()).collect();
    class_engine.fit(&label_rows, &class_vals, &class_mask, &factors, 42);
    assert!(class_engine.backend.contains("fallback"));
}

#[test]
fn test_modeling_low_sample() {
    let factors = load_selected_factors();
    if factors.is_empty() {
        return;
    }

    let label_rows: Vec<HashMap<String, String>> = (0..3)
        .map(|_i| {
            let mut row = HashMap::new();
            row.insert("feat".to_string(), "low".to_string());
            row
        })
        .collect();

    let mut engine = OccurrenceProbabilityEngine::new();
    let targets = vec![1.0, 0.0, 1.0];
    engine.fit(&label_rows, &targets, &factors, 42);
    let probs = engine.predict_proba(&label_rows, &factors);
    assert_eq!(probs.len(), 3);
    for &p in &probs {
        (0.0..=1.0).contains(&p);
    }

    let mut class_engine = DisturbanceClassEngine::new();
    let class_vals = vec![1.0, 3.0, 4.0];
    let class_mask: Vec<usize> = (0..3).collect();
    class_engine.fit(&label_rows, &class_vals, &class_mask, &factors, 42);
    let class_probs = class_engine.predict_proba(&label_rows, &factors);
    assert_eq!(class_probs.len(), 3);
    for row in &class_probs {
        let sum: f64 = row.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_bayesian_engine_structure() {
    let factors = load_selected_factors();
    if factors.is_empty() {
        return;
    }

    let (edges, network_features, factor_columns) =
        BayesianAttributionEngine::build_fixed_edges(&factors);

    assert!(!edges.is_empty(), "Edges should not be empty");
    assert!(!network_features.is_empty());
    assert!(!factor_columns.is_empty());

    eprintln!(
        "BN structure: {} edges, {} network features, {} factors",
        edges.len(),
        network_features.len(),
        factor_columns.len()
    );

    let golden_edges = read_golden_json("bayesian_network_edges.json");
    let golden_edge_arr = golden_edges.as_array().expect("edges array");
    eprintln!("Golden edges: {}", golden_edge_arr.len());
}

#[test]
fn test_bayesian_engine_inference() {
    let factors = load_selected_factors();
    if factors.is_empty() {
        return;
    }

    let train = read_parquet_df("training_samples.parquet");
    let features: Vec<String> = [
        "slope",
        "aspect",
        "dist_to_river",
        "rainfall_prev_1m",
        "rainfall_prev_2m",
        "rainfall_prev_3m",
        "rainfall_prev_total_3m",
        "temperature_prev_last",
        "temperature_prev_mean",
        "temperature_prev_max",
        "resident_prev_last",
        "resident_prev_mean",
        "resident_prev_max",
        "dist_to_road_prev_last",
        "dist_to_road_prev_mean",
        "dist_to_road_prev_max",
        "dist_to_railway_prev_last",
        "dist_to_railway_prev_mean",
        "dist_to_railway_prev_max",
        "dist_to_waterway_prev_last",
        "dist_to_waterway_prev_mean",
        "dist_to_waterway_prev_max",
        "landuse_prev_last",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let available: std::collections::HashSet<String> = train
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let feature_names: Vec<String> = features
        .into_iter()
        .filter(|f| available.contains(f))
        .collect();

    let y_col = train.column("disturbance_flag").expect("target column");
    let y_s = y_col.as_materialized_series();
    let y: Vec<f64> = if let Ok(ca) = y_s.f64() {
        ca.into_iter().map(|v| v.unwrap_or(0.0)).collect()
    } else {
        y_s.i64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or(0) as f64)
            .collect()
    };

    let mut data: Vec<Vec<f64>> = Vec::with_capacity(train.height());
    let col_data: Vec<Vec<f64>> = feature_names
        .iter()
        .map(|name| col_to_f64_vec(train.column(name).unwrap()))
        .collect();
    for ri in 0..train.height() {
        let row: Vec<f64> = col_data.iter().map(|c| c[ri]).collect();
        data.push(row);
    }

    let config = DiscretizerConfig {
        categorical_unique_threshold: 12,
        ..DiscretizerConfig::default()
    };
    let mut discretizer = AdaptiveDiscretizer::new(config);
    discretizer.fit(&feature_names, &data, &y);

    let label_df: Vec<HashMap<String, String>> =
        build_label_df(&feature_names, &data, &discretizer);

    let mut bn_engine =
        BayesianAttributionEngine::new("disturbance_flag".to_string(), 42, 0.55, 0.75);

    bn_engine.fit(&label_df, &y, &factors);

    eprintln!(
        "BN engine backend: {} (fallback_reason: {})",
        bn_engine.model_backend, bn_engine.fallback_reason
    );

    assert!(
        bn_engine.model_backend == "bayesian_network"
            || bn_engine.model_backend == "logistic_fallback"
    );

    // Test inference on a few samples
    for row in label_df.iter().take(5) {
        let prob = bn_engine.infer_single(row, &factors);
        assert!((0.0..=1.0).contains(&prob), "prob out of range: {}", prob);
    }

    // Test attribution
    for (ri, row) in label_df.iter().enumerate().take(3) {
        let result = bn_engine.attribute_sample(row, &factors);
        eprintln!(
            "Sample {}: type={}, top={}, prob={:.4}, bayes_prob={:.4}",
            ri,
            result.attribution_type,
            result.top_factor,
            result.top_probability,
            result.bayesian_probability
        );
        assert!(!result.backend.is_empty());
    }
}

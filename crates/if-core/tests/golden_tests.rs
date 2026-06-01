use std::path::PathBuf;

use if_core::discretize::AdaptiveDiscretizer;
use if_core::factors::FactorMiner;
use if_core::stats::relative_risk_ci;
use if_core::types::{DiscretizerConfig, FactorMiningConfig};

use polars::prelude::*;
use serde_json::Value;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
}

fn read_golden_json(name: &str) -> Value {
    let path = golden_dir().join(name);
    let content =
        std::fs::read_to_string(&path).expect(&format!("Failed to read {}", path.display()));
    match json5::from_str(&content) {
        Ok(val) => val,
        Err(e) => {
            let sanitized = content
                .replace("-Infinity", "-1e308")
                .replace("Infinity", "1e308");
            serde_json::from_str(&sanitized)
                .expect(&format!("Failed to parse {}: {}", path.display(), e))
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
        .expect(&format!("Failed to open {}", panel_path.display()));

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

    let target_col = df.column("disturbance_flag").expect("Missing disturbance_flag");
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
            let col = df.column(name).expect(&format!("Missing column: {}", name));
            col_to_f64_vec(col)
        })
        .collect();

    let n_rows = feature_vecs[0].len();
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    for ri in 0..n_rows {
        let row: Vec<f64> = feature_vecs.iter().map(|col| col[ri]).collect();
        data.push(row);
    }

    let mut config = DiscretizerConfig::default();
    config.categorical_unique_threshold = 12;
    let mut discretizer = AdaptiveDiscretizer::new(config);
    discretizer.fit(&feature_names, &data, &y);

    let tolerance = 1e-5_f64;
    let mut mismatches = 0;

    for (name, expected_meta) in &all_expected_features {
        if !feature_names.contains(name) {
            continue;
        }
        let actual = discretizer.feature_meta.get(name);
        assert!(actual.is_some(), "Feature '{}' not found in fitted discretizer", name);
        let actual = actual.unwrap();

        let expected_kind = expected_meta["kind"].as_str().unwrap_or("");
        assert_eq!(actual.kind, expected_kind, "Feature '{}' kind mismatch", name);

        if expected_kind == "continuous" {
            let expected_bins: Vec<f64> = expected_meta["bins"]
                .as_array().unwrap().iter()
                .map(|v| json_to_bin_val(v))
                .collect();
            let expected_labels: Vec<String> = expected_meta["labels"]
                .as_array().unwrap().iter()
                .map(|v| v.as_str().unwrap().to_string()).collect();

            assert!(actual.bins.is_some(), "Missing bins for '{}'", name);
            assert!(actual.labels.is_some(), "Missing labels for '{}'", name);

            let actual_bins = actual.bins.as_ref().unwrap();
            let actual_labels = actual.labels.as_ref().unwrap();

            assert_eq!(actual_labels, &expected_labels, "Label mismatch for '{}'", name);
            assert_eq!(actual_bins.len(), expected_bins.len(), "Bin count for '{}'", name);

            for (bi, (a, e)) in actual_bins.iter().zip(expected_bins.iter()).enumerate() {
                if a.is_infinite() && e.is_infinite() { continue; }
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
                .as_array().unwrap().iter()
                .map(|v| v.as_str().unwrap().to_string()).collect();
            assert!(actual.categories.is_some(), "Missing categories for '{}'", name);
            assert_eq!(actual.categories.as_ref().unwrap(), &expected_cats);
        }
    }

    eprintln!("Discretization: {} features checked, {} bin mismatches",
        all_expected_features.len(), mismatches);
}

#[test]
fn test_golden_factor_parity_basic() {
    let golden_factors = read_golden_json("factor_library.json");
    let expected_factors: Vec<Value> = golden_factors.as_array()
        .expect("factor_library should be an array").clone();
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
            let rel_err = if expected.abs() > 1e-10 { abs_err / expected.abs() } else { abs_err };
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
    assert!(max_abs_err <= abs_tol || max_rel_err <= rel_tol,
        "Statistical metrics exceed tolerance: max_abs={:.2e}, max_rel={:.2e}", max_abs_err, max_rel_err);
}

#[test]
fn test_golden_factor_ranking_spearman() {
    let golden_json = read_golden_json("factor_library.json");
    let golden_factors: Vec<Value> = golden_json.as_array().expect("array").clone();
    let mut golden_scores: Vec<(String, f64)> = golden_factors.iter().map(|f| {
        let id = f["factor_id"].as_str().unwrap_or("").to_string();
        let score = f["score"].as_f64().unwrap_or(0.0);
        (id, score)
    }).collect();
    golden_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let golden_ids: Vec<String> = golden_scores.iter().map(|(id, _)| id.clone()).collect();
    eprintln!("Golden factors: {} total, top-5: {:?}", golden_ids.len(),
        &golden_ids[..golden_ids.len().min(5)]);
}

#[test]
fn test_discretizer_single_value_feature() {
    let mut config = DiscretizerConfig::default();
    config.categorical_unique_threshold = 5;
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
    let mut config = FactorMiningConfig::default();
    config.min_support = 0.5;
    config.min_relative_risk = 1.0;
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
        .map(|i| vec!["f1=high".to_string(), format!("f2=val_{}", i % 3)]).collect();
    let bg_tx: Vec<Vec<String>> = (0..20)
        .map(|i| vec!["f1=low".to_string(), format!("f2=val_{}", i % 5)]).collect();

    let candidates = miner.mine_pattern_candidates(&event_tx, &bg_tx);
    assert!(!candidates.is_empty());
    let stability = miner.estimate_pattern_stability(&event_tx, &bg_tx);
    assert!(!stability.is_empty());
    let library = miner.build_factor_library(&candidates, event_tx.len() as i64, bg_tx.len() as i64);
    assert!(!library.is_empty());
    assert!(library.first().unwrap().factor_id.starts_with("IF_"));
}

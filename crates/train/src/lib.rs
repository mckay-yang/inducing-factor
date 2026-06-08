//! Training stage boundary.
//!
//! Training consumes preprocessed feature tables and writes model artifacts. It
//! depends on `model` for the artifact contract, but prediction does not
//! depend on this crate.

use factor_core::discretize::AdaptiveDiscretizer;
use factor_core::factors::FactorMiner;
use factor_core::model::{DisturbanceClassEngine, OccurrenceProbabilityEngine};
use factor_core::types::{DiscretizerConfig, FactorMiningConfig, InducingFactor};
use model::BasinModelArtifact;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelMissingDataPolicy {
    NativeMissingCategory,
    FeatureDropout,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainPlan {
    pub training_years: Vec<i32>,
    pub feature_path: PathBuf,
    pub model_path: PathBuf,
    pub missing_data_policy: ModelMissingDataPolicy,
    pub workers: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("model artifact error: {0}")]
    Model(#[from] model::ModelArtifactError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("thread pool error: {0}")]
    ThreadPool(String),
    #[error("missing required column: {0}")]
    MissingColumn(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BasinTrainReport {
    pub model_path: PathBuf,
    pub feature_columns: Vec<String>,
    pub n_rows: usize,
    pub n_events: usize,
    pub n_background: usize,
    pub n_factors: usize,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BasinTrainSettings {
    pub discretizer_method: String,
    pub discretize_bins: usize,
    pub max_pattern_size: usize,
    pub min_support: f64,
    pub min_relative_risk: f64,
    pub min_stability: f64,
    pub bootstrap_iterations: usize,
    pub max_bootstrap_samples: usize,
    pub max_network_factors: usize,
    pub random_state: u64,
    pub missing_data_policy: ModelMissingDataPolicy,
    pub workers: usize,
}

impl BasinTrainSettings {
    pub fn with_missing_policy_and_workers(
        missing_data_policy: ModelMissingDataPolicy,
        workers: usize,
    ) -> Self {
        Self {
            workers,
            missing_data_policy,
            ..Self::default()
        }
    }
}

impl Default for BasinTrainSettings {
    fn default() -> Self {
        Self {
            discretizer_method: "supervised_tree".into(),
            discretize_bins: 3,
            max_pattern_size: 3,
            min_support: 0.05,
            min_relative_risk: 1.5,
            min_stability: 0.6,
            bootstrap_iterations: 100,
            max_bootstrap_samples: 3000,
            max_network_factors: 6,
            random_state: 42,
            missing_data_policy: ModelMissingDataPolicy::NativeMissingCategory,
            workers: 0,
        }
    }
}

pub fn train_basin_with_settings(
    feature_path: &Path,
    out_dir: &Path,
    settings: &BasinTrainSettings,
) -> Result<BasinTrainReport, TrainError> {
    let file = std::fs::File::open(feature_path)?;
    let df = ParquetReader::new(file).finish()?;

    let total_rows = df.height();
    let label_cols = [
        "disturbance_flag".to_string(),
        "disturbance_class".to_string(),
        "row".to_string(),
        "col".to_string(),
        "ccdc_timing_year".to_string(),
        "year".to_string(),
    ];
    let feature_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|c| c.to_string())
        .filter(|c| !label_cols.contains(c))
        .collect();

    if df.column("disturbance_flag").is_err() {
        return Err(TrainError::MissingColumn("disturbance_flag".into()));
    }

    let h = df.height();
    let workers = effective_workers(settings.workers);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| TrainError::ThreadPool(e.to_string()))?;

    let feature_columns_data = collect_feature_columns(&df, &feature_cols, h);
    let targets = collect_targets(&df, h);
    let feature_data: Vec<Vec<f64>> = pool.install(|| {
        (0..h)
            .into_par_iter()
            .map(|row_idx| {
                feature_columns_data
                    .iter()
                    .map(|column| column[row_idx])
                    .collect()
            })
            .collect()
    });

    let n_events = targets.iter().filter(|&&v| v > 0.5).count();
    let n_background = total_rows - n_events;

    let disc_config = DiscretizerConfig {
        method: settings.discretizer_method.clone(),
        n_bins: settings.discretize_bins,
        ..Default::default()
    };
    let mut discretizer = AdaptiveDiscretizer::new(disc_config);
    if !feature_data.is_empty() {
        discretizer.fit(&feature_cols, &feature_data, &targets);
    }

    let all_labels = discretizer.transform(&feature_cols, &feature_data, true);

    let mining_config = FactorMiningConfig {
        max_pattern_size: settings.max_pattern_size,
        min_support: settings.min_support,
        min_relative_risk: settings.min_relative_risk,
        min_stability: settings.min_stability,
        bootstrap_iterations: settings.bootstrap_iterations,
        max_bootstrap_samples: settings.max_bootstrap_samples,
        random_state: settings.random_state,
        score_method: "log_rr".into(),
    };
    let miner = FactorMiner::new(mining_config);

    let transactions = miner.transactions_from_labels(&all_labels, &feature_cols);
    let event_transactions: Vec<Vec<String>> = transactions
        .iter()
        .zip(targets.iter())
        .filter(|(_, &t)| t > 0.5)
        .map(|(tx, _)| tx.clone())
        .collect();
    let bg_transactions: Vec<Vec<String>> = transactions
        .iter()
        .zip(targets.iter())
        .filter(|(_, &t)| t <= 0.5)
        .map(|(tx, _)| tx.clone())
        .collect();

    let pattern_stats = miner.mine_pattern_candidates(&event_transactions, &bg_transactions);
    let pattern_stability = miner.estimate_pattern_stability(&event_transactions, &bg_transactions);
    let reduced_patterns = miner.reduce_redundant_patterns(&pattern_stats, &pattern_stability, 0.1);
    let mut factors = miner.build_factor_library(
        &reduced_patterns,
        event_transactions.len() as i64,
        bg_transactions.len() as i64,
    );

    if factors.is_empty() && !pattern_stats.is_empty() {
        // Fallback keeps model training usable when bootstrap stability is too
        // strict for sparse basin data. These factors still preserve explicit
        // missing states through the discretizer.
        factors = miner.build_factor_library(
            &pattern_stats,
            event_transactions.len() as i64,
            bg_transactions.len() as i64,
        );
    }
    let selected_factors = select_network_factors(factors, settings.max_network_factors);

    let label_dicts: Vec<HashMap<String, String>> = pool.install(|| {
        all_labels
            .par_iter()
            .map(|row| {
                let mut m = HashMap::new();
                for (i, name) in feature_cols.iter().enumerate() {
                    m.insert(name.clone(), row.get(i).cloned().unwrap_or_default());
                }
                m
            })
            .collect()
    });

    let mut occ_engine = OccurrenceProbabilityEngine::new();
    occ_engine.fit(
        &label_dicts,
        &targets,
        &selected_factors,
        settings.random_state,
    );
    let class_training = collect_disturbance_class_targets(&df, h);
    let mut class_engine = DisturbanceClassEngine::new();
    class_engine.fit(
        &label_dicts,
        &class_training.values,
        &class_training.mask,
        &selected_factors,
        settings.random_state,
    );

    let model_path = out_dir.join("basin_model.json");
    let basin_model = BasinModelArtifact {
        version: "0.1.0".into(),
        feature_columns: feature_cols.clone(),
        discretizer,
        selected_factors: selected_factors.clone(),
        occurrence_engine: occ_engine,
        class_engine: Some(class_engine),
        n_rows: total_rows,
        n_events,
        n_background,
        backend: "logistic_regression".into(),
        missing_data_policy: missing_policy_name(&settings.missing_data_policy).into(),
        random_state: settings.random_state,
    };
    basin_model.write_json(&model_path)?;

    std::fs::create_dir_all(&out_dir)?;
    let meta = serde_json::json!({
        "version": "0.1.0",
        "feature_columns": &feature_cols,
        "n_rows": total_rows,
        "n_events": n_events,
        "n_background": n_background,
        "backend": "logistic_regression",
        "missing_data_policy": missing_policy_name(&settings.missing_data_policy),
        "random_state": settings.random_state,
    });
    std::fs::write(
        out_dir.join("basin_model_metadata.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;

    Ok(BasinTrainReport {
        model_path,
        feature_columns: feature_cols,
        n_rows: total_rows,
        n_events,
        n_background,
        n_factors: selected_factors.len(),
        workers,
    })
}

fn collect_feature_columns(
    df: &DataFrame,
    feature_cols: &[String],
    n_rows: usize,
) -> Vec<Vec<f64>> {
    feature_cols
        .iter()
        .map(|name| collect_numeric_feature_column(df, name, n_rows))
        .collect()
}

fn collect_numeric_feature_column(df: &DataFrame, name: &str, n_rows: usize) -> Vec<f64> {
    let Ok(column) = df.column(name) else {
        return vec![f64::NAN; n_rows];
    };
    let Ok(cast_column) = column.cast(&DataType::Float64) else {
        return vec![f64::NAN; n_rows];
    };
    let Ok(series) = cast_column.f64() else {
        return vec![f64::NAN; n_rows];
    };

    (0..n_rows)
        .map(|row_idx| series.get(row_idx).unwrap_or(f64::NAN))
        .collect()
}

fn collect_targets(df: &DataFrame, n_rows: usize) -> Vec<f64> {
    let targets = df
        .column("disturbance_flag")
        .ok()
        .and_then(|c| c.i32().ok());
    (0..n_rows)
        .map(|row_idx| {
            targets
                .as_ref()
                .and_then(|s| s.get(row_idx))
                .map(|v| v as f64)
                .unwrap_or(0.0)
        })
        .collect()
}

struct ClassTargets {
    values: Vec<f64>,
    mask: Vec<usize>,
}

fn collect_disturbance_class_targets(df: &DataFrame, n_rows: usize) -> ClassTargets {
    let flags = df
        .column("disturbance_flag")
        .ok()
        .and_then(|c| c.i32().ok());
    let classes = df
        .column("disturbance_class")
        .ok()
        .and_then(|c| c.i32().ok());

    let mut values = Vec::new();
    let mut mask = Vec::new();
    for row_idx in 0..n_rows {
        let flag = flags.as_ref().and_then(|s| s.get(row_idx)).unwrap_or(0);
        let class_value = classes.as_ref().and_then(|s| s.get(row_idx)).unwrap_or(0);
        if flag > 0 && (1..=4).contains(&class_value) {
            values.push(class_value as f64);
            mask.push(row_idx);
        }
    }

    ClassTargets { values, mask }
}

fn effective_workers(configured_workers: usize) -> usize {
    if configured_workers > 0 {
        configured_workers
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
    .max(1)
}

fn missing_policy_name(policy: &ModelMissingDataPolicy) -> &'static str {
    match policy {
        ModelMissingDataPolicy::NativeMissingCategory => "native_missing_category",
        ModelMissingDataPolicy::FeatureDropout => "feature_dropout",
    }
}

fn select_network_factors(factors: Vec<InducingFactor>, max_factors: usize) -> Vec<InducingFactor> {
    if max_factors == 0 || factors.is_empty() {
        return Vec::new();
    }

    let singleton_quota = (max_factors / 4).max(1);
    let mut selected = Vec::with_capacity(max_factors.min(factors.len()));
    let mut selected_names = std::collections::HashSet::new();

    for factor in factors.iter().filter(|factor| factor.items.len() == 1) {
        if selected.len() >= singleton_quota || selected.len() >= max_factors {
            break;
        }
        selected_names.insert(factor.name.clone());
        selected.push(factor.clone());
    }

    for factor in factors {
        if selected.len() >= max_factors {
            break;
        }
        if selected_names.insert(factor.name.clone()) {
            selected.push(factor);
        }
    }

    for (idx, factor) in selected.iter_mut().enumerate() {
        factor.factor_id = format!("IF_{:03}", idx + 1);
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use factor_core::types::InducingFactor;

    #[test]
    fn workers_flow_into_settings() {
        let settings = BasinTrainSettings::with_missing_policy_and_workers(
            ModelMissingDataPolicy::NativeMissingCategory,
            4,
        );
        assert_eq!(settings.workers, 4);
    }

    #[test]
    fn zero_workers_means_auto_at_least_one() {
        assert!(effective_workers(0) >= 1);
        assert_eq!(effective_workers(2), 2);
    }

    #[test]
    fn network_factor_selection_keeps_singletons_and_cooccurrences() {
        let mut factors = vec![
            factor("co1", vec!["a=high", "b=low"], 10.0),
            factor("co2", vec!["a=high", "c=medium"], 9.0),
            factor("co3", vec!["d=high", "e=low"], 8.0),
            factor("single1", vec!["a=high"], 1.0),
            factor("single2", vec!["b=low"], 0.9),
        ];
        for (idx, factor) in factors.iter_mut().enumerate() {
            factor.factor_id = format!("IF_{:03}", idx + 1);
        }

        let selected = select_network_factors(factors, 4);
        assert_eq!(selected.len(), 4);
        assert!(selected.iter().any(|factor| factor.items.len() == 1));
        assert!(selected.iter().any(|factor| factor.items.len() > 1));
    }

    fn factor(name: &str, items: Vec<&str>, score: f64) -> InducingFactor {
        InducingFactor {
            factor_id: String::new(),
            name: name.into(),
            semantics: name.into(),
            items: items.into_iter().map(String::from).collect(),
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 1.0,
            score,
            count_event: 10,
            count_background: 1,
            n_event: 20,
            n_background: 20,
            rr_ci_lower: 1.0,
            rr_ci_upper: 10.0,
            p_value: 0.01,
            fdr_significant: true,
            likelihood_ratio: 5.0,
            true_relative_risk: 5.0,
        }
    }
}

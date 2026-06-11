//! Training stage boundary.
//!
//! Training consumes preprocessed feature tables and writes model artifacts. It
//! depends on `model` for the artifact contract, but prediction does not
//! depend on this crate.

use factor_core::discretize::AdaptiveDiscretizer;
use factor_core::model::OccurrenceProbabilityEngine;
use factor_core::types::DiscretizerConfig;
#[cfg(test)]
use factor_core::types::InducingFactor;
use model::BasinModelArtifact;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

macro_rules! status {
    ($($arg:tt)*) => {{
        log_status(&format!($($arg)*));
    }};
}

fn log_status(message: &str) {
    eprintln!("[{}] {}", timestamp(), message);
}

fn timestamp() -> String {
    Command::new("date")
        .arg("+%Y-%m-%d %H:%M:%S")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown-time".to_string())
}

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
    #[error("interaction mining error: {0}")]
    Interaction(String),
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
    status!(
        "[train] start: features={}, out_dir={}, workers={}, max_network_factors={}",
        feature_path.display(),
        out_dir.display(),
        settings.workers,
        settings.max_network_factors
    );
    let file = std::fs::File::open(feature_path)?;
    let mut df = ParquetReader::new(file).finish()?;
    if let Ok(max_rows_raw) = std::env::var("BASIN_TRAIN_MAX_ROWS") {
        if let Ok(max_rows) = max_rows_raw.parse::<usize>() {
            if max_rows > 0 && max_rows < df.height() {
                status!(
                    "[train] smoke row limit active: using first {} / {} rows from cached features",
                    max_rows,
                    df.height()
                );
                df = df.head(Some(max_rows));
            }
        }
    }

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

    status!(
        "[train] loaded feature table: rows={}, columns={}, feature_columns={}",
        total_rows,
        df.width(),
        feature_cols.len()
    );

    if df.column("disturbance_flag").is_err() {
        return Err(TrainError::MissingColumn("disturbance_flag".into()));
    }

    let h = df.height();
    let workers = effective_workers(settings.workers);
    status!(
        "[train] initialize thread pool: effective_workers={}",
        workers
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| TrainError::ThreadPool(e.to_string()))?;

    status!(
        "[train] collect feature columns: rows={}, feature_columns={}",
        h,
        feature_cols.len()
    );
    let feature_columns_data = collect_feature_columns(&df, &feature_cols, h);
    let targets = collect_targets(&df, h);
    status!(
        "[train] build dense feature matrix: rows={}, feature_columns={}, workers={}",
        h,
        feature_cols.len(),
        workers
    );
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
    status!(
        "[train] labels: events={}, background={}, event_rate={:.4}",
        n_events,
        n_background,
        if total_rows > 0 {
            n_events as f64 / total_rows as f64
        } else {
            0.0
        }
    );

    status!(
        "[train] train numeric vector occurrence model: rows={}, dimensions={}",
        feature_data.len(),
        feature_cols.len()
    );
    let mut occ_engine = OccurrenceProbabilityEngine::new();
    let train_progress = |message: &str| log_status(message);
    occ_engine.fit_numeric_with_progress(
        &feature_cols,
        &feature_data,
        &targets,
        settings.random_state,
        &train_progress,
    );
    let selected_factors = Vec::new();
    let discretizer = AdaptiveDiscretizer::new(DiscretizerConfig {
        method: "none_numeric_vector".into(),
        n_bins: settings.discretize_bins,
        ..Default::default()
    });
    status!(
        "[train] numeric occurrence model fitted: backend={}, weights={}",
        occ_engine.backend,
        occ_engine.weights.len()
    );

    let class_training = collect_disturbance_class_targets(&df, h);
    status!(
        "[train] class labels usable for disturbance type model: {}",
        class_training.mask.len()
    );
    status!("[train] skip discrete factor class model for numeric-vector run");

    let model_path = out_dir.join("basin_model.json");
    let basin_model = BasinModelArtifact {
        version: "0.1.0".into(),
        feature_columns: feature_cols.clone(),
        discretizer,
        selected_factors: selected_factors.clone(),
        occurrence_engine: occ_engine,
        class_engine: None,
        n_rows: total_rows,
        n_events,
        n_background,
        backend: "numeric_logistic_regression".into(),
        missing_data_policy: missing_policy_name(&settings.missing_data_policy).into(),
        random_state: settings.random_state,
    };
    basin_model.write_json(&model_path)?;
    status!("[train] model written: {}", model_path.display());

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

    mine_interaction_factors(
        out_dir,
        &feature_cols,
        &feature_data,
        &targets,
        &basin_model.occurrence_engine,
        workers,
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

pub fn ensure_interaction_factors(
    feature_path: &Path,
    model_path: &Path,
    out_dir: &Path,
    workers: usize,
) -> Result<(), TrainError> {
    let result_path = out_dir.join("interaction_factors.json");
    if file_is_nonempty(&result_path) {
        status!("[interactions] exists, skip: {}", result_path.display());
        return Ok(());
    }

    status!(
        "[interactions] start from cached features={}, model={}",
        feature_path.display(),
        model_path.display()
    );
    let model = BasinModelArtifact::read_json(model_path)?;
    let file = std::fs::File::open(feature_path)?;
    let df = ParquetReader::new(file).finish()?;
    let feature_cols = model.occurrence_engine.numeric_feature_columns.clone();
    if feature_cols.is_empty() {
        return Err(TrainError::Interaction(
            "numeric baseline model has no feature columns".into(),
        ));
    }
    let h = df.height();
    let feature_columns_data = collect_feature_columns(&df, &feature_cols, h);
    let targets = collect_targets(&df, h);
    let effective = effective_workers(workers);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(effective)
        .build()
        .map_err(|e| TrainError::ThreadPool(e.to_string()))?;
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

    mine_interaction_factors(
        out_dir,
        &feature_cols,
        &feature_data,
        &targets,
        &model.occurrence_engine,
        effective,
    )
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

    let mut mask = Vec::new();
    for row_idx in 0..n_rows {
        let flag = flags.as_ref().and_then(|s| s.get(row_idx)).unwrap_or(0);
        let class_value = classes.as_ref().and_then(|s| s.get(row_idx)).unwrap_or(0);
        if flag > 0 && (1..=4).contains(&class_value) {
            mask.push(row_idx);
        }
    }

    ClassTargets { mask }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InteractionSignature {
    feature_names: Vec<String>,
    feature_means: Vec<f64>,
    feature_scales: Vec<f64>,
    risk_directions: Vec<f64>,
    activation_gamma: f64,
    sample_rows: usize,
    positive_rows: usize,
    background_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InteractionFactorStat {
    rank: usize,
    order: usize,
    factors: Vec<String>,
    feature_indices: Vec<usize>,
    support_pos: f64,
    support_neg: f64,
    lift: f64,
    log_lift: f64,
    stability: f64,
    score: f64,
    direction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateStat {
    indices: Vec<usize>,
    support_pos: f64,
    support_neg: f64,
    lift: f64,
    log_lift: f64,
    score: f64,
    parent_score: f64,
}

fn mine_interaction_factors(
    out_dir: &Path,
    feature_cols: &[String],
    feature_data: &[Vec<f64>],
    targets: &[f64],
    engine: &OccurrenceProbabilityEngine,
    workers: usize,
) -> Result<(), TrainError> {
    let result_path = out_dir.join("interaction_factors.json");
    if file_is_nonempty(&result_path) {
        status!("[interactions] exists, skip: {}", result_path.display());
        return Ok(());
    }
    if feature_data.is_empty() || feature_cols.is_empty() || engine.weights.is_empty() {
        status!("[interactions] skipped: missing numeric baseline inputs");
        return Ok(());
    }

    let cache_dir = PathBuf::from(".cache/interactions");
    let tables_dir = PathBuf::from("results/tables");
    std::fs::create_dir_all(&cache_dir)?;
    std::fs::create_dir_all(&tables_dir)?;

    let n_cols = feature_cols
        .len()
        .min(engine.weights.len())
        .min(engine.numeric_means.len())
        .min(engine.numeric_scales.len());
    let gamma = 1.5;
    let max_per_class = std::env::var("INTERACTION_MAX_PER_CLASS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(250_000);
    let pair_beam = std::env::var("INTERACTION_PAIR_BEAM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(300);
    let higher_order_beam = std::env::var("INTERACTION_HIGHER_ORDER_BEAM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(600);
    let final_top = std::env::var("INTERACTION_FINAL_TOP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(200);

    let sample_indices = stratified_sample_indices(targets, max_per_class);
    let sample_pos = sample_indices
        .iter()
        .filter(|&&idx| targets.get(idx).copied().unwrap_or(0.0) > 0.5)
        .count();
    let sample_neg = sample_indices.len().saturating_sub(sample_pos);
    if sample_pos == 0 || sample_neg == 0 {
        status!("[interactions] skipped: sampled labels lack positive/background rows");
        return Ok(());
    }

    status!(
        "[interactions] sampled rows: total={}, positive={}, background={}, features={}, gamma={}",
        sample_indices.len(),
        sample_pos,
        sample_neg,
        n_cols,
        gamma
    );

    let activation = build_risk_activation_matrix(
        feature_data,
        &sample_indices,
        &engine.numeric_means[..n_cols],
        &engine.numeric_scales[..n_cols],
        &engine.weights[..n_cols],
        gamma,
    );
    let sample_labels: Vec<bool> = sample_indices
        .iter()
        .map(|&idx| targets.get(idx).copied().unwrap_or(0.0) > 0.5)
        .collect();

    let signature = InteractionSignature {
        feature_names: feature_cols[..n_cols].to_vec(),
        feature_means: engine.numeric_means[..n_cols].to_vec(),
        feature_scales: engine.numeric_scales[..n_cols].to_vec(),
        risk_directions: engine.weights[..n_cols]
            .iter()
            .map(|&w| if w < 0.0 { -1.0 } else { 1.0 })
            .collect(),
        activation_gamma: gamma,
        sample_rows: sample_indices.len(),
        positive_rows: sample_pos,
        background_rows: sample_neg,
    };
    write_json_pretty(cache_dir.join("feature_signature.json"), &signature)?;

    status!("[interactions] compute order-1/order-2 continuous supports");
    let order1 = compute_order1_stats(&activation, &sample_labels, n_cols);
    let order2 = compute_order2_stats(&activation, &sample_labels, n_cols);
    write_json_pretty(cache_dir.join("order_1_stats.json"), &order1)?;
    write_json_pretty(cache_dir.join("order_2_stats.json"), &order2)?;

    let strong_pairs: Vec<CandidateStat> = order2
        .iter()
        .filter(|s| s.lift > 1.0 && s.support_pos > 0.0 && s.score.is_finite())
        .take(pair_beam)
        .cloned()
        .collect();
    let strong_pair_keys: HashSet<Vec<usize>> =
        strong_pairs.iter().map(|s| s.indices.clone()).collect();

    status!(
        "[interactions] build order-3 candidates from {} strong pairs",
        strong_pairs.len()
    );
    let order3_candidates =
        build_order3_candidates(n_cols, &strong_pair_keys, &strong_pairs, feature_cols);
    write_json_pretty(
        cache_dir.join("order_3_candidates.json"),
        &candidate_indices(&order3_candidates),
    )?;
    let order3 = compute_candidate_stats(
        &activation,
        &sample_labels,
        n_cols,
        &order3_candidates,
        workers,
        "order-3",
    );
    write_json_pretty(cache_dir.join("order_3_stats.json"), &order3)?;

    let strong_triples: Vec<CandidateStat> = order3
        .iter()
        .filter(|s| s.lift > 1.0 && s.support_pos > 0.0 && s.score.is_finite())
        .take(higher_order_beam)
        .cloned()
        .collect();
    let strong_triple_keys: HashSet<Vec<usize>> =
        strong_triples.iter().map(|s| s.indices.clone()).collect();

    status!(
        "[interactions] build order-4 candidates from {} strong triples",
        strong_triples.len()
    );
    let order4_candidates =
        build_order4_candidates(n_cols, &strong_triple_keys, &strong_triples, feature_cols);
    write_json_pretty(
        cache_dir.join("order_4_candidates.json"),
        &candidate_indices(&order4_candidates),
    )?;
    let order4 = compute_candidate_stats(
        &activation,
        &sample_labels,
        n_cols,
        &order4_candidates,
        workers,
        "order-4",
    );
    write_json_pretty(cache_dir.join("order_4_stats.json"), &order4)?;

    let mut all = Vec::new();
    all.extend(order1.into_iter().filter(|s| s.indices.len() >= 2));
    all.extend(order2.into_iter().take(pair_beam));
    all.extend(order3.into_iter().take(higher_order_beam));
    all.extend(order4.into_iter().take(higher_order_beam));
    all.retain(|s| s.lift > 1.0 && s.support_pos > 0.0 && s.score.is_finite());
    all.retain(|s| interaction_family_ok(&s.indices, feature_cols));
    all.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranked = Vec::new();
    for (rank, stat) in all.into_iter().take(final_top).enumerate() {
        let stability = estimate_stability(&activation, &sample_labels, n_cols, &stat.indices);
        let score = stat.support_pos * stat.log_lift.max(0.0) * stability;
        ranked.push(InteractionFactorStat {
            rank: rank + 1,
            order: stat.indices.len(),
            factors: stat
                .indices
                .iter()
                .map(|&idx| feature_cols[idx].clone())
                .collect(),
            feature_indices: stat.indices.clone(),
            support_pos: stat.support_pos,
            support_neg: stat.support_neg,
            lift: stat.lift,
            log_lift: stat.log_lift,
            stability,
            score,
            direction: interaction_direction(&stat.indices, feature_cols, &engine.weights),
        });
    }
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, item) in ranked.iter_mut().enumerate() {
        item.rank = idx + 1;
    }

    write_json_pretty(&result_path, &ranked)?;
    write_interaction_csv(tables_dir.join("interaction_factor_ranking.csv"), &ranked)?;
    status!(
        "[interactions] complete: ranked={}, json={}, csv={}",
        ranked.len(),
        result_path.display(),
        tables_dir.join("interaction_factor_ranking.csv").display()
    );
    Ok(())
}

fn stratified_sample_indices(targets: &[f64], max_per_class: usize) -> Vec<usize> {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for (idx, &target) in targets.iter().enumerate() {
        if target > 0.5 {
            pos.push(idx);
        } else {
            neg.push(idx);
        }
    }
    let pos = deterministic_take(&pos, max_per_class);
    let neg = deterministic_take(&neg, max_per_class);
    pos.into_iter().chain(neg).collect()
}

fn deterministic_take(values: &[usize], max_count: usize) -> Vec<usize> {
    if values.len() <= max_count {
        return values.to_vec();
    }
    (0..max_count)
        .map(|i| values[i * values.len() / max_count])
        .collect()
}

fn build_risk_activation_matrix(
    feature_data: &[Vec<f64>],
    sample_indices: &[usize],
    means: &[f64],
    scales: &[f64],
    weights: &[f64],
    gamma: f64,
) -> Vec<f32> {
    let n_cols = means.len().min(scales.len()).min(weights.len());
    let mut activation = vec![0.0f32; sample_indices.len() * n_cols];
    activation
        .par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(sample_row, out)| {
            let source_row = sample_indices[sample_row];
            let row = &feature_data[source_row];
            for col_idx in 0..n_cols {
                let value = row.get(col_idx).copied().unwrap_or(f64::NAN);
                let scale = if scales[col_idx].is_finite() && scales[col_idx].abs() > 1e-12 {
                    scales[col_idx]
                } else {
                    1.0
                };
                let filled = if value.is_finite() {
                    value
                } else {
                    means[col_idx]
                };
                let z = ((filled - means[col_idx]) / scale).clamp(-10.0, 10.0);
                let direction = if weights[col_idx] < 0.0 { -1.0 } else { 1.0 };
                out[col_idx] = sigmoid_local(gamma * direction * z) as f32;
            }
        });
    activation
}

fn compute_order1_stats(activation: &[f32], labels: &[bool], n_cols: usize) -> Vec<CandidateStat> {
    let mut pos = vec![0.0f64; n_cols];
    let mut neg = vec![0.0f64; n_cols];
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    for (row_idx, &label) in labels.iter().enumerate() {
        let row = &activation[row_idx * n_cols..(row_idx + 1) * n_cols];
        if label {
            n_pos += 1;
            for col_idx in 0..n_cols {
                pos[col_idx] += row[col_idx] as f64;
            }
        } else {
            n_neg += 1;
            for col_idx in 0..n_cols {
                neg[col_idx] += row[col_idx] as f64;
            }
        }
    }
    (0..n_cols)
        .map(|idx| stat_from_sums(vec![idx], pos[idx], neg[idx], n_pos, n_neg, 0.0))
        .collect()
}

fn compute_order2_stats(activation: &[f32], labels: &[bool], n_cols: usize) -> Vec<CandidateStat> {
    let n_pairs = n_cols * (n_cols - 1) / 2;
    let mut pos = vec![0.0f64; n_pairs];
    let mut neg = vec![0.0f64; n_pairs];
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    for (row_idx, &label) in labels.iter().enumerate() {
        let row = &activation[row_idx * n_cols..(row_idx + 1) * n_cols];
        let sums = if label {
            n_pos += 1;
            &mut pos
        } else {
            n_neg += 1;
            &mut neg
        };
        let mut pair_idx = 0usize;
        for a in 0..n_cols {
            for b in (a + 1)..n_cols {
                sums[pair_idx] += row[a] as f64 * row[b] as f64;
                pair_idx += 1;
            }
        }
    }
    let mut stats = Vec::with_capacity(n_pairs);
    let mut pair_idx = 0usize;
    for a in 0..n_cols {
        for b in (a + 1)..n_cols {
            stats.push(stat_from_sums(
                vec![a, b],
                pos[pair_idx],
                neg[pair_idx],
                n_pos,
                n_neg,
                0.0,
            ));
            pair_idx += 1;
        }
    }
    stats.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    stats
}

fn build_order3_candidates(
    n_cols: usize,
    strong_pair_keys: &HashSet<Vec<usize>>,
    strong_pairs: &[CandidateStat],
    feature_cols: &[String],
) -> Vec<Vec<usize>> {
    let pair_score = score_lookup(strong_pairs);
    let mut candidates = Vec::new();
    for a in 0..n_cols {
        for b in (a + 1)..n_cols {
            for c in (b + 1)..n_cols {
                let pairs = [vec![a, b], vec![a, c], vec![b, c]];
                let indices = vec![a, b, c];
                if !interaction_family_ok(&indices, feature_cols) {
                    continue;
                }
                let strong_count = pairs
                    .iter()
                    .filter(|pair| strong_pair_keys.contains(*pair))
                    .count();
                if strong_count >= 2 {
                    let parent = pairs
                        .iter()
                        .map(|pair| pair_score.get(pair).copied().unwrap_or(0.0))
                        .sum::<f64>();
                    candidates.push((indices, parent));
                }
            }
        }
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates
        .into_iter()
        .take(600)
        .map(|(idx, _)| idx)
        .collect()
}

fn build_order4_candidates(
    n_cols: usize,
    strong_triple_keys: &HashSet<Vec<usize>>,
    strong_triples: &[CandidateStat],
    feature_cols: &[String],
) -> Vec<Vec<usize>> {
    let triple_score = score_lookup(strong_triples);
    let mut candidates = Vec::new();
    for a in 0..n_cols {
        for b in (a + 1)..n_cols {
            for c in (b + 1)..n_cols {
                for d in (c + 1)..n_cols {
                    let indices = vec![a, b, c, d];
                    if !interaction_family_ok(&indices, feature_cols) {
                        continue;
                    }
                    let triples = [vec![a, b, c], vec![a, b, d], vec![a, c, d], vec![b, c, d]];
                    let strong_count = triples
                        .iter()
                        .filter(|triple| strong_triple_keys.contains(*triple))
                        .count();
                    if strong_count >= 3 {
                        let parent = triples
                            .iter()
                            .map(|triple| triple_score.get(triple).copied().unwrap_or(0.0))
                            .sum::<f64>();
                        candidates.push((indices, parent));
                    }
                }
            }
        }
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates
        .into_iter()
        .take(600)
        .map(|(idx, _)| idx)
        .collect()
}

fn interaction_family_ok(indices: &[usize], feature_cols: &[String]) -> bool {
    if indices.len() <= 2 {
        return true;
    }
    let mut counts: std::collections::HashMap<&'static str, usize> =
        std::collections::HashMap::new();
    for &idx in indices {
        let family = feature_family(feature_cols.get(idx).map(String::as_str).unwrap_or(""));
        *counts.entry(family).or_insert(0) += 1;
    }
    let distinct = counts.len();
    let max_same_family = counts.values().copied().max().unwrap_or(0);
    distinct >= 3 && max_same_family <= 2
}

fn feature_family(name: &str) -> &'static str {
    if name.starts_with("temp_month_") || name.starts_with("temp_avg_prev_") {
        "temperature"
    } else if name.starts_with("rainfall_month_")
        || name.starts_with("rainfall_prev")
        || name.starts_with("rainfall_avg_prev_")
    {
        "rainfall"
    } else if matches!(name, "elevation" | "slope" | "aspect") {
        "terrain"
    } else if matches!(name, "dist_to_road" | "dist_to_railway") {
        "transport_distance"
    } else if matches!(name, "dist_to_river" | "dist_to_waterway") {
        "hydrology_distance"
    } else if name.starts_with("dist_to_urban_settlement")
        || name.starts_with("dist_to_rural_settlement")
        || name == "resident"
    {
        "settlement"
    } else if name == "landuse" {
        "landuse"
    } else {
        "other"
    }
}

fn compute_candidate_stats(
    activation: &[f32],
    labels: &[bool],
    n_cols: usize,
    candidates: &[Vec<usize>],
    workers: usize,
    label: &str,
) -> Vec<CandidateStat> {
    status!(
        "[interactions] compute {} stats: candidates={}, sampled_rows={}, workers={}",
        label,
        candidates.len(),
        labels.len(),
        workers
    );
    let n_pos = labels.iter().filter(|&&v| v).count();
    let n_neg = labels.len().saturating_sub(n_pos);
    let mut stats: Vec<_> = candidates
        .par_iter()
        .map(|indices| {
            let mut pos_sum = 0.0;
            let mut neg_sum = 0.0;
            for (row_idx, &is_pos) in labels.iter().enumerate() {
                let row = &activation[row_idx * n_cols..(row_idx + 1) * n_cols];
                let value = indices
                    .iter()
                    .fold(1.0f64, |acc, &idx| acc * row[idx] as f64);
                if is_pos {
                    pos_sum += value;
                } else {
                    neg_sum += value;
                }
            }
            stat_from_sums(indices.clone(), pos_sum, neg_sum, n_pos, n_neg, 0.0)
        })
        .collect();
    stats.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    stats
}

fn stat_from_sums(
    indices: Vec<usize>,
    pos_sum: f64,
    neg_sum: f64,
    n_pos: usize,
    n_neg: usize,
    parent_score: f64,
) -> CandidateStat {
    let eps = 1e-6;
    let support_pos = pos_sum / (n_pos as f64).max(1.0);
    let support_neg = neg_sum / (n_neg as f64).max(1.0);
    let lift = (support_pos + eps) / (support_neg + eps);
    let log_lift = lift.ln();
    let score = support_pos * log_lift.max(0.0);
    CandidateStat {
        indices,
        support_pos,
        support_neg,
        lift,
        log_lift,
        score,
        parent_score,
    }
}

fn estimate_stability(
    activation: &[f32],
    labels: &[bool],
    n_cols: usize,
    indices: &[usize],
) -> f64 {
    let folds = 5usize;
    let mut stable = 0usize;
    for fold in 0..folds {
        let mut pos_sum = 0.0;
        let mut neg_sum = 0.0;
        let mut n_pos = 0usize;
        let mut n_neg = 0usize;
        for (row_idx, &is_pos) in labels.iter().enumerate() {
            if row_idx % folds != fold {
                continue;
            }
            let row = &activation[row_idx * n_cols..(row_idx + 1) * n_cols];
            let value = indices
                .iter()
                .fold(1.0f64, |acc, &idx| acc * row[idx] as f64);
            if is_pos {
                n_pos += 1;
                pos_sum += value;
            } else {
                n_neg += 1;
                neg_sum += value;
            }
        }
        let stat = stat_from_sums(indices.to_vec(), pos_sum, neg_sum, n_pos, n_neg, 0.0);
        if stat.lift > 1.0 && stat.score > 0.0 {
            stable += 1;
        }
    }
    stable as f64 / folds as f64
}

fn score_lookup(stats: &[CandidateStat]) -> std::collections::HashMap<Vec<usize>, f64> {
    stats
        .iter()
        .map(|s| (s.indices.clone(), s.score.max(s.parent_score)))
        .collect()
}

fn candidate_indices(candidates: &[Vec<usize>]) -> Vec<Vec<usize>> {
    candidates.to_vec()
}

fn interaction_direction(indices: &[usize], feature_cols: &[String], weights: &[f64]) -> String {
    indices
        .iter()
        .map(|&idx| {
            let name = feature_cols
                .get(idx)
                .map(String::as_str)
                .unwrap_or("unknown");
            let side = if weights.get(idx).copied().unwrap_or(0.0) < 0.0 {
                if name.starts_with("dist_to_") {
                    "near"
                } else {
                    "low"
                }
            } else {
                "high"
            };
            format!("{name} {side}")
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn sigmoid_local(x: f64) -> f64 {
    if x >= 0.0 {
        let neg = (-x).exp();
        1.0 / (1.0 + neg)
    } else {
        let pos = x.exp();
        pos / (1.0 + pos)
    }
}

fn write_json_pretty(path: impl AsRef<Path>, value: &impl Serialize) -> Result<(), TrainError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn write_interaction_csv(
    path: impl AsRef<Path>,
    ranked: &[InteractionFactorStat],
) -> Result<(), TrainError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = String::from(
        "rank,order,factors,support_pos,support_neg,lift,log_lift,stability,score,direction\n",
    );
    for item in ranked {
        out.push_str(&format!(
            "{},{},\"{}\",{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},\"{}\"\n",
            item.rank,
            item.order,
            csv_escape(&item.factors.join(" * ")),
            item.support_pos,
            item.support_neg,
            item.lift,
            item.log_lift,
            item.stability,
            item.score,
            csv_escape(&item.direction)
        ));
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn csv_escape(value: &str) -> String {
    value.replace('"', "\"\"")
}

fn file_is_nonempty(path: &Path) -> bool {
    path.metadata().map(|m| m.len() > 0).unwrap_or(false)
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

#[cfg(test)]
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

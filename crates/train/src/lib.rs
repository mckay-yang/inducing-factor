//! Training stage boundary.
//!
//! Training consumes preprocessed feature tables and writes model artifacts. It
//! depends on `model` for the artifact contract, but prediction does not
//! depend on this crate.

use factor_core::discretize::AdaptiveDiscretizer;
use factor_core::model::{
    DisturbanceClassEngine, DisturbanceTypeDiagnostics, FeatureThreshold, InteractionSearchConfig,
    InteractionSearchStats, LearnedBin, NumericInteractionTerm, NumericRiskThreshold,
    OccurrenceProbabilityEngine, TypeClassDiagnostic,
};
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
    let learned_thresholds = learn_feature_thresholds(
        &feature_cols,
        &feature_data,
        &targets,
        &occ_engine.numeric_means,
        &occ_engine.weights,
    );
    occ_engine.activation_midpoints = activation_midpoints(
        &feature_cols,
        &occ_engine.weights,
        &occ_engine.numeric_means,
    );
    occ_engine.learned_thresholds = learned_thresholds.clone();
    write_threshold_outputs(out_dir, &learned_thresholds)?;
    let interaction_cache_dir = interaction_cache_dir_from_feature_path(feature_path);
    let interaction_terms = mine_interaction_factors(
        out_dir,
        &interaction_cache_dir,
        &feature_cols,
        &feature_data,
        &targets,
        &occ_engine,
        workers,
    )?;
    occ_engine.interaction_search_config =
        read_json_optional(&interaction_cache_dir.join("search_config.json"))?.unwrap_or_default();
    occ_engine.interaction_search_stats =
        read_json_optional(&interaction_cache_dir.join("search_stats.json"))?.unwrap_or_default();
    let model_interaction_top = std::env::var("INTERACTION_MODEL_TOP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1024);
    let model_interaction_terms =
        select_model_interaction_terms(interaction_terms, model_interaction_top);
    if model_interaction_terms.is_empty() {
        status!("[train] no continuous interaction terms selected for enhanced model");
    } else {
        status!(
            "[train] train enhanced numeric occurrence model with continuous interactions: rows={}, main_dimensions={}, interaction_terms={}",
            feature_data.len(),
            feature_cols.len(),
            model_interaction_terms.len()
        );
        occ_engine.fit_numeric_interactions_with_progress(
            &feature_data,
            &targets,
            &model_interaction_terms,
            1.5,
            &train_progress,
        );
        write_weighted_interaction_outputs(out_dir, &occ_engine.interactions)?;
    }
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
    let (class_engine, type_diagnostics) = if class_training.mask.is_empty() {
        status!("[train] skip numeric disturbance type model: no usable class labels");
        (None, build_type_diagnostics(&[], &[]))
    } else {
        status!(
            "[train] train numeric disturbance type model: rows={}, classes from disturbance_class, main_dimensions={}, interaction_terms={}",
            class_training.mask.len(),
            feature_cols.len(),
            occ_engine.interactions.len()
        );
        let mut engine = DisturbanceClassEngine::new();
        engine.fit_numeric_with_interactions(
            &feature_cols,
            &feature_data,
            &class_training.values,
            &class_training.mask,
            &occ_engine,
            &train_progress,
        );
        status!(
            "[train] numeric disturbance type model fitted: backend={}, classes={:?}",
            engine.backend,
            engine.classes
        );
        let diagnostics = build_type_diagnostics(&class_training.values, &engine.classes);
        write_json_pretty(
            out_dir.join("disturbance_type_diagnostics.json"),
            &diagnostics,
        )?;
        (Some(engine), diagnostics)
    };
    write_json_pretty(
        out_dir.join("disturbance_type_diagnostics.json"),
        &type_diagnostics,
    )?;
    for warning in &type_diagnostics.warnings {
        status!("[type-diagnostics] warning: {}", warning);
    }

    let model_backend = if occ_engine.interactions.is_empty() {
        "numeric_logistic_regression".to_string()
    } else {
        "numeric_logistic_regression_with_interactions".to_string()
    };
    let model_path = out_dir.join("basin_model.json");
    let basin_model = BasinModelArtifact {
        version: "0.1.0".into(),
        feature_columns: feature_cols.clone(),
        discretizer,
        selected_factors: selected_factors.clone(),
        occurrence_engine: occ_engine,
        class_engine,
        n_rows: total_rows,
        n_events,
        n_background,
        backend: model_backend,
        missing_data_policy: missing_policy_name(&settings.missing_data_policy).into(),
        random_state: settings.random_state,
        type_diagnostics: Some(type_diagnostics.clone()),
        prediction_feature_assumptions: prediction_feature_assumptions(),
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
        "backend": &basin_model.backend,
        "missing_data_policy": missing_policy_name(&settings.missing_data_policy),
        "random_state": settings.random_state,
        "interaction_terms": basin_model.occurrence_engine.interactions.len(),
        "class_backend": basin_model.class_engine.as_ref().map(|engine| engine.backend.clone()),
        "type_diagnostics": &type_diagnostics,
        "prediction_feature_assumptions": prediction_feature_assumptions(),
    });
    std::fs::write(
        out_dir.join("basin_model_metadata.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    write_json_pretty(
        out_dir.join("prediction_feature_assumptions.json"),
        &prediction_feature_assumptions(),
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
    status!(
        "[interactions] start from cached features={}, model={}",
        feature_path.display(),
        model_path.display()
    );
    let model = BasinModelArtifact::read_json(model_path)?;
    if !model.occurrence_engine.interactions.is_empty() {
        write_weighted_interaction_outputs(out_dir, &model.occurrence_engine.interactions)?;
        status!(
            "[interactions] model already contains weighted interaction terms: {}",
            model.occurrence_engine.interactions.len()
        );
        return Ok(());
    }
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

    let terms = mine_interaction_factors(
        out_dir,
        &interaction_cache_dir_from_feature_path(feature_path),
        &feature_cols,
        &feature_data,
        &targets,
        &model.occurrence_engine,
        effective,
    )?;
    if !model.occurrence_engine.interactions.is_empty() {
        write_weighted_interaction_outputs(out_dir, &model.occurrence_engine.interactions)?;
    } else if !terms.is_empty() {
        write_weighted_interaction_outputs(out_dir, &terms)?;
    }
    Ok(())
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
    values: Vec<f64>,
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
    let mut values = Vec::new();
    for row_idx in 0..n_rows {
        let flag = flags.as_ref().and_then(|s| s.get(row_idx)).unwrap_or(0);
        let class_value = classes.as_ref().and_then(|s| s.get(row_idx)).unwrap_or(0);
        if flag > 0 && (1..=4).contains(&class_value) {
            mask.push(row_idx);
            values.push(class_value as f64);
        }
    }

    ClassTargets { mask, values }
}

fn activation_midpoints(
    feature_cols: &[String],
    weights: &[f64],
    means: &[f64],
) -> Vec<NumericRiskThreshold> {
    (0..feature_cols.len().min(weights.len()).min(means.len()))
        .map(|idx| {
            let feature_name = feature_cols[idx].clone();
            let direction = if weights[idx] < 0.0 {
                if feature_name.starts_with("dist_to_") {
                    "near_or_lower"
                } else {
                    "lower_or_equal"
                }
            } else {
                "greater_or_equal"
            };
            NumericRiskThreshold {
                feature_index: idx,
                feature_name,
                direction: direction.into(),
                midpoint_value: means[idx],
                z_midpoint: 0.0,
                activation_midpoint: 0.5,
            }
        })
        .collect()
}

fn learn_feature_thresholds(
    feature_cols: &[String],
    feature_data: &[Vec<f64>],
    targets: &[f64],
    means: &[f64],
    weights: &[f64],
) -> Vec<FeatureThreshold> {
    let n_cols = feature_cols.len().min(means.len()).min(weights.len());
    if feature_data.is_empty() || n_cols == 0 || targets.is_empty() {
        return Vec::new();
    }
    let max_per_class = std::env::var("THRESHOLD_MAX_PER_CLASS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(250_000);
    let sample_indices = stratified_sample_indices(targets, max_per_class);
    status!(
        "[thresholds] learn supervised one-dimensional thresholds: rows={}, features={}",
        sample_indices.len(),
        n_cols
    );
    (0..n_cols)
        .into_par_iter()
        .map(|col_idx| {
            learn_one_feature_threshold(
                col_idx,
                &feature_cols[col_idx],
                feature_data,
                targets,
                &sample_indices,
                means[col_idx],
                weights[col_idx],
            )
        })
        .collect()
}

fn learn_one_feature_threshold(
    col_idx: usize,
    feature_name: &str,
    feature_data: &[Vec<f64>],
    targets: &[f64],
    sample_indices: &[usize],
    mean: f64,
    weight: f64,
) -> FeatureThreshold {
    let mut values = Vec::with_capacity(sample_indices.len());
    let mut labels = Vec::with_capacity(sample_indices.len());
    for &row_idx in sample_indices {
        let raw = feature_data
            .get(row_idx)
            .and_then(|row| row.get(col_idx))
            .copied()
            .unwrap_or(f64::NAN);
        let value = if raw.is_finite() { raw } else { mean };
        if value.is_finite() {
            values.push(value);
            labels.push(targets.get(row_idx).copied().unwrap_or(0.0) > 0.5);
        }
    }
    let n_pos = labels.iter().filter(|&&v| v).count();
    let n_neg = labels.len().saturating_sub(n_pos);
    let activation_direction = if weight < 0.0 {
        if feature_name.starts_with("dist_to_") {
            "near_or_lower"
        } else {
            "lower_or_equal"
        }
    } else {
        "greater_or_equal"
    };
    if values.len() < 20 || n_pos == 0 || n_neg == 0 {
        return FeatureThreshold {
            feature_name: feature_name.into(),
            direction: activation_direction.into(),
            activation_midpoint: mean,
            learned_threshold: None,
            learned_bins: Vec::new(),
            method: "quantile_threshold_search".into(),
            support_pos: 0.0,
            support_neg: 0.0,
            lift: 0.0,
            youden_j: 0.0,
            information_gain: 0.0,
            auc_1d: 0.5,
            confidence: "low".into(),
            warning: Some("insufficient finite labeled samples".into()),
            explanation_text: format!(
                "{} risk-side activation high, threshold low-confidence",
                feature_name
            ),
        };
    }

    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut candidates = Vec::new();
    for q in 1..100 {
        let idx = ((sorted.len() - 1) * q) / 100;
        let value = sorted[idx];
        if candidates
            .last()
            .map(|last: &f64| (*last - value).abs() > 1e-9)
            .unwrap_or(true)
        {
            candidates.push(value);
        }
    }

    let parent_gini = gini_impurity(n_pos, n_neg);
    let mut best: Option<ThresholdEval> = None;
    for threshold in candidates {
        for high_when_high in [true, false] {
            let eval = evaluate_threshold(
                &values,
                &labels,
                threshold,
                high_when_high,
                n_pos,
                n_neg,
                parent_gini,
            );
            if eval.support_pos < 0.02 || eval.lift <= 1.0 {
                continue;
            }
            if best
                .as_ref()
                .map(|current| eval.threshold_score > current.threshold_score)
                .unwrap_or(true)
            {
                best = Some(eval);
            }
        }
    }

    let bins = learned_bins(&values, &labels, 10);
    let auc = auc_1d(&values, &labels);
    let Some(best) = best else {
        return FeatureThreshold {
            feature_name: feature_name.into(),
            direction: activation_direction.into(),
            activation_midpoint: mean,
            learned_threshold: None,
            learned_bins: bins.clone(),
            method: "quantile_threshold_search".into(),
            support_pos: 0.0,
            support_neg: 0.0,
            lift: 0.0,
            youden_j: 0.0,
            information_gain: 0.0,
            auc_1d: auc,
            confidence: "low".into(),
            warning: Some("no threshold passed support/lift constraints".into()),
            explanation_text: best_bin_explanation(feature_name, &bins).unwrap_or_else(|| {
                format!("{feature_name} risk-side activation high, threshold low-confidence")
            }),
        };
    };
    let direction = if best.high_when_high {
        "high_risk_when_high"
    } else {
        "high_risk_when_low"
    };
    let confidence = if best.support_pos >= 0.05 && best.lift >= 1.5 && best.youden_j > 0.05 {
        "medium"
    } else {
        "low"
    };
    let explanation_text = if best.high_when_high {
        format!("{} >= {:.6}", feature_name, best.threshold)
    } else {
        format!("{} <= {:.6}", feature_name, best.threshold)
    };
    FeatureThreshold {
        feature_name: feature_name.into(),
        direction: direction.into(),
        activation_midpoint: mean,
        learned_threshold: Some(best.threshold),
        learned_bins: bins,
        method: "quantile_threshold_search".into(),
        support_pos: best.support_pos,
        support_neg: best.support_neg,
        lift: best.lift,
        youden_j: best.youden_j,
        information_gain: best.information_gain,
        auc_1d: auc,
        confidence: confidence.into(),
        warning: (confidence == "low").then(|| "threshold confidence is low".into()),
        explanation_text,
    }
}

#[derive(Debug, Clone)]
struct ThresholdEval {
    threshold: f64,
    high_when_high: bool,
    support_pos: f64,
    support_neg: f64,
    lift: f64,
    youden_j: f64,
    information_gain: f64,
    threshold_score: f64,
}

fn evaluate_threshold(
    values: &[f64],
    labels: &[bool],
    threshold: f64,
    high_when_high: bool,
    n_pos: usize,
    n_neg: usize,
    parent_gini: f64,
) -> ThresholdEval {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tn = 0usize;
    for (&value, &label) in values.iter().zip(labels.iter()) {
        let active = if high_when_high {
            value >= threshold
        } else {
            value <= threshold
        };
        match (active, label) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let support_pos = tp as f64 / n_pos.max(1) as f64;
    let support_neg = fp as f64 / n_neg.max(1) as f64;
    let lift = (support_pos + 1e-6) / (support_neg + 1e-6);
    let youden_j = support_pos - support_neg;
    let left_n = tp + fp;
    let right_n = fn_ + tn;
    let total = left_n + right_n;
    let weighted_gini = if total > 0 {
        (left_n as f64 / total as f64) * gini_impurity(tp, fp)
            + (right_n as f64 / total as f64) * gini_impurity(fn_, tn)
    } else {
        parent_gini
    };
    let information_gain = (parent_gini - weighted_gini).max(0.0);
    let threshold_score = if lift > 1.0 && youden_j > 0.0 {
        youden_j * lift.ln() * support_pos.sqrt()
    } else {
        0.0
    };
    ThresholdEval {
        threshold,
        high_when_high,
        support_pos,
        support_neg,
        lift,
        youden_j,
        information_gain,
        threshold_score,
    }
}

fn gini_impurity(pos: usize, neg: usize) -> f64 {
    let n = pos + neg;
    if n == 0 {
        return 0.0;
    }
    let p = pos as f64 / n as f64;
    let q = neg as f64 / n as f64;
    1.0 - p * p - q * q
}

fn learned_bins(values: &[f64], labels: &[bool], n_bins: usize) -> Vec<LearnedBin> {
    if values.is_empty() || n_bins == 0 {
        return Vec::new();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut edges = Vec::new();
    for b in 1..n_bins {
        edges.push(sorted[(sorted.len() - 1) * b / n_bins]);
    }
    edges.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let total_pos = labels.iter().filter(|&&v| v).count().max(1);
    let total_neg = labels.len().saturating_sub(total_pos).max(1);
    let mut counts = vec![(0usize, 0usize); edges.len() + 1];
    for (&value, &label) in values.iter().zip(labels.iter()) {
        let idx = edges.partition_point(|edge| value > *edge);
        if label {
            counts[idx].0 += 1;
        } else {
            counts[idx].1 += 1;
        }
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(idx, (count_pos, count_neg))| {
            let lower = if idx == 0 {
                None
            } else {
                edges.get(idx - 1).copied()
            };
            let upper = edges.get(idx).copied();
            let pos_rate = count_pos as f64 / total_pos as f64;
            let neg_rate = count_neg as f64 / total_neg as f64;
            LearnedBin {
                lower,
                upper,
                pos_rate,
                neg_rate,
                lift: (pos_rate + 1e-6) / (neg_rate + 1e-6),
                count_pos,
                count_neg,
            }
        })
        .collect()
}

fn best_bin_explanation(feature_name: &str, bins: &[LearnedBin]) -> Option<String> {
    let best = bins.iter().max_by(|a, b| {
        a.lift
            .partial_cmp(&b.lift)
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    match (best.lower, best.upper) {
        (Some(lower), Some(upper)) => Some(format!("{feature_name} in [{lower:.6}, {upper:.6}]")),
        (None, Some(upper)) => Some(format!("{feature_name} <= {upper:.6}")),
        (Some(lower), None) => Some(format!("{feature_name} >= {lower:.6}")),
        (None, None) => None,
    }
}

fn auc_1d(values: &[f64], labels: &[bool]) -> f64 {
    let mut pairs: Vec<(f64, bool)> = values.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_pos = pairs.iter().filter(|(_, label)| *label).count() as f64;
    let n_neg = pairs.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    let mut rank_sum = 0.0;
    for (idx, (_, label)) in pairs.iter().enumerate() {
        if *label {
            rank_sum += (idx + 1) as f64;
        }
    }
    let auc = (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);
    auc.max(1.0 - auc)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct InteractionSignature {
    algorithm_version: String,
    feature_names: Vec<String>,
    feature_means: Vec<f64>,
    feature_scales: Vec<f64>,
    baseline_weights: Vec<f64>,
    risk_directions: Vec<f64>,
    activation_gamma: f64,
    max_order: usize,
    min_support_pos: f64,
    min_lift: f64,
    top_k_pairs: usize,
    top_k_triples: usize,
    top_k_quads: usize,
    top_k_quints: usize,
    order2_bonus: f64,
    order3_bonus: f64,
    order4_bonus: f64,
    climate_influence_scale: f64,
    climate_only_interaction_penalty: f64,
    mixed_climate_interaction_penalty: f64,
    train_rows: usize,
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
    #[serde(default, rename = "model_weight", alias = "weight")]
    weight: f64,
    #[serde(default)]
    risk_thresholds: Vec<NumericRiskThreshold>,
    #[serde(default)]
    activation_midpoints: Vec<NumericRiskThreshold>,
    #[serde(default)]
    learned_thresholds: Vec<FeatureThreshold>,
    #[serde(default)]
    explanation_text: String,
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
}

fn mine_interaction_factors(
    out_dir: &Path,
    cache_dir: &Path,
    feature_cols: &[String],
    feature_data: &[Vec<f64>],
    targets: &[f64],
    engine: &OccurrenceProbabilityEngine,
    workers: usize,
) -> Result<Vec<NumericInteractionTerm>, TrainError> {
    let result_path = out_dir.join("interaction_factors.json");
    if feature_data.is_empty() || feature_cols.is_empty() || engine.weights.is_empty() {
        status!("[interactions] skipped: missing numeric baseline inputs");
        return Ok(Vec::new());
    }

    let tables_dir = tables_dir_from_model_dir(out_dir);
    std::fs::create_dir_all(&cache_dir)?;
    std::fs::create_dir_all(&tables_dir)?;

    let n_cols = feature_cols
        .len()
        .min(engine.weights.len())
        .min(engine.numeric_means.len())
        .min(engine.numeric_scales.len());
    let gamma = 1.5;
    let max_order = std::env::var("INTERACTION_MAX_ORDER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(2, 4))
        .unwrap_or(4);
    let climate_scale = std::env::var("CLIMATE_INFLUENCE_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.65);
    let climate_only_penalty = std::env::var("CLIMATE_ONLY_INTERACTION_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.35);
    let mixed_climate_penalty = std::env::var("MIXED_CLIMATE_INTERACTION_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.75);
    let min_support_pos = std::env::var("INTERACTION_MIN_SUPPORT_POS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.02);
    let min_lift = std::env::var("INTERACTION_MIN_LIFT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.2);
    let max_per_class = std::env::var("INTERACTION_MAX_PER_CLASS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(250_000);
    let pair_beam = std::env::var("INTERACTION_PAIR_BEAM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(2048);
    let triple_beam = std::env::var("INTERACTION_TRIPLE_BEAM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(4096);
    let quad_beam = std::env::var("INTERACTION_QUAD_BEAM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(2048);
    let final_top = std::env::var("INTERACTION_FINAL_TOP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1024);

    let sample_indices = stratified_sample_indices(targets, max_per_class);
    let sample_pos = sample_indices
        .iter()
        .filter(|&&idx| targets.get(idx).copied().unwrap_or(0.0) > 0.5)
        .count();
    let sample_neg = sample_indices.len().saturating_sub(sample_pos);
    if sample_pos == 0 || sample_neg == 0 {
        status!("[interactions] skipped: sampled labels lack positive/background rows");
        return Ok(Vec::new());
    }

    status!(
        "[interactions] sampled rows: total={}, positive={}, background={}, features={}, gamma={}, max_order={}, climate_scale={:.3}, climate_only_penalty={:.3}, mixed_climate_penalty={:.3}",
        sample_indices.len(),
        sample_pos,
        sample_neg,
        n_cols,
        gamma,
        max_order,
        climate_scale,
        climate_only_penalty,
        mixed_climate_penalty
    );

    let activation = build_risk_activation_matrix(
        &feature_cols[..n_cols],
        feature_data,
        &sample_indices,
        &engine.numeric_means[..n_cols],
        &engine.numeric_scales[..n_cols],
        &engine.weights[..n_cols],
        gamma,
        climate_scale,
    );
    let sample_labels: Vec<bool> = sample_indices
        .iter()
        .map(|&idx| targets.get(idx).copied().unwrap_or(0.0) > 0.5)
        .collect();

    let signature = InteractionSignature {
        algorithm_version: "apriori_beam_v3_climate_suppressed".into(),
        feature_names: feature_cols[..n_cols].to_vec(),
        feature_means: engine.numeric_means[..n_cols].to_vec(),
        feature_scales: engine.numeric_scales[..n_cols].to_vec(),
        baseline_weights: engine.weights[..n_cols].to_vec(),
        risk_directions: engine.weights[..n_cols]
            .iter()
            .map(|&w| if w < 0.0 { -1.0 } else { 1.0 })
            .collect(),
        activation_gamma: gamma,
        max_order,
        min_support_pos,
        min_lift,
        top_k_pairs: pair_beam,
        top_k_triples: triple_beam,
        top_k_quads: quad_beam,
        top_k_quints: 0,
        order2_bonus: 1.0,
        order3_bonus: 1.10,
        order4_bonus: 1.15,
        climate_influence_scale: climate_scale,
        climate_only_interaction_penalty: climate_only_penalty,
        mixed_climate_interaction_penalty: mixed_climate_penalty,
        train_rows: feature_data.len(),
        sample_rows: sample_indices.len(),
        positive_rows: sample_pos,
        background_rows: sample_neg,
    };
    let signature_path = cache_dir.join("feature_signature.json");
    if signature_matches(&signature_path, &signature) && file_is_nonempty(&result_path) {
        let raw = std::fs::read_to_string(&result_path)?;
        let ranked: Vec<InteractionFactorStat> = serde_json::from_str(&raw)?;
        status!(
            "[interactions] cache hit: ranked={}, json={}",
            ranked.len(),
            result_path.display()
        );
        return Ok(ranked_to_terms(&ranked));
    }
    write_json_pretty(&signature_path, &signature)?;

    let search_config = InteractionSearchConfig {
        algorithm: "apriori_beam".into(),
        max_order,
        min_support_pos,
        min_lift,
        pair_beam,
        triple_beam,
        quad_beam,
        final_top,
        order2_bonus: 1.0,
        order3_bonus: 1.10,
        order4_bonus: 1.15,
        climate_influence_scale: climate_scale,
        climate_only_interaction_penalty: climate_only_penalty,
        mixed_climate_interaction_penalty: mixed_climate_penalty,
    };
    status!("[interactions] compute order-1/order-2 continuous supports");
    let order1 = compute_order1_stats(&activation, &sample_labels, n_cols);
    let order2 = compute_order2_stats(&activation, &sample_labels, n_cols);
    write_json_pretty(cache_dir.join("order_1_stats.json"), &order1)?;
    write_json_pretty(cache_dir.join("order_2_stats.json"), &order2)?;

    let mut search_stats = InteractionSearchStats {
        order2_total: order2.len(),
        ..Default::default()
    };
    let strong_pairs = select_strong_candidates(
        &order2,
        &feature_cols[..n_cols],
        min_support_pos,
        min_lift,
        pair_beam,
        climate_only_penalty,
        mixed_climate_penalty,
    );
    search_stats.strong_pairs = strong_pairs.len();
    let strong_pair_set: HashSet<(usize, usize)> = strong_pairs
        .iter()
        .map(|s| pair_key(s.indices[0], s.indices[1]))
        .collect();
    status!(
        "[interactions] apriori beam: order2_total={}, strong_pairs={}, pair_beam={}",
        search_stats.order2_total,
        search_stats.strong_pairs,
        pair_beam
    );
    write_json_pretty(cache_dir.join("strong_order_2_stats.json"), &strong_pairs)?;

    let mut order3 = Vec::new();
    let mut strong_triples = Vec::new();
    let mut strong_triple_set: HashSet<(usize, usize, usize)> = HashSet::new();
    if max_order >= 3 {
        let order3_candidates =
            build_order3_candidates_from_pairs(n_cols, &strong_pairs, &strong_pair_set);
        search_stats.generated_triples_from_beam = order3_candidates.len();
        status!(
            "[interactions] apriori beam: generated_triples_from_beam={}, triple_beam={}",
            search_stats.generated_triples_from_beam,
            triple_beam
        );
        write_json_pretty(
            cache_dir.join("order_3_candidates.json"),
            &candidate_indices(&order3_candidates),
        )?;
        order3 = compute_candidate_stats(
            &activation,
            &sample_labels,
            n_cols,
            &order3_candidates,
            workers,
            "order-3",
        );
        write_json_pretty(cache_dir.join("order_3_stats.json"), &order3)?;
        strong_triples = select_strong_candidates(
            &order3,
            &feature_cols[..n_cols],
            min_support_pos,
            min_lift,
            triple_beam,
            climate_only_penalty,
            mixed_climate_penalty,
        );
        search_stats.strong_triples = strong_triples.len();
        strong_triple_set = strong_triples
            .iter()
            .filter_map(|s| triple_key_from_indices(&s.indices))
            .collect();
        write_json_pretty(cache_dir.join("strong_order_3_stats.json"), &strong_triples)?;
        status!(
            "[interactions] apriori beam: strong_triples={}",
            search_stats.strong_triples
        );
    }

    let mut order4 = Vec::new();
    if max_order >= 4 {
        let order4_candidates =
            build_order4_candidates_from_triples(n_cols, &strong_triples, &strong_triple_set);
        search_stats.generated_quads_from_beam = order4_candidates.len();
        status!(
            "[interactions] apriori beam: generated_quads_from_beam={}, quad_beam={}",
            search_stats.generated_quads_from_beam,
            quad_beam
        );
        write_json_pretty(
            cache_dir.join("order_4_candidates.json"),
            &candidate_indices(&order4_candidates),
        )?;
        order4 = compute_candidate_stats(
            &activation,
            &sample_labels,
            n_cols,
            &order4_candidates,
            workers,
            "order-4",
        );
        write_json_pretty(cache_dir.join("order_4_stats.json"), &order4)?;
        search_stats.strong_quads = select_strong_candidates(
            &order4,
            &feature_cols[..n_cols],
            min_support_pos,
            min_lift,
            quad_beam,
            climate_only_penalty,
            mixed_climate_penalty,
        )
        .len();
    }

    let mut all = Vec::new();
    all.extend(order1.into_iter().filter(|s| s.indices.len() >= 2));
    all.extend(order2);
    all.extend(order3);
    all.extend(order4);
    all.retain(|s| s.lift >= min_lift && s.support_pos >= min_support_pos && s.score.is_finite());
    all.sort_by(|a, b| {
        interaction_final_score(
            b,
            &feature_cols[..n_cols],
            climate_only_penalty,
            mixed_climate_penalty,
        )
            .partial_cmp(&interaction_final_score(
                a,
                &feature_cols[..n_cols],
                climate_only_penalty,
                mixed_climate_penalty,
            ))
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
            weight: 0.0,
            risk_thresholds: Vec::new(),
            activation_midpoints: interaction_thresholds(
                &stat.indices,
                feature_cols,
                &engine.weights,
                &engine.numeric_means,
            ),
            learned_thresholds: interaction_learned_thresholds(
                &stat.indices,
                &engine.learned_thresholds,
            ),
            explanation_text: interaction_explanation_text(
                &stat.indices,
                feature_cols,
                &engine.learned_thresholds,
                &engine.weights,
            ),
            direction: interaction_direction(&stat.indices, feature_cols, &engine.weights),
        });
    }
    ranked.sort_by(|a, b| {
        interaction_rank_score(
            b.score,
            b.order,
            climate_interaction_penalty(
                &b.factors,
                climate_only_penalty,
                mixed_climate_penalty,
            ),
        )
            .partial_cmp(&interaction_rank_score(
                a.score,
                a.order,
                climate_interaction_penalty(
                    &a.factors,
                    climate_only_penalty,
                    mixed_climate_penalty,
                ),
            ))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, item) in ranked.iter_mut().enumerate() {
        item.rank = idx + 1;
    }

    write_json_pretty(&result_path, &ranked)?;
    write_interaction_csv(tables_dir.join("interaction_factor_ranking.csv"), &ranked)?;
    search_stats.final_interactions = ranked.len();
    write_json_pretty(cache_dir.join("search_config.json"), &search_config)?;
    write_json_pretty(cache_dir.join("search_stats.json"), &search_stats)?;
    status!(
        "[interactions] complete: order2_total={}, strong_pairs={}, generated_triples_from_beam={}, generated_quads_from_beam={}, final_interactions={}, json={}, csv={}",
        search_stats.order2_total,
        search_stats.strong_pairs,
        search_stats.generated_triples_from_beam,
        search_stats.generated_quads_from_beam,
        ranked.len(),
        result_path.display(),
        tables_dir.join("interaction_factor_ranking.csv").display()
    );
    let mut terms = ranked_to_terms(&ranked);
    for term in &mut terms {
        term.explanation_text = interaction_explanation_text(
            &term.feature_indices,
            feature_cols,
            &engine.learned_thresholds,
            &engine.weights,
        );
    }
    Ok(terms)
}

fn select_model_interaction_terms(
    mut terms: Vec<NumericInteractionTerm>,
    top_k: usize,
) -> Vec<NumericInteractionTerm> {
    if top_k == 0 {
        return Vec::new();
    }
    terms.sort_by(|a, b| {
        interaction_rank_score(
            b.score,
            b.order,
            climate_interaction_penalty(
                &b.feature_names,
                climate_only_interaction_penalty(),
                mixed_climate_interaction_penalty(),
            ),
        )
            .partial_cmp(&interaction_rank_score(
                a.score,
                a.order,
                climate_interaction_penalty(
                    &a.feature_names,
                    climate_only_interaction_penalty(),
                    mixed_climate_interaction_penalty(),
                ),
            ))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    terms.into_iter().take(top_k).collect()
}

fn interaction_order_bonus(order: usize) -> f64 {
    match order {
        4 => 1.15,
        3 => 1.10,
        2 => 1.00,
        _ => 0.0,
    }
}

fn interaction_rank_score(score: f64, order: usize, climate_penalty: f64) -> f64 {
    score * interaction_order_bonus(order) * climate_penalty
}

fn interaction_final_score(
    stat: &CandidateStat,
    feature_cols: &[String],
    climate_only_penalty: f64,
    mixed_climate_penalty: f64,
) -> f64 {
    interaction_rank_score(
        stat.score,
        stat.indices.len(),
        climate_interaction_penalty_for_indices(
            &stat.indices,
            feature_cols,
            climate_only_penalty,
            mixed_climate_penalty,
        ),
    )
}

fn select_strong_candidates(
    candidates: &[CandidateStat],
    feature_cols: &[String],
    min_support_pos: f64,
    min_lift: f64,
    beam: usize,
    climate_only_penalty: f64,
    mixed_climate_penalty: f64,
) -> Vec<CandidateStat> {
    let mut strong: Vec<_> = candidates
        .iter()
        .filter(|stat| {
            stat.support_pos >= min_support_pos
                && stat.lift >= min_lift
                && stat.score > 0.0
                && stat.score.is_finite()
        })
        .cloned()
        .collect();
    strong.sort_by(|a, b| {
        interaction_final_score(b, feature_cols, climate_only_penalty, mixed_climate_penalty)
            .partial_cmp(&interaction_final_score(
                a,
                feature_cols,
                climate_only_penalty,
                mixed_climate_penalty,
            ))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    strong.truncate(beam);
    strong
}

fn climate_only_interaction_penalty() -> f64 {
    std::env::var("CLIMATE_ONLY_INTERACTION_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.35)
}

fn mixed_climate_interaction_penalty() -> f64 {
    std::env::var("MIXED_CLIMATE_INTERACTION_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.75)
}

fn climate_interaction_penalty(
    feature_names: &[String],
    climate_only_penalty: f64,
    mixed_climate_penalty: f64,
) -> f64 {
    if feature_names.is_empty() {
        return 1.0;
    }
    let climate_count = feature_names
        .iter()
        .filter(|name| is_climate_feature_name(name))
        .count();
    if climate_count == 0 {
        1.0
    } else if climate_count == feature_names.len() {
        climate_only_penalty
    } else {
        let climate_share = climate_count as f64 / feature_names.len() as f64;
        mixed_climate_penalty.powf(climate_share)
    }
}

fn climate_interaction_penalty_for_indices(
    indices: &[usize],
    feature_cols: &[String],
    climate_only_penalty: f64,
    mixed_climate_penalty: f64,
) -> f64 {
    let names: Vec<String> = indices
        .iter()
        .filter_map(|&idx| feature_cols.get(idx).cloned())
        .collect();
    climate_interaction_penalty(&names, climate_only_penalty, mixed_climate_penalty)
}

fn is_climate_feature_name(name: &str) -> bool {
    name.starts_with("rainfall_") || name.starts_with("temp_")
}

fn pair_key(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn triple_key(a: usize, b: usize, c: usize) -> (usize, usize, usize) {
    let mut values = [a, b, c];
    values.sort_unstable();
    (values[0], values[1], values[2])
}

fn triple_key_from_indices(indices: &[usize]) -> Option<(usize, usize, usize)> {
    if indices.len() == 3 {
        Some(triple_key(indices[0], indices[1], indices[2]))
    } else {
        None
    }
}

fn build_order3_candidates_from_pairs(
    n_cols: usize,
    strong_pairs: &[CandidateStat],
    strong_pair_set: &HashSet<(usize, usize)>,
) -> Vec<Vec<usize>> {
    let mut out: HashSet<(usize, usize, usize)> = HashSet::new();
    for pair in strong_pairs {
        if pair.indices.len() != 2 {
            continue;
        }
        let a = pair.indices[0];
        let b = pair.indices[1];
        for c in 0..n_cols {
            if c == a || c == b {
                continue;
            }
            let pair_votes = [
                strong_pair_set.contains(&pair_key(a, b)),
                strong_pair_set.contains(&pair_key(a, c)),
                strong_pair_set.contains(&pair_key(b, c)),
            ]
            .iter()
            .filter(|&&v| v)
            .count();
            if pair_votes >= 2 {
                out.insert(triple_key(a, b, c));
            }
        }
    }
    let mut candidates: Vec<_> = out.into_iter().map(|(a, b, c)| vec![a, b, c]).collect();
    candidates.sort();
    candidates
}

fn build_order4_candidates_from_triples(
    n_cols: usize,
    strong_triples: &[CandidateStat],
    strong_triple_set: &HashSet<(usize, usize, usize)>,
) -> Vec<Vec<usize>> {
    let mut out: HashSet<(usize, usize, usize, usize)> = HashSet::new();
    for triple in strong_triples {
        if triple.indices.len() != 3 {
            continue;
        }
        let a = triple.indices[0];
        let b = triple.indices[1];
        let c = triple.indices[2];
        for d in 0..n_cols {
            if d == a || d == b || d == c {
                continue;
            }
            let votes = [
                strong_triple_set.contains(&triple_key(a, b, c)),
                strong_triple_set.contains(&triple_key(a, b, d)),
                strong_triple_set.contains(&triple_key(a, c, d)),
                strong_triple_set.contains(&triple_key(b, c, d)),
            ]
            .iter()
            .filter(|&&v| v)
            .count();
            if votes >= 3 {
                let mut values = [a, b, c, d];
                values.sort_unstable();
                out.insert((values[0], values[1], values[2], values[3]));
            }
        }
    }
    let mut candidates: Vec<_> = out
        .into_iter()
        .map(|(a, b, c, d)| vec![a, b, c, d])
        .collect();
    candidates.sort();
    candidates
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
    feature_cols: &[String],
    feature_data: &[Vec<f64>],
    sample_indices: &[usize],
    means: &[f64],
    scales: &[f64],
    weights: &[f64],
    gamma: f64,
    climate_scale: f64,
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
                let mut z = ((filled - means[col_idx]) / scale).clamp(-10.0, 10.0);
                if feature_cols
                    .get(col_idx)
                    .map(|name| is_climate_feature_name(name))
                    .unwrap_or(false)
                {
                    z *= climate_scale;
                }
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
        .map(|idx| stat_from_sums(vec![idx], pos[idx], neg[idx], n_pos, n_neg))
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
            stat_from_sums(indices.clone(), pos_sum, neg_sum, n_pos, n_neg)
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
        let stat = stat_from_sums(indices.to_vec(), pos_sum, neg_sum, n_pos, n_neg);
        if stat.lift > 1.0 && stat.score > 0.0 {
            stable += 1;
        }
    }
    stable as f64 / folds as f64
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

fn interaction_thresholds(
    indices: &[usize],
    feature_cols: &[String],
    weights: &[f64],
    means: &[f64],
) -> Vec<NumericRiskThreshold> {
    indices
        .iter()
        .map(|&idx| {
            let feature_name = feature_cols
                .get(idx)
                .cloned()
                .unwrap_or_else(|| "unknown".into());
            let direction = if weights.get(idx).copied().unwrap_or(0.0) < 0.0 {
                if feature_name.starts_with("dist_to_") {
                    "near_or_lower"
                } else {
                    "lower_or_equal"
                }
            } else {
                "greater_or_equal"
            };
            NumericRiskThreshold {
                feature_index: idx,
                feature_name,
                direction: direction.into(),
                midpoint_value: means.get(idx).copied().unwrap_or(0.0),
                z_midpoint: 0.0,
                activation_midpoint: 0.5,
            }
        })
        .collect()
}

fn interaction_learned_thresholds(
    indices: &[usize],
    learned_thresholds: &[FeatureThreshold],
) -> Vec<FeatureThreshold> {
    indices
        .iter()
        .filter_map(|&idx| learned_thresholds.get(idx).cloned())
        .collect()
}

fn interaction_explanation_text(
    indices: &[usize],
    feature_cols: &[String],
    learned_thresholds: &[FeatureThreshold],
    weights: &[f64],
) -> String {
    indices
        .iter()
        .map(|&idx| {
            if let Some(threshold) = learned_thresholds.get(idx) {
                if !threshold.explanation_text.is_empty() {
                    return threshold.explanation_text.clone();
                }
                if let Some(text) =
                    best_bin_explanation(&threshold.feature_name, &threshold.learned_bins)
                {
                    return text;
                }
            }
            let name = feature_cols
                .get(idx)
                .map(String::as_str)
                .unwrap_or("unknown");
            let side = if weights.get(idx).copied().unwrap_or(0.0) < 0.0 {
                "risk-side activation high, threshold low-confidence"
            } else {
                "risk-side activation high, threshold low-confidence"
            };
            format!("{name} {side}")
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn write_threshold_outputs(
    out_dir: &Path,
    thresholds: &[FeatureThreshold],
) -> Result<(), TrainError> {
    write_json_pretty(out_dir.join("learned_thresholds.json"), &thresholds)?;
    let path = tables_dir_from_model_dir(out_dir).join("learned_thresholds.csv");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = String::from(
        "feature_name,direction,activation_midpoint,learned_threshold,confidence,support_pos,support_neg,lift,youden_j,information_gain,auc_1d,warning,explanation_text\n",
    );
    for item in thresholds {
        out.push_str(&format!(
            "\"{}\",\"{}\",{:.8},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},\"{}\",\"{}\"\n",
            csv_escape(&item.feature_name),
            csv_escape(&item.direction),
            item.activation_midpoint,
            item.learned_threshold
                .map(|v| format!("{v:.8}"))
                .unwrap_or_else(|| "".into()),
            item.confidence,
            item.support_pos,
            item.support_neg,
            item.lift,
            item.youden_j,
            item.information_gain,
            item.auc_1d,
            csv_escape(item.warning.as_deref().unwrap_or("")),
            csv_escape(&item.explanation_text)
        ));
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn build_type_diagnostics(class_values: &[f64], classes: &[i64]) -> DisturbanceTypeDiagnostics {
    let total = class_values.len();
    let mut observed: Vec<i64> = if classes.is_empty() {
        vec![1, 2, 3, 4]
    } else {
        classes.to_vec()
    };
    observed.sort_unstable();
    observed.dedup();
    let mut class_items = Vec::new();
    let mut warnings = Vec::new();
    for class_value in observed {
        let positive_count = class_values
            .iter()
            .filter(|&&value| value as i64 == class_value)
            .count();
        let negative_count = total.saturating_sub(positive_count);
        let mut class_warnings = Vec::new();
        if positive_count < 100 {
            class_warnings.push("very_low_sample_count_lt_100".into());
        } else if positive_count < 500 {
            class_warnings.push("low_sample_count_lt_500".into());
        }
        let positive_rate = positive_count as f64 / total.max(1) as f64;
        if positive_rate < 0.02 || positive_rate > 0.98 {
            class_warnings.push("extreme_class_imbalance".into());
        }
        for warning in &class_warnings {
            warnings.push(format!("class {class_value}: {warning}"));
        }
        class_items.push(TypeClassDiagnostic {
            class_value,
            label: disturbance_type_label(class_value).into(),
            positive_count,
            negative_count,
            positive_rate,
            warnings: class_warnings,
        });
    }
    if total == 0 {
        warnings.push("no usable disturbance_class labels".into());
    }
    DisturbanceTypeDiagnostics {
        total_labeled: total,
        classes: class_items,
        warnings,
        probability_semantics:
            "type_probability is joint P(type=k | x) = P(disturbance | x) * P(type=k | disturbance, x)"
                .into(),
    }
}

fn disturbance_type_label(class_value: i64) -> &'static str {
    match class_value {
        1 => "fire",
        2 => "logging",
        3 => "geo_hazard",
        4 => "road_construction",
        _ => "unknown",
    }
}

fn prediction_feature_assumptions() -> serde_json::Value {
    serde_json::json!({
        "prediction_climate_windows": "prediction climate windows use default reference month = June when observed disturbance timing is unavailable",
        "data_fallback": "missing target-year climate or source rasters may fallback to earlier available years according to feature extraction lookup rules",
        "timing_semantics": "these are prediction-time assumptions, not observed disturbance timing",
        "future_external_inputs": "external forecasted variables can be injected by regenerating prediction feature parquet files",
    })
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

fn signature_matches(path: &Path, current: &InteractionSignature) -> bool {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(existing) = serde_json::from_str::<InteractionSignature>(&raw) else {
        return false;
    };
    existing == *current
}

fn ranked_to_terms(ranked: &[InteractionFactorStat]) -> Vec<NumericInteractionTerm> {
    ranked
        .iter()
        .map(|item| NumericInteractionTerm {
            id: item.factors.join("*"),
            order: item.order,
            feature_indices: item.feature_indices.clone(),
            feature_names: item.factors.clone(),
            weight: item.weight,
            support_pos: item.support_pos,
            support_neg: item.support_neg,
            lift: item.lift,
            log_lift: item.log_lift,
            stability: item.stability,
            score: item.score,
            risk_thresholds: item.risk_thresholds.clone(),
            activation_midpoints: item.activation_midpoints.clone(),
            learned_thresholds: item.learned_thresholds.clone(),
            explanation_text: item.explanation_text.clone(),
        })
        .collect()
}

fn terms_to_ranked(terms: &[NumericInteractionTerm]) -> Vec<InteractionFactorStat> {
    let mut ranked: Vec<_> = terms
        .iter()
        .enumerate()
        .map(|(idx, term)| InteractionFactorStat {
            rank: idx + 1,
            order: term.order,
            factors: term.feature_names.clone(),
            feature_indices: term.feature_indices.clone(),
            support_pos: term.support_pos,
            support_neg: term.support_neg,
            lift: term.lift,
            log_lift: term.log_lift,
            stability: term.stability,
            score: term.score,
            weight: term.weight,
            risk_thresholds: term.risk_thresholds.clone(),
            activation_midpoints: term.activation_midpoints.clone(),
            learned_thresholds: term.learned_thresholds.clone(),
            explanation_text: term.explanation_text.clone(),
            direction: if term.explanation_text.is_empty() {
                term.feature_names.join(" + ")
            } else {
                term.explanation_text.clone()
            },
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, item) in ranked.iter_mut().enumerate() {
        item.rank = idx + 1;
    }
    ranked
}

fn write_weighted_interaction_outputs(
    out_dir: &Path,
    terms: &[NumericInteractionTerm],
) -> Result<(), TrainError> {
    let ranked = terms_to_ranked(terms);
    write_json_pretty(out_dir.join("interaction_factors.json"), &ranked)?;
    write_interaction_csv(
        tables_dir_from_model_dir(out_dir).join("interaction_factor_ranking.csv"),
        &ranked,
    )?;
    Ok(())
}

fn interaction_cache_dir_from_feature_path(feature_path: &Path) -> PathBuf {
    feature_path
        .parent()
        .and_then(Path::parent)
        .map(|cache_root| cache_root.join("interactions"))
        .unwrap_or_else(|| PathBuf::from("data/cache/interactions"))
}

fn tables_dir_from_model_dir(model_dir: &Path) -> PathBuf {
    model_dir
        .parent()
        .map(|result_root| result_root.join("tables"))
        .unwrap_or_else(|| PathBuf::from("data/result/tables"))
}

fn write_json_pretty(path: impl AsRef<Path>, value: &impl Serialize) -> Result<(), TrainError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn read_json_optional<T: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
) -> Result<Option<T>, TrainError> {
    let path = path.as_ref();
    if !file_is_nonempty(path) {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path)?;
    Ok(Some(serde_json::from_str(&raw)?))
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
        "rank,order,factors,support_pos,support_neg,lift,log_lift,stability,score,model_weight,direction,explanation_text,learned_thresholds\n",
    );
    for item in ranked {
        let thresholds = item
            .learned_thresholds
            .iter()
            .map(|t| {
                if t.explanation_text.is_empty() {
                    format!(
                        "{} {} {:?}",
                        t.feature_name, t.direction, t.learned_threshold
                    )
                } else {
                    t.explanation_text.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        out.push_str(&format!(
            "{},{},\"{}\",{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},\"{}\",\"{}\",\"{}\"\n",
            item.rank,
            item.order,
            csv_escape(&item.factors.join(" * ")),
            item.support_pos,
            item.support_neg,
            item.lift,
            item.log_lift,
            item.stability,
            item.score,
            item.weight,
            csv_escape(&item.direction),
            csv_escape(&item.explanation_text),
            csv_escape(&thresholds)
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

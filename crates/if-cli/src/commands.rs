use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use tracing::{error, info, warn};

use if_config::AppConfig;
use if_core::discretize::AdaptiveDiscretizer;
use if_core::factors::FactorMiner;
use if_core::model::{DisturbanceClassEngine, OccurrenceProbabilityEngine};
use if_core::types::{DiscretizerConfig, FactorMiningConfig, InducingFactor};
use if_pipeline::{
    build_chunk_dataframe, chunk_cache_path, derive_chunk_grid, resolve_num_workers,
    write_audit_report, write_chunk_cache, AnnualProbabilityMapper, ChunkAudit,
    run_production_pipeline,
};

fn load_config(path: &Path) -> Option<AppConfig> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) => {
            error!("Failed to read config file {:?}: {}", path, e);
            return None;
        }
    };
    match serde_yaml::from_str(&raw) {
        Ok(config) => Some(config),
        Err(e) => {
            error!("Failed to parse config file {:?}: {}", path, e);
            None
        }
    }
}

fn resolve_worker_count(config: &AppConfig, cli_workers: Option<usize>) -> usize {
    if let Some(n) = cli_workers {
        n
    } else {
        resolve_num_workers(&config.pipeline)
    }
}

fn setup_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "if_cli=info".into()),
        )
        .with_target(false)
        .try_init()
        .ok();
}

#[allow(clippy::type_complexity)]
fn generate_synthetic_points(
    start_year: i32,
    end_year: i32,
    num_points: usize,
    seed: u64,
) -> (
    Vec<(i64, f64, f64, i32)>,
    HashMap<String, f64>,
    HashMap<i64, bool>,
    HashMap<i64, i32>,
    HashMap<i64, f64>,
    HashMap<i64, Vec<f64>>,
) {
    use rand::prelude::*;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(seed);
    let num_years = (end_year - start_year + 1).max(1) as usize;
    let total = num_points * num_years;

    let mut points: Vec<(i64, f64, f64, i32)> = Vec::with_capacity(total);
    let mut features: HashMap<String, f64> = HashMap::new();
    features.insert("slope".into(), 15.0);
    features.insert("aspect".into(), 180.0);
    features.insert("dist_to_river".into(), 5000.0);
    features.insert("dist_to_road".into(), 3000.0);
    features.insert("dist_to_railway".into(), 10000.0);
    features.insert("dist_to_waterway".into(), 8000.0);
    features.insert("temperature".into(), 15.0);
    features.insert("resident".into(), 100.0);
    features.insert("landuse".into(), 2.0);

    let mut dist_flags: HashMap<i64, bool> = HashMap::new();
    let mut dist_classes: HashMap<i64, i32> = HashMap::new();
    let mut dist_months: HashMap<i64, f64> = HashMap::new();
    let mut rainfall: HashMap<i64, Vec<f64>> = HashMap::new();

    for pid in 0..num_points as i64 {
        let x = rng.gen_range(0.0..255.0);
        let y = rng.gen_range(0.0..255.0);
        let is_disturbed = rng.gen_bool(0.3);
        if is_disturbed {
            dist_flags.insert(pid, true);
            dist_classes.insert(pid, rng.gen_range(1..=4));
            dist_months.insert(pid, rng.gen_range(1.0..=12.0));
        }
        let mut monthly: Vec<f64> = Vec::with_capacity(12);
        for _ in 0..12 {
            monthly.push(rng.gen_range(0.0..300.0));
        }
        rainfall.insert(pid, monthly);

        for yr in start_year..=end_year {
            points.push((pid, x, y, yr));
        }
    }

    (
        points,
        features,
        dist_flags,
        dist_classes,
        dist_months,
        rainfall,
    )
}

fn stage_header(stage: &str, phase: usize, total: usize, description: &str) {
    info!("[{}][phase {}/{}] {}", stage, phase, total, description);
}

#[allow(clippy::type_complexity)]
pub fn run_pipeline(config_path: &Path, workers: Option<usize>, dry_run: bool) {
    setup_logging();
    let start_time = Instant::now();
    let Some(config) = load_config(config_path) else {
        return;
    };

    // Route to production pipeline if real DEM raster is configured
    if config.data_sources.dem_raster.is_some() {
        eprintln!("[pipeline] Routing to production pipeline (DEM raster configured)");
        if dry_run {
            info!("[pipeline] Production mode dry-run for {:?}", config_path);
            return;
        }
        run_production_pipeline(&config, workers);
        return;
    }

    let num_workers = resolve_worker_count(&config, workers);
    info!("[pipeline][phase 0/3] Workers: {}", num_workers);

    let tile_size = config.chunks.tile_size;
    let tasks = derive_chunk_grid(256, 256, tile_size);
    let total_chunks = tasks.len();

    info!(
        "[pipeline][phase 1/3] Chunk plan: {} chunks (tile={}px)",
        total_chunks, tile_size
    );
    info!(
        "[pipeline] num_workers={}, years={}..={}",
        num_workers, config.pipeline.start_year, config.pipeline.end_year
    );

    if dry_run {
        for (i, task) in tasks.iter().enumerate() {
            info!(
                "[pipeline]   chunk[{}]: @({},{}) {}x{}px",
                i, task.row_off, task.col_off, task.width, task.height
            );
        }
        let elapsed = start_time.elapsed();
        info!(
            "[pipeline] Dry-run validated {:?} -- chunk plan OK, no processing. elapsed={:.2?}",
            config_path, elapsed
        );
        return;
    }

    if let Err(e) = std::fs::create_dir_all(&config.paths.cache_dir) {
        error!(
            "Failed to create cache dir {:?}: {}",
            config.paths.cache_dir, e
        );
        return;
    }

    let pb = ProgressBar::new(total_chunks as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) eta={eta}",
        )
        .unwrap(),
    );

    let (points, features, dist_flags, dist_classes, dist_months, rainfall) =
        generate_synthetic_points(
            config.pipeline.start_year,
            config.pipeline.end_year,
            100,
            config.attribution.random_state,
        );

    let mut audits: Vec<ChunkAudit> = Vec::new();

    for (i, task) in tasks.iter().enumerate() {
        match build_chunk_dataframe(
            task,
            &points,
            &features,
            &dist_flags,
            &dist_classes,
            &dist_months,
            &rainfall,
            config.attribution.random_state + i as u64,
        ) {
            Ok(mut df) => {
                let cache_path = chunk_cache_path(&config.paths.cache_dir, task.chunk_idx);
                if let Err(e) = write_chunk_cache(&mut df, &cache_path) {
                    error!("Failed to write chunk {}: {}", i, e);
                    audits.push(ChunkAudit::empty(i, "error"));
                } else {
                    let n = df.height();
                    let d_count = df
                        .column("disturbance_flag")
                        .ok()
                        .and_then(|c| c.i32().ok())
                        .map(|s| s.into_iter().filter(|v| v.is_some_and(|x| x == 1)).count())
                        .unwrap_or(0);
                    audits.push(ChunkAudit {
                        chunk_idx: i,
                        row_off: task.row_off,
                        col_off: task.col_off,
                        width: task.width,
                        height: task.height,
                        status: "ok".into(),
                        point_count: n,
                        disturbance_count: d_count,
                        background_count: n - d_count,
                        valid_pixels: n,
                        years: (config.pipeline.start_year..=config.pipeline.end_year).collect(),
                        worker_id: None,
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        cache_file: Some(cache_path.to_string_lossy().into()),
                    });
                }
            }
            Err(e) => {
                error!("Failed to build chunk {}: {}", i, e);
                audits.push(ChunkAudit::empty(i, "build_error"));
            }
        }
        pb.inc(1);
    }

    pb.finish_with_message("Pipeline complete");

    if let Err(e) = write_audit_report(&audits, &config.paths.cache_dir) {
        error!("Failed to write audit report: {}", e);
    }

    let elapsed = start_time.elapsed();
    let ok_count = audits.iter().filter(|a| a.status == "ok").count();
    let total_pts: usize = audits.iter().map(|a| a.point_count).sum();
    let total_dist: usize = audits.iter().map(|a| a.disturbance_count).sum();
    let pct = total_chunks.checked_mul(100).map_or(100, |p| p / ok_count);

    info!(
        "[pipeline] Pipeline complete {} / {} ({}%) elapsed={:.2?} eta=0s",
        ok_count, total_chunks, pct, elapsed
    );
    info!(
        "[pipeline] Total points: {}, disturbances: {}, ok chunks: {}/{}",
        total_pts, total_dist, ok_count, total_chunks
    );
}

/// Train a single basin-wide model from CLASS-derived training features.
///
/// Loads `.cache/features/training_features.parquet` (produced by T10 feature extraction),
/// auto-detects feature columns, fits a discretizer on all data, mines factor patterns
/// from event vs background populations, trains an occurrence probability engine,
/// and exports all artifacts to `results/models/`.
fn run_train_basin(config: &AppConfig, output_dir: Option<&Path>, feature_path: &Path) {
    setup_logging();
    let start_time = Instant::now();

    let out_dir = output_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| config.paths.output_dir.clone());

    // ── Phase 1: Load basin training features ──────────────────────
    stage_header("train", 1, 4, "Loading basin training features");

    let file = match std::fs::File::open(feature_path) {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to open {}: {}", feature_path.display(), e);
            return;
        }
    };
    let reader = ParquetReader::new(file);
    let df = match reader.finish() {
        Ok(df) => df,
        Err(e) => {
            error!("Failed to read parquet: {}", e);
            return;
        }
    };

    let total_rows = df.height();
    eprintln!(
        "[train] Basin features: {} rows, {} columns",
        total_rows,
        df.width()
    );

    // ── Auto-detect feature columns ───────────────────────────────
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
    eprintln!(
        "[train] Feature columns ({}): {:?}",
        feature_cols.len(),
        feature_cols
    );

    // ── Extract feature vectors and targets ───────────────────────
    let h = df.height();
    let mut feature_data: Vec<Vec<f64>> = Vec::with_capacity(h);
    let mut targets: Vec<f64> = Vec::with_capacity(h);

    for row_idx in 0..h {
        let mut row: Vec<f64> = Vec::with_capacity(feature_cols.len());
        for name in &feature_cols {
            let val = df
                .column(name)
                .ok()
                .and_then(|c| c.f64().ok())
                .and_then(|s| s.get(row_idx))
                .unwrap_or(f64::NAN);
            row.push(val);
        }
        let target = df
            .column("disturbance_flag")
            .ok()
            .and_then(|c| c.i32().ok())
            .and_then(|s| s.get(row_idx))
            .map(|v| v as f64)
            .unwrap_or(0.0);
        targets.push(target);
        feature_data.push(row);
    }

    let n_events = targets.iter().filter(|&&v| v > 0.5).count();
    let n_bg = total_rows - n_events;
    eprintln!("[train] Events: {}, Background: {}", n_events, n_bg);

    // ── Phase 2: Fit discretizer on ALL data ──────────────────────
    stage_header("train", 2, 4, "Fitting discretizer");

    let disc_config = DiscretizerConfig {
        method: config.attribution.discretizer_method.clone(),
        n_bins: config.attribution.discretize_bins,
        ..Default::default()
    };
    let mut discretizer = AdaptiveDiscretizer::new(disc_config);

    if !feature_data.is_empty() {
        discretizer.fit(&feature_cols, &feature_data, &targets);
        info!(
            "[train] Discretizer fit on {} samples",
            feature_data.len()
        );
    }

    // ── Discretize all data ───────────────────────────────────────
    let all_labels = discretizer.transform(&feature_cols, &feature_data, true);

    // ── Phase 3: Mine inducing factors ────────────────────────────
    stage_header("train", 3, 4, "Mining inducing factors");

    let mining_config = FactorMiningConfig {
        max_pattern_size: config.attribution.max_pattern_size,
        min_support: config.attribution.min_support,
        min_relative_risk: config.attribution.min_relative_risk,
        min_stability: config.attribution.min_stability,
        bootstrap_iterations: config.attribution.bootstrap_iterations,
        max_bootstrap_samples: config.attribution.max_bootstrap_samples,
        random_state: config.attribution.random_state,
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

    let n_event_i64 = event_transactions.len() as i64;
    let n_bg_i64 = bg_transactions.len() as i64;

    let pattern_stats = miner.mine_pattern_candidates(&event_transactions, &bg_transactions);
    info!("[train] Mined {} candidate patterns", pattern_stats.len());
    eprintln!("[train] Patterns found: {}", pattern_stats.len());

    let factors = miner.build_factor_library(&pattern_stats, n_event_i64, n_bg_i64);
    let selected_factors: Vec<InducingFactor> = factors.into_iter().take(6).collect();
    info!(
        "[train] Selected {} factors for model",
        selected_factors.len()
    );

    // ── Phase 4: Fit models ───────────────────────────────────────
    stage_header("train", 4, 4, "Fitting models");

    let label_dicts: Vec<HashMap<String, String>> = all_labels
        .iter()
        .map(|row| {
            let mut m = HashMap::new();
            for (i, name) in feature_cols.iter().enumerate() {
                m.insert(name.clone(), row.get(i).cloned().unwrap_or_default());
            }
            m
        })
        .collect();

    let mut occ_engine = OccurrenceProbabilityEngine::new();
    occ_engine.fit(
        &label_dicts,
        &targets,
        &selected_factors,
        config.attribution.random_state,
    );

    // ── AUC diagnostic (if computable) ────────────────────────────
    if !label_dicts.is_empty()
        && !selected_factors.is_empty()
        && n_events > 0
        && n_bg > 0
    {
        let preds = occ_engine.predict_proba(&label_dicts, &selected_factors);
        let pos_pred: Vec<f64> = targets
            .iter()
            .zip(preds.iter())
            .filter(|(_, &t)| t > 0.5)
            .map(|(_, &p)| p)
            .collect();
        let neg_pred: Vec<f64> = targets
            .iter()
            .zip(preds.iter())
            .filter(|(_, &t)| t <= 0.5)
            .map(|(_, &p)| p)
            .collect();
        let n_pos = pos_pred.len();
        let n_neg = neg_pred.len();
        if n_pos > 0 && n_neg > 0 {
            let mut auc = 0.0_f64;
            for &pp in &pos_pred {
                auc += neg_pred.iter().filter(|&np| pp > *np).count() as f64;
            }
            auc /= (n_pos * n_neg) as f64;
            eprintln!("[train] Model AUC: {:.4}", auc);
        }
    }

    // ── Export artifacts ──────────────────────────────────────────
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        error!("Failed to create output dir: {}", e);
        return;
    }

    let disc_json = serde_json::to_string_pretty(&discretizer).unwrap_or_default();
    std::fs::write(out_dir.join("basin_discretizer.json"), disc_json).ok();

    let fac_wrapper = serde_json::json!({ "factors": &selected_factors });
    let fac_json = serde_json::to_string_pretty(&fac_wrapper).unwrap_or_default();
    std::fs::write(out_dir.join("basin_factors.json"), fac_json).ok();

    let occ_json = serde_json::to_string_pretty(&occ_engine).unwrap_or_default();
    std::fs::write(out_dir.join("basin_occurrence_model.json"), occ_json).ok();

    let basin_model = serde_json::json!({
        "version": "0.1.0",
        "feature_columns": &feature_cols,
        "discretizer": serde_json::to_value(&discretizer).unwrap_or_default(),
        "selected_factors": &selected_factors,
        "occurrence_engine": serde_json::to_value(&occ_engine).unwrap_or_default(),
        "n_rows": total_rows,
        "n_events": n_events,
        "n_background": n_bg,
        "backend": "logistic_regression",
        "random_state": config.attribution.random_state,
    });
    let basin_json = serde_json::to_string_pretty(&basin_model).unwrap_or_default();
    std::fs::write(out_dir.join("basin_model.json"), basin_json).ok();

    let meta = serde_json::json!({
        "version": "0.1.0",
        "feature_columns": &feature_cols,
        "n_rows": total_rows,
        "n_events": n_events,
        "n_background": n_bg,
        "backend": "logistic_regression",
        "random_state": config.attribution.random_state,
    });
    std::fs::write(
        out_dir.join("basin_model_metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap_or_default(),
    )
    .ok();

    let elapsed = start_time.elapsed();
    info!(
        "[train] Basin training complete 4/4 (100%) elapsed={:.2?} eta=0s",
        elapsed
    );
    eprintln!(
        "[train] Basin model saved to {:?}",
        out_dir.join("basin_model.json")
    );
}

pub fn run_train(config_path: &Path, output_dir: Option<&Path>) {
    setup_logging();
    let start_time = Instant::now();
    let Some(mut config) = load_config(config_path) else {
        return;
    };

    if let Some(od) = output_dir {
        config.paths.output_dir = od.to_path_buf();
    }

    if let Err(e) = std::fs::create_dir_all(&config.paths.output_dir) {
        error!(
            "Failed to create output dir {:?}: {}",
            config.paths.output_dir, e
        );
        return;
    }

    // ── Basin training path: use CLASS-derived single parquet ────
    let feature_path = PathBuf::from(".cache/features/training_features.parquet");
    if feature_path.exists() {
        run_train_basin(&config, output_dir, &feature_path);
        return;
    }

    stage_header("train", 1, 4, "Loading chunk cache");

    let cache_dir = &config.paths.cache_dir;
    let mut all_dfs: Vec<DataFrame> = Vec::new();

    if cache_dir.exists() {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(cache_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| {
                let e = e.ok()?;
                let p = e.path();
                if p.extension().is_some_and(|ext| ext == "parquet") {
                    Some(p)
                } else {
                    None
                }
            })
            .collect();
        paths.sort();
        if !paths.is_empty() {
            for path in &paths {
                if let Ok(file) = std::fs::File::open(path) {
                    let reader = ParquetReader::new(file);
                    match reader.finish() {
                        Ok(df) => all_dfs.push(df),
                        Err(e) => warn!("Failed to read chunk {:?}: {}", path, e),
                    }
                }
            }
        }
    }

        if all_dfs.is_empty() {
        error!("[train] No chunk cache files found in {:?}. Run 'pipeline' before 'train'.", cache_dir);
        return;
    }
    eprintln!("[train] First chunk columns: {:?}", all_dfs[0].get_column_names());
    if let Ok(col) = all_dfs[0].column("disturbance_flag") {
        eprintln!("[train] dist_flag dtype: {:?}, i32 ok={}, f64 ok={}", col.dtype(), col.i32().is_ok(), col.f64().is_ok());
        if let Ok(s) = col.i32() {
            let sum: i32 = s.into_iter().filter_map(|v| v).sum();
            eprintln!("[train] dist_flag sum: {}", sum);
        }
    }

    let total_rows: usize = all_dfs.iter().map(|d| d.height()).sum();
    info!(
        "[train] Loaded {} rows from {} chunks",
        total_rows,
        all_dfs.len()
    );

    stage_header("train", 2, 4, "Fitting discretizer");

    let disc_config = DiscretizerConfig {
        method: config.attribution.discretizer_method.clone(),
        n_bins: config.attribution.discretize_bins,
        ..Default::default()
    };
    let mut discretizer = AdaptiveDiscretizer::new(disc_config);

    let mut feature_names: Vec<String> = [
        "slope",
        "aspect",
        "dist_to_river",
        "dist_to_road",
        "dist_to_railway",
        "dist_to_waterway",
        "temperature",
        "resident",
        "landuse",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for m in 1..=12 {
        feature_names.push(format!("rainfall_month_{:02}", m));
    }

    let limit = config.attribution.discretizer_sample_limit;
    let mut feature_data: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for df in &all_dfs {
        if feature_data.len() >= limit {
            break;
        }
        let h = df.height();
        let n_to_take = (limit - feature_data.len()).min(h);
        for row_idx in 0..n_to_take {
            let mut row: Vec<f64> = Vec::with_capacity(feature_names.len());
            for name in &feature_names {
                let val = df
                    .column(name)
                    .ok()
                    .and_then(|c| c.f64().ok())
                    .and_then(|s| s.get(row_idx))
                    .unwrap_or(f64::NAN);
                row.push(val);
            }
            let target = df
                .column("disturbance_flag")
                .ok()
                .and_then(|c| c.i32().ok())
                .and_then(|s| s.get(row_idx))
                .map(|v| v as f64)
                .unwrap_or(0.0);
            targets.push(target);
            feature_data.push(row);
        }
    }

    if !feature_data.is_empty() {
        discretizer.fit(&feature_names, &feature_data, &targets);
        info!("[train] Discretizer fit on {} samples", feature_data.len());
    } else {
        warn!("[train] No samples for discretizer; using defaults.");
    }

    stage_header("train", 3, 4, "Mining inducing factors");

    let mining_config = FactorMiningConfig {
        max_pattern_size: config.attribution.max_pattern_size,
        min_support: config.attribution.min_support,
        min_relative_risk: config.attribution.min_relative_risk,
        min_stability: config.attribution.min_stability,
        bootstrap_iterations: config.attribution.bootstrap_iterations,
        max_bootstrap_samples: config.attribution.max_bootstrap_samples,
        random_state: config.attribution.random_state,
        score_method: "log_rr".into(),
    };
    let miner = FactorMiner::new(mining_config);

    let limit2 = config.attribution.model_sample_limit;
    let mut all_label_rows: Vec<Vec<String>> = Vec::new();
    let mut model_targets: Vec<f64> = Vec::new();

    use rand::prelude::*;
    use rand::rngs::StdRng;
    use rand::seq::index;
    let seed: [u8; 32] = {
        let mut s = [0u8; 32];
        let b = config.attribution.random_state.to_le_bytes();
        s[..8].copy_from_slice(&b);
        s
    };
    let mut rng = StdRng::from_seed(seed);

    for df in &all_dfs {
        if all_label_rows.len() >= limit2 {
            break;
        }
        let h = df.height();
        let n_to_take = (limit2 - all_label_rows.len()).min(h);
        let indices = index::sample(&mut rng, h, n_to_take);
        let mut chunk_feature_data: Vec<Vec<f64>> = Vec::with_capacity(n_to_take);
        for row_idx in indices.iter() {
            let feature_vec: Vec<f64> = feature_names
                .iter()
                .map(|name| {
                    df.column(name)
                        .ok()
                        .and_then(|c| c.f64().ok())
                        .and_then(|s| s.get(row_idx))
                        .unwrap_or(f64::NAN)
                })
                .collect();
            chunk_feature_data.push(feature_vec);

            let target = df
                .column("disturbance_flag")
                .ok()
                .and_then(|c| c.i32().ok())
                .and_then(|s| s.get(row_idx))
                .map(|v| v as f64)
                .unwrap_or(0.0);
            model_targets.push(target);
        }
        if !chunk_feature_data.is_empty() {
            let labels = discretizer.transform(&feature_names, &chunk_feature_data, true);
            all_label_rows.extend(labels);
        }
    }

    let transactions = miner.transactions_from_labels(&all_label_rows, &feature_names);
    let event_transactions: Vec<Vec<String>> = transactions
        .iter()
        .zip(model_targets.iter())
        .filter(|(_, &t)| t > 0.5)
        .map(|(tx, _)| tx.clone())
        .collect();
    let bg_transactions: Vec<Vec<String>> = transactions
        .iter()
        .zip(model_targets.iter())
        .filter(|(_, &t)| t <= 0.5)
        .map(|(tx, _)| tx.clone())
        .collect();

    let n_event = event_transactions.len() as i64;
    let n_background = bg_transactions.len() as i64;
    let sample_targets: Vec<f64> = model_targets.iter().take(10).copied().collect();
    eprintln!("[train] events={}, bg={}, tx={}, first_targets={:?}", n_event, n_background, transactions.len(), sample_targets);

    let pattern_stats = miner.mine_pattern_candidates(&event_transactions, &bg_transactions);
    info!("[train] Mined {} candidate patterns", pattern_stats.len());

    let factors = miner.build_factor_library(&pattern_stats, n_event, n_background);
    let selected_factors: Vec<InducingFactor> = factors.into_iter().take(6).collect();
    info!(
        "[train] Selected {} factors for model",
        selected_factors.len()
    );

    stage_header("train", 4, 4, "Fitting models");

    let label_dicts: Vec<HashMap<String, String>> = if !all_label_rows.is_empty() {
        all_label_rows
            .iter()
            .map(|row| {
                let mut m = HashMap::new();
                for (i, name) in feature_names.iter().enumerate() {
                    m.insert(name.clone(), row.get(i).cloned().unwrap_or_default());
                }
                m
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut occ_engine = OccurrenceProbabilityEngine::new();
    occ_engine.fit(
        &label_dicts,
        &model_targets,
        &selected_factors,
        config.attribution.random_state,
    );

    let mut class_mask: Vec<usize> = Vec::new();
    let mut class_values: Vec<f64> = Vec::new();
    let max_labeled_rows = all_label_rows.len();
    let mut global_idx = 0usize;
    for df in &all_dfs {
        if global_idx >= max_labeled_rows {
            break;
        }
        let disturbance_classes = df
            .column("disturbance_class")
            .ok()
            .and_then(|c| c.i32().ok());
        for row_idx in 0..df.height() {
            if global_idx >= max_labeled_rows {
                break;
            }
            if let Some(class_id) = disturbance_classes.as_ref().and_then(|s| s.get(row_idx)) {
                if (1..=4).contains(&class_id) {
                    class_mask.push(global_idx);
                    class_values.push(class_id as f64);
                }
            }
            global_idx += 1;
        }
    }

    let mut class_engine = DisturbanceClassEngine::new();
    class_engine.fit(
        &label_dicts,
        &class_values,
        &class_mask,
        &selected_factors,
        config.attribution.random_state,
    );

    let out_dir = &config.paths.output_dir;
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        error!("Failed to create output dir: {}", e);
        return;
    }

    let disc_json = serde_json::to_string_pretty(&discretizer).unwrap_or_default();
    std::fs::write(out_dir.join("discretizer.joblib"), disc_json).ok();

    let occ_json = serde_json::to_string_pretty(&occ_engine).unwrap_or_default();
    std::fs::write(out_dir.join("occurrence_probability_model.json"), occ_json).ok();

    let cls_json = serde_json::to_string_pretty(&class_engine).unwrap_or_default();
    std::fs::write(out_dir.join("disturbance_class_model.json"), cls_json).ok();

    let fac_wrapper = serde_json::json!({ "factors": &selected_factors });
    let fac_json = serde_json::to_string_pretty(&fac_wrapper).unwrap_or_default();
    std::fs::write(out_dir.join("selected_factors.json"), fac_json).ok();

    let meta = serde_json::json!({
        "version": "0.1.0",
        "feature_columns": &feature_names,
        "backend": "logistic_regression",
        "random_state": config.attribution.random_state,
    });
    std::fs::write(
        out_dir.join("model_metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap_or_default(),
    )
    .ok();

    let elapsed = start_time.elapsed();
    info!(
        "[train] Training complete 4/4 (100%) elapsed={:.2?} eta=0s",
        elapsed
    );
    info!("[train] Artifacts saved to {:?}", out_dir);
}

pub fn run_predict(config_path: &Path, year: Option<i32>, workers: Option<usize>) {
    setup_logging();
    let Some(config) = load_config(config_path) else {
        return;
    };

    // ── Basin model path: use single-model prediction ────────────
    let basin_model = config.paths.output_dir.join("basin_model.json");
    let feature_dir = PathBuf::from(".cache/features/prediction_2025");
    if basin_model.exists() && feature_dir.exists() {
        let target_year = year.unwrap_or(config.pipeline.end_year);
        run_predict_basin(&config, None, target_year);
        return;
    }

    let start_time = Instant::now();

    let target_year = year.unwrap_or(config.pipeline.end_year);
    let num_workers = resolve_worker_count(&config, workers);

    info!(
        "[predict][phase 1/3] Target year: {}, workers: {}",
        target_year, num_workers
    );

    let mut mapper = AnnualProbabilityMapper::new(config);
    mapper.load_artifacts();

    if mapper.occurrence_engine.is_none() {
        error!("[predict] No occurrence model loaded - run 'train' first.");
        return;
    }

    let chunk_files = mapper.iter_chunk_files();
    if chunk_files.is_empty() {
        warn!(
            "[predict] No chunk cache files found in {:?}",
            mapper.config.paths.cache_dir
        );
        return;
    }

    info!("[predict] Found {} chunk files", chunk_files.len());

    let total = chunk_files.len();
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) eta={eta}",
        )
        .unwrap(),
    );

    let mut prediction_results = Vec::new();

    for chunk_path in &chunk_files {
        if let Ok(file) = std::fs::File::open(chunk_path) {
            let reader = ParquetReader::new(file);
            if let Ok(df) = reader.finish() {
                if let Some(pred_df) = mapper.predict_chunk(&df, target_year) {
                    prediction_results.push(pred_df);
                }
            }
        }
        pb.inc(1);
    }
    pb.finish_with_message("Prediction complete");

    info!(
        "[predict][phase 2/3] Predicted {} chunks",
        prediction_results.len()
    );

    let output_dir = mapper.config.paths.output_dir.join("probability_maps");
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        error!("Failed to create output dir {:?}: {}", output_dir, e);
        return;
    }

    let h = 256;
    let w = 256;
    let nodata = -9999.0f32;
    let mut prob_raster = vec![nodata; h * w];

    for df in &prediction_results {
        let raster_col = df.column("raster_col").ok().and_then(|c| c.i64().ok());
        let raster_row = df.column("raster_row").ok().and_then(|c| c.i64().ok());
        let prob_col = df
            .column("disturbance_probability")
            .ok()
            .and_then(|c| c.f64().ok());

        if let (Some(cols), Some(rows), Some(probs)) = (raster_col, raster_row, prob_col) {
            for i in 0..df.height() {
                if let (Some(c), Some(r), Some(p)) = (cols.get(i), rows.get(i), probs.get(i)) {
                    let idx = r as usize * w + c as usize;
                    if idx < prob_raster.len() {
                        prob_raster[idx] = p as f32;
                    }
                }
            }
        }
    }

    let meta = serde_json::json!({
        "width": w,
        "height": h,
        "year": target_year,
        "nodata": nodata,
        "crs": "EPSG:4326",
        "geo_transform": [0.0, 0.01, 0.0, 10.0, 0.0, -0.01],
    });
    std::fs::write(
        output_dir.join(format!("probability_map_{}.json", target_year)),
        serde_json::to_string_pretty(&meta).unwrap_or_default(),
    )
    .ok();

    let prob_arr = ndarray::Array2::from_shape_vec((h, w), prob_raster.clone()).unwrap_or_default();
    let flat: Vec<f32> = prob_arr.iter().copied().collect();
    std::fs::write(
        output_dir.join(format!("disturbance_probability_{}.npy", target_year)),
        serde_json::to_string(&flat).unwrap_or_default(),
    )
    .ok();

    let elapsed = start_time.elapsed();
    let pct = if total > 0 { 100 } else { 0 };
    info!(
        "[predict] Prediction complete {}/{} ({}%) elapsed={:.2?} eta=0s",
        total, total, pct, elapsed
    );
    info!("[predict] Maps written to {:?}", output_dir);
}

/// Predict disturbance probabilities for the entire CLASS grid using a
/// basin-wide model, then write a single GeoTIFF probability map.
///
/// Reads tile parquets from `.cache/features/prediction_2025/`, loads the
/// basin model from `results/models/basin_model.json`, discretizes features,
/// runs inference, and assembles a full-AOI GeoTIFF.
fn run_predict_basin(config: &AppConfig, output_dir: Option<&Path>, _target_year: i32) {
    setup_logging();
    let start_time = Instant::now();

    let out_dir = output_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| config.paths.output_dir.join("predictions"));

    // ── Phase 1: Load basin model ──────────────────────────────────
    let basin_model_path = config.paths.output_dir.join("basin_model.json");
    if !basin_model_path.exists() {
        error!(
            "[predict] Basin model not found at {:?}. Run 'train' first.",
            basin_model_path
        );
        return;
    }

    let model_raw = match std::fs::read_to_string(&basin_model_path) {
        Ok(s) => s,
        Err(e) => {
            error!("[predict] Failed to read basin model: {}", e);
            return;
        }
    };
    let model_json: serde_json::Value =
        match serde_json::from_str(&model_raw) {
            Ok(v) => v,
            Err(e) => {
                error!("[predict] Failed to parse basin model: {}", e);
                return;
            }
        };

    // Extract discretizer.
    let discretizer: AdaptiveDiscretizer = match serde_json::from_value(
        model_json["discretizer"].clone(),
    ) {
        Ok(d) => d,
        Err(e) => {
            error!(
                "[predict] Failed to deserialize discretizer: {}",
                e
            );
            return;
        }
    };

    // Extract feature columns.
    let feature_columns: Vec<String> = match model_json["feature_columns"]
        .as_array()
    {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        None => {
            error!("[predict] Missing 'feature_columns' in basin model");
            return;
        }
    };

    // Extract selected factors.
    let selected_factors: Vec<InducingFactor> = match serde_json::from_value(
        model_json["selected_factors"].clone(),
    ) {
        Ok(f) => f,
        Err(e) => {
            error!("[predict] Failed to deserialize factors: {}", e);
            return;
        }
    };

    // Extract occurrence engine.
    let occ_engine: OccurrenceProbabilityEngine =
        match serde_json::from_value(
            model_json["occurrence_engine"].clone(),
        ) {
            Ok(eng) => eng,
            Err(e) => {
                error!(
                    "[predict] Failed to deserialize occurrence engine: {}",
                    e
                );
                return;
            }
        };

    info!(
        "[predict][basin] Model loaded: {} features, {} factors, backend={}",
        feature_columns.len(),
        selected_factors.len(),
        occ_engine.backend
    );

    // ── Phase 2: Open CLASS grid for output dimensions ─────────────
    use if_pipeline::class_grid::ClassGrid;
    let class_grid = match ClassGrid::open(
        "data/disturbance_class/disturbance_class_res.tif",
        2016,
    ) {
        Ok(g) => g,
        Err(e) => {
            error!("[predict] Failed to open CLASS grid: {}", e);
            return;
        }
    };
    info!(
        "[predict][basin] CLASS grid: {}×{}, CRS={}",
        class_grid.width, class_grid.height, class_grid.crs
    );

    // ── Phase 3: Create output GeoTIFF ─────────────────────────────
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        error!("[predict] Failed to create output dir: {}", e);
        return;
    }

    let output_path = out_dir.join("probability_2025.tif");

    #[cfg(feature = "gdal")]
    let ds = {
        use gdal::DriverManager;
        let driver = DriverManager::get_driver_by_name("GTiff")
            .expect("GTiff driver not available");
        let mut ds = driver
            .create_with_band_type::<f32, _>(
                &output_path,
                class_grid.width,
                class_grid.height,
                1,
            )
            .unwrap_or_else(|e| {
                panic!("Failed to create GeoTIFF {:?}: {}", output_path, e)
            });
        ds.set_geo_transform(&class_grid.geo_transform)
            .expect("set geo_transform");
        ds.set_projection(&class_grid.crs)
            .expect("set projection");
        info!(
            "[predict][basin] Created output GeoTIFF: {:?}",
            output_path
        );
        ds
    };

    // ── Phase 4: Predict tiles and write to GeoTIFF ────────────────
    let feature_dir = PathBuf::from(".cache/features/prediction_2025");
    if !feature_dir.exists() {
        error!(
            "[predict] Feature tiles not found at {:?}. Run feature extraction first.",
            feature_dir
        );
        return;
    }

    let mut tile_paths: Vec<PathBuf> = std::fs::read_dir(&feature_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            if p.extension().is_some_and(|ext| ext == "parquet") {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    tile_paths.sort();

    if tile_paths.is_empty() {
        error!("[predict] No tile parquets found in {:?}", feature_dir);
        return;
    }

    let total_tiles = tile_paths.len();
    info!("[predict][basin] Found {} tile parquets", total_tiles);

    let pb = ProgressBar::new(total_tiles as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) eta={eta}",
        )
        .unwrap(),
    );

    // Feature columns for prediction: everything except row/col metadata.
    let metadata_cols: Vec<&str> = vec!["row", "col"];
    let pred_feature_cols: Vec<String> = feature_columns
        .iter()
        .filter(|c| !metadata_cols.contains(&c.as_str()))
        .cloned()
        .collect();

    info!(
        "[predict][basin] Prediction feature columns ({}): {:?}",
        pred_feature_cols.len(),
        &pred_feature_cols[..pred_feature_cols.len().min(10)]
    );

    for tile_path in &tile_paths {
        let file = match std::fs::File::open(tile_path) {
            Ok(f) => f,
            Err(e) => {
                warn!("[predict] Failed to open {:?}: {}", tile_path, e);
                pb.inc(1);
                continue;
            }
        };
        let reader = ParquetReader::new(file);
        let df = match reader.finish() {
            Ok(df) => df,
            Err(e) => {
                warn!("[predict] Failed to read {:?}: {}", tile_path, e);
                pb.inc(1);
                continue;
            }
        };

        let h = df.height();
        if h == 0 {
            pb.inc(1);
            continue;
        }

        // Extract row/col for georeferencing.
        let rows = df
            .column("row")
            .ok()
            .and_then(|c| c.i32().ok());
        let cols = df
            .column("col")
            .ok()
            .and_then(|c| c.i32().ok());

        // Build feature vectors.
        let mut feature_data: Vec<Vec<f64>> = Vec::with_capacity(h);
        for row_idx in 0..h {
            let mut row: Vec<f64> =
                Vec::with_capacity(pred_feature_cols.len());
            for name in &pred_feature_cols {
                let val = df
                    .column(name)
                    .ok()
                    .and_then(|c| c.f64().ok())
                    .and_then(|s| s.get(row_idx))
                    .unwrap_or(f64::NAN);
                row.push(val);
            }
            feature_data.push(row);
        }

        // Discretize.
        let labels = discretizer.transform(
            &pred_feature_cols,
            &feature_data,
            true,
        );

        // Convert to HashMap rows for prediction.
        let label_dicts: Vec<HashMap<String, String>> = labels
            .iter()
            .map(|row| {
                let mut m = HashMap::new();
                for (i, name) in pred_feature_cols.iter().enumerate() {
                    m.insert(
                        name.clone(),
                        row.get(i).cloned().unwrap_or_default(),
                    );
                }
                m
            })
            .collect();

        // Predict probabilities.
        let probs = occ_engine.predict_proba(
            &label_dicts,
            &selected_factors,
        );

        // Write to GeoTIFF.
        #[cfg(feature = "gdal")]
        {
            for i in 0..h {
                let r = rows
                    .as_ref()
                    .and_then(|s| s.get(i))
                    .unwrap_or(-1) as usize;
                let c = cols
                    .as_ref()
                    .and_then(|s| s.get(i))
                    .unwrap_or(-1) as usize;
                if r >= class_grid.height || c >= class_grid.width {
                    continue;
                }
                let prob = probs.get(i).copied().unwrap_or(0.0) as f32;
                let mut band = ds.rasterband(1)
                    .expect("rasterband 1");
                let mut buf = gdal::raster::Buffer::new((1, 1), vec![prob]);
                band.write(
                    (c as isize, r as isize),
                    (1, 1),
                    &mut buf,
                )
                .ok();
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("Basin prediction complete");

    // Flush and close the GeoTIFF.
    #[cfg(feature = "gdal")]
    {
        drop(ds);
    }

    let elapsed = start_time.elapsed();
    info!(
        "[predict][basin] Prediction complete: {} tiles, elapsed={:.2?}",
        total_tiles, elapsed
    );
    info!(
        "[predict][basin] Probability map written to {:?}",
        output_path
    );
}

pub fn run_all(config_path: &Path, dry_run: bool) {
    setup_logging();
    let start_time = Instant::now();

    info!("[all][phase 0/3] Starting full pipeline");

    if dry_run {
        run_pipeline(config_path, None, true);
        let elapsed = start_time.elapsed();
        info!(
            "[all] Dry-run complete 0/0 (100%) elapsed={:.2?} eta=0s",
            elapsed
        );
        return;
    }

    stage_header("all", 1, 3, "Running feature extraction pipeline");
    run_pipeline(config_path, None, false);

    stage_header("all", 2, 3, "Training attribution model");
    run_train(config_path, None);

    stage_header("all", 3, 3, "Generating probability maps");
    let Some(config) = load_config(config_path) else {
        return;
    };
    let end_year = config.pipeline.end_year;
    run_predict(config_path, Some(end_year), None);

    let elapsed = start_time.elapsed();
    info!(
        "[all] Full pipeline complete 3/3 (100%) elapsed={:.2?} eta=0s",
        elapsed
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_test_config(root: &Path, cache_dir: &Path, output_dir: &Path) -> PathBuf {
        let config_path = root.join("config.yaml");
        let yaml = format!(
            r#"
paths:
  cache_dir: "{}"
  output_dir: "{}"
pipeline:
  start_year: 2020
  end_year: 2020
  chunk_size: 1024
  num_workers: 1
  max_auto_workers: 1
  memory_target_fraction: 0.7
  estimated_worker_gb: 1.0
attribution:
  lookback_years: 3
  background_sample_rate: 0.05
  discretizer_method: "supervised_tree"
  discretize_bins: 3
  max_pattern_size: 2
  min_support: 0.01
  min_relative_risk: 1.0
  bootstrap_iterations: 1
  min_stability: 0.0
  max_bootstrap_samples: 100
  max_network_factors: 3
  dominant_threshold: 0.55
  synergy_threshold: 0.75
  score_method: "log_rr"
  spatial_buffer_distance: 0.0
  deduplicate_temporal: true
  bn_bootstrap_iterations: 1
  bn_edge_frequency_threshold: 0.5
  bdeu_equivalent_sample_size: 10.0
  redundancy_threshold: 0.1
  random_state: 42
  discretizer_sample_limit: 200
  model_sample_limit: 200
  progress_log_every: 1
  run_num_workers: 1
  run_max_auto_workers: 1
  run_memory_target_fraction: 0.7
  run_estimated_worker_gb: 1.0
  predict_num_workers: 1
  predict_max_auto_workers: 1
  predict_memory_target_fraction: 0.7
  predict_estimated_worker_gb: 1.0
  analyze_by_class: false
  min_count_per_class: 1
  class_prior_lift_threshold: 1.0
  top_n_per_class: 5
data_sources:
  aoi_shp: "data/aoi/missing.shp"
  dem_tif: "data/dem/missing.tif"
  landuse_dir: "data/landuse"
  rainfall_dir: "data/rainfall"
  temperature_dir: "data/temperature"
  resident_dir: "data/resident"
  river_shp_dir: "data/river"
  road_shp_dir: "data/road"
  ccdc_dir: "data/ccdc_res"
  disturbance_class_tif: "data/disturbance_class/missing.tif"
  landuse_file_pattern: "CLCD_{{year}}.tif"
  road_file_pattern: "osm_{{year}}_roads.shp"
  railway_file_pattern: "osm_{{year}}_railways.shp"
  waterway_file_pattern: "osm_{{year}}_waterways.shp"
  rainfall_monthly_file_pattern: "pre{{year}}{{month:02d}}.tif"
  temperature_file_pattern: "tmp_{{year}}.nc"
  resident_file_pattern: "GURS_{{year}}.tif"
  ccdc_file_pattern: "{{year}}.tif"
  disturbance_class_start_year: 2016
  dem_nodata: -9999.0
  resident_nodata: 127.0
"#,
            cache_dir.display(),
            output_dir.display()
        );
        std::fs::write(&config_path, yaml).unwrap();
        config_path
    }

    fn write_training_chunk(cache_dir: &Path) {
        std::fs::create_dir_all(cache_dir).unwrap();
        let mut df = DataFrame::new(vec![
            Column::new("point_id".into(), [1i64, 2, 3, 4, 5, 6]),
            Column::new("x".into(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            Column::new("y".into(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            Column::new("year".into(), [2020i32; 6]),
            Column::new("slope".into(), [10.0, 12.0, 14.0, 30.0, 32.0, 34.0]),
            Column::new("aspect".into(), [90.0, 100.0, 110.0, 200.0, 210.0, 220.0]),
            Column::new(
                "dist_to_river".into(),
                [100.0, 120.0, 140.0, 2000.0, 2200.0, 2400.0],
            ),
            Column::new(
                "dist_to_road".into(),
                [500.0, 520.0, 540.0, 2500.0, 2600.0, 2700.0],
            ),
            Column::new("dist_to_railway".into(), [5000.0; 6]),
            Column::new("dist_to_waterway".into(), [3000.0; 6]),
            Column::new("temperature".into(), [15.0, 16.0, 17.0, 18.0, 19.0, 20.0]),
            Column::new("resident".into(), [10.0, 11.0, 12.0, 30.0, 31.0, 32.0]),
            Column::new("landuse".into(), [1i32, 1, 1, 2, 2, 2]),
            Column::new("disturbance_flag".into(), [1i32, 1, 1, 1, 1, 1]),
            Column::new("disturbance_class".into(), [1i32, 1, 1, 3, 3, 3]),
            Column::new("disturbance_month".into(), [5.0, 5.0, 5.0, 8.0, 8.0, 8.0]),
        ])
        .unwrap();
        let file = std::fs::File::create(cache_dir.join("chunk_000000.parquet")).unwrap();
        ParquetWriter::new(file).finish(&mut df).unwrap();
    }

    #[test]
    fn production_train_requires_existing_cache() {
        let temp = tempdir().unwrap();
        let cache_dir = temp.path().join("missing-cache");
        let out_dir = temp.path().join("models");
        let config_path = write_test_config(temp.path(), &cache_dir, &out_dir);

        run_train(&config_path, None);

        assert!(
            !out_dir.join("occurrence_probability_model.json").exists(),
            "production train must not silently generate synthetic cache when cache is missing"
        );
    }

    #[test]
    fn class_training_uses_disturbance_class_values() {
        let temp = tempdir().unwrap();
        let cache_dir = temp.path().join("cache");
        let out_dir = temp.path().join("models");
        write_training_chunk(&cache_dir);
        let config_path = write_test_config(temp.path(), &cache_dir, &out_dir);

        run_train(&config_path, None);

        let json = std::fs::read_to_string(out_dir.join("disturbance_class_model.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["classes"], serde_json::json!([1, 3]));
    }

    #[test]
    fn missing_config_does_not_panic() {
        let temp = tempdir().unwrap();
        let missing = temp.path().join("missing.yaml");

        let result = std::panic::catch_unwind(|| run_train(&missing, None));

        assert!(
            result.is_ok(),
            "missing config should be reported without panicking"
        );
    }
}

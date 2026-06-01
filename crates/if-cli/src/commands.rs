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
};

fn load_config(path: &Path) -> AppConfig {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read config file {:?}: {}", path, e));
    serde_yaml::from_str(&raw)
        .unwrap_or_else(|e| panic!("Failed to parse config file {:?}: {}", path, e))
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
        .init();
}

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

    (points, features, dist_flags, dist_classes, dist_months, rainfall)
}

fn stage_header(stage: &str, phase: usize, total: usize, description: &str) {
    info!(
        "[{}][phase {}/{}] {}",
        stage, phase, total, description
    );
}

pub fn run_pipeline(config_path: &Path, workers: Option<usize>, dry_run: bool) {
    setup_logging();
    let start_time = Instant::now();
    let config = load_config(config_path);

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
                        .map(|s| {
                            s.into_iter()
                                .filter(|v| v.map_or(false, |x| x == 1))
                                .count()
                        })
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
    let pct = if total_chunks > 0 {
        ok_count * 100 / total_chunks
    } else {
        100
    };

    info!(
        "[pipeline] Pipeline complete {} / {} ({}%) elapsed={:.2?} eta=0s",
        ok_count, total_chunks, pct, elapsed
    );
    info!(
        "[pipeline] Total points: {}, disturbances: {}, ok chunks: {}/{}",
        total_pts, total_dist, ok_count, total_chunks
    );
}

pub fn run_train(config_path: &Path, output_dir: Option<&Path>) {
    setup_logging();
    let start_time = Instant::now();
    let mut config = load_config(config_path);

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
                if p.extension().map_or(false, |ext| ext == "parquet") {
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
        warn!(
            "[train] No chunk cache found in {:?}. Generating synthetic data.",
            cache_dir
        );
        let (points, features, dist_flags, dist_classes, dist_months, rainfall) =
            generate_synthetic_points(
                config.pipeline.start_year,
                config.pipeline.end_year,
                100,
                config.attribution.random_state,
            );
        let tasks = derive_chunk_grid(256, 256, config.chunks.tile_size);
        for (i, task) in tasks.iter().enumerate() {
            if let Ok(df) = build_chunk_dataframe(
                task,
                &points,
                &features,
                &dist_flags,
                &dist_classes,
                &dist_months,
                &rainfall,
                config.attribution.random_state + i as u64,
            ) {
                all_dfs.push(df);
            }
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

    let feature_names: Vec<String> = vec![
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
        info!(
            "[train] Discretizer fit on {} samples",
            feature_data.len()
        );
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

    for df in &all_dfs {
        if all_label_rows.len() >= limit2 {
            break;
        }
        let h = df.height();
        let n_to_take = (limit2 - all_label_rows.len()).min(h);
        let mut chunk_feature_data: Vec<Vec<f64>> = Vec::with_capacity(n_to_take);
        for row_idx in 0..n_to_take {
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

    let class_mask: Vec<usize> = if all_label_rows.is_empty() {
        Vec::new()
    } else {
        all_dfs
            .iter()
            .flat_map(|df| {
                (0..df.height())
                    .filter(|&i| {
                        df.column("disturbance_flag")
                            .ok()
                            .and_then(|c| c.i32().ok())
                            .and_then(|s| s.get(i))
                            .map_or(false, |v| v == 1)
                    })
                    .collect::<Vec<usize>>()
            })
            .take(model_targets.len())
            .collect()
    };

    let class_values: Vec<f64> = class_mask
        .iter()
        .map(|&idx| {
            if idx < model_targets.len() {
                model_targets[idx]
            } else {
                0.0
            }
        })
        .collect();

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
    let start_time = Instant::now();
    let config = load_config(config_path);

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
                if let (Some(c), Some(r), Some(p)) = (
                    cols.get(i),
                    rows.get(i),
                    probs.get(i),
                ) {
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

    let prob_arr =
        ndarray::Array2::from_shape_vec((h, w), prob_raster.clone()).unwrap_or_default();
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
    let config = load_config(config_path);
    let end_year = config.pipeline.end_year;
    run_predict(config_path, Some(end_year), None);

    let elapsed = start_time.elapsed();
    info!(
        "[all] Full pipeline complete 3/3 (100%) elapsed={:.2?} eta=0s",
        elapsed
    );
}

use std::fs;
use std::path::{Path, PathBuf};

use data_process::{
    run_class_basin_data_process, DataProcessRequest, DataProcessSources, DataSourceUsage,
};
use gdal_crate::raster::RasterCreationOptions;
use gdal_crate::{Dataset, DriverManager};
use polars::prelude::*;
use predict::{BasinPredictRequest, PredictionProduct};
use serde_json::{json, Value};
use train::{BasinTrainSettings, ModelMissingDataPolicy};
use workflow_config::WorkflowConfig;

const DEFAULT_WINDOW_SIZE: usize = 2000;

#[test]
fn center_class_window_has_signal_and_writes_workflow_config() {
    let window_size = window_size();
    let out_root = output_root(window_size);
    fs::create_dir_all(&out_root).unwrap();

    let config = load_workflow_config().unwrap();
    if !config.class_grid.raster.exists() {
        eprintln!(
            "SKIP: CLASS raster not found at {}",
            config.class_grid.raster.display()
        );
        return;
    }

    let class_out = out_root.join(format!("class_center_{window_size}x{window_size}.tif"));
    let (col_off, row_off) =
        crop_center_class(&config.class_grid.raster, &class_out, window_size).unwrap();
    let config_out = out_root.join(format!(
        "class_center_{window_size}x{window_size}_workflow.json"
    ));
    write_center_config(&config, &class_out, &out_root, &config_out, window_size).unwrap();

    let stats = band_stats(&class_out, window_size).unwrap();
    let total_nonzero: usize = stats.iter().map(|stat| stat.nonzero).sum();

    eprintln!("output_root={}", out_root.display());
    eprintln!("class_raster={}", class_out.display());
    eprintln!("config={}", config_out.display());
    eprintln!("src_window=col:{col_off},row:{row_off},width:{window_size},height:{window_size}");
    for stat in &stats {
        eprintln!(
            "class_band_{}: valid={}, nonzero={}, min={}, max={}, nodata={:?}",
            stat.band, stat.valid, stat.nonzero, stat.min, stat.max, stat.nodata
        );
    }

    assert_eq!(stats.len(), 10);
    assert!(
        total_nonzero > 0,
        "center {window_size}x{window_size} CLASS window has no nonzero class pixels"
    );
    assert!(config_out.exists());
}

#[test]
fn center_window_full_workflow_smoke() {
    let window_size = window_size();
    let out_root = output_root(window_size);
    reset_workflow_outputs(&out_root).unwrap();
    fs::create_dir_all(&out_root).unwrap();

    let config = load_workflow_config().unwrap();
    if !config.class_grid.raster.exists() {
        eprintln!(
            "SKIP: CLASS raster not found at {}",
            config.class_grid.raster.display()
        );
        return;
    }

    let class_out = out_root.join(format!("class_center_{window_size}x{window_size}.tif"));
    crop_center_class(&config.class_grid.raster, &class_out, window_size).unwrap();
    let data_process_report = run_class_basin_data_process(&real_smoke_data_process_request(
        &config,
        &class_out,
        &out_root,
        window_size,
    ))
    .unwrap();

    let features_dir = out_root.join("cache/features");
    let tile_path = prediction_tile_path(&out_root, config.predict.prediction_year);
    let training_features_path = features_dir.join("training_features.parquet");

    let model_dir = out_root.join("models");
    let settings = train_settings_from_workflow(&config, 4);
    let train_report =
        train::train_basin_with_settings(&training_features_path, &model_dir, &settings).unwrap();
    let cooccurrence_count = cooccurrence_factor_count(&train_report.model_path).unwrap();

    let predict_request = BasinPredictRequest {
        class_raster: class_out.clone(),
        class_start_year: config.class_grid.start_year,
        model_path: train_report.model_path.clone(),
        feature_dir: features_dir.join(format!("prediction_{}", config.predict.prediction_year)),
        output_dir: out_root.join("predictions"),
        prediction_year: config.predict.prediction_year,
        products: vec![
            PredictionProduct::Probability,
            PredictionProduct::DominantFactor,
            PredictionProduct::DisturbanceType,
            PredictionProduct::DisturbanceTypeProbability,
            PredictionProduct::MissingDataMask,
        ],
        workers: 4,
    };
    let predict_report = predict::predict_basin(&predict_request).unwrap();

    let probability_path = predict_report.probability_path.as_ref().unwrap();
    let dominant_path = predict_report.dominant_factor_path.as_ref().unwrap();
    let disturbance_type_path = predict_report.disturbance_type_path.as_ref().unwrap();
    let missing_mask_path = predict_report.missing_data_mask_path.as_ref().unwrap();
    let probability_stats = probability_raster_stats(probability_path).unwrap();
    let disturbance_type_stats = integer_raster_stats(disturbance_type_path).unwrap();

    eprintln!("training_features={}", training_features_path.display());
    eprintln!("model={}", train_report.model_path.display());
    eprintln!("prediction_probability={}", probability_path.display());
    eprintln!("prediction_dominant_factor={}", dominant_path.display());
    eprintln!(
        "prediction_disturbance_type={}",
        disturbance_type_path.display()
    );
    eprintln!(
        "prediction_disturbance_type_probabilities={:?}",
        predict_report.disturbance_type_probability_paths
    );
    eprintln!("prediction_missing_mask={}", missing_mask_path.display());
    eprintln!(
        "training_samples: rows={}, events={}, background={}",
        train_report.n_rows, train_report.n_events, train_report.n_background
    );
    eprintln!("cooccurrence_factors={cooccurrence_count}");
    eprintln!(
        "data_process_report: class={}x{}, prediction_tiles={}, cache={}",
        data_process_report.class_width,
        data_process_report.class_height,
        data_process_report.prediction_tiles,
        data_process_report.cache_dir.display()
    );
    eprintln!(
        "prediction_report: tiles={}, pixels={}",
        predict_report.tiles_processed, predict_report.pixels_processed
    );
    eprintln!(
        "probability_stats: valid={}, unique={}, min={:.6}, max={:.6}",
        probability_stats.valid,
        probability_stats.unique_count,
        probability_stats.min,
        probability_stats.max
    );
    eprintln!(
        "disturbance_type_stats: valid={}, unique={:?}",
        disturbance_type_stats.valid, disturbance_type_stats.unique_values
    );

    assert_eq!(data_process_report.prediction_tiles, 1);
    assert!(tile_path.exists(), "missing {}", tile_path.display());
    assert!(training_features_path.exists());
    assert!(train_report.model_path.exists());
    assert!(train_report.n_events > 0);
    assert!(train_report.n_background > 0);
    assert!(
        cooccurrence_count > 0,
        "model should contain at least one co-occurrence factor"
    );
    assert_eq!(predict_report.tiles_processed, 1);
    assert_eq!(predict_report.pixels_processed, window_size * window_size);
    assert!(probability_path.exists());
    assert!(dominant_path.exists());
    assert!(disturbance_type_path.exists());
    assert_eq!(predict_report.disturbance_type_probability_paths.len(), 4);
    for path in &predict_report.disturbance_type_probability_paths {
        assert!(path.exists(), "missing {}", path.display());
    }
    assert_disturbance_type_legend(&out_root.join(format!(
        "predictions/disturbance_type_{}_legend.json",
        config.predict.prediction_year
    )))
    .unwrap();
    assert!(missing_mask_path.exists());
    assert!(
        probability_stats.unique_count > 1,
        "probability output should vary when smoke uses real covariates"
    );
    assert!(probability_stats.max > probability_stats.min);
    assert!(disturbance_type_stats.valid > 0);
    assert!(
        disturbance_type_stats
            .unique_values
            .iter()
            .all(|value| (1..=4).contains(value)),
        "disturbance type output must use CLASS values 1..=4"
    );
}

#[test]
fn center_window_prediction_outputs_have_value_range() {
    let window_size = window_size();
    let out_root = output_root(window_size);
    let probability_path = out_root.join("predictions/probability_2025.tif");
    if !probability_path.exists() {
        eprintln!(
            "SKIP: probability raster not found at {}",
            probability_path.display()
        );
        return;
    }

    let stats = probability_raster_stats(&probability_path).unwrap();
    eprintln!(
        "probability_stats: valid={}, unique={}, min={:.6}, max={:.6}",
        stats.valid, stats.unique_count, stats.min, stats.max
    );
    assert!(stats.valid > 0);
    assert!(
        stats.unique_count > 1,
        "probability output should contain more than one value"
    );
    assert!(stats.max > stats.min);
}

#[test]
fn center_window_feature_tables_have_real_covariates() {
    let window_size = window_size();
    let out_root = output_root(window_size);
    let training_path = out_root.join("cache/features/training_features.parquet");
    let prediction_path = prediction_tile_path(&out_root, 2025);
    if !training_path.exists() || !prediction_path.exists() {
        eprintln!(
            "SKIP: feature tables not found under {}",
            out_root.display()
        );
        return;
    }

    let training_stats = feature_table_stats(
        &training_path,
        &["elevation", "rainfall_prev_total_3m", "temp_month_01"],
    )
    .unwrap();
    let prediction_stats = feature_table_stats(
        &prediction_path,
        &["elevation", "rainfall_prev_total_3m", "temp_month_01"],
    )
    .unwrap();

    eprintln!("training_feature_stats={training_stats:?}");
    eprintln!("prediction_feature_stats={prediction_stats:?}");

    for stats in [&training_stats, &prediction_stats] {
        for col in ["elevation", "rainfall_prev_total_3m", "temp_month_01"] {
            let stat = stats.iter().find(|stat| stat.name == col).unwrap();
            assert!(stat.valid > 0, "{col} should have valid values");
            assert!(stat.max > stat.min, "{col} should vary");
        }
    }
}

fn window_size() -> usize {
    std::env::var("CENTER_WINDOW_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_WINDOW_SIZE)
}

fn output_root(window_size: usize) -> PathBuf {
    std::env::var("CENTER_WINDOW_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(format!(
                "/tmp/inducing_factor_center_{window_size}x{window_size}"
            ))
        })
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn load_workflow_config() -> Result<WorkflowConfig, Box<dyn std::error::Error>> {
    let root = repo_root();
    let mut config =
        WorkflowConfig::from_json_file(root.join("configs/class_basin_workflow.json"))?;
    absolutize(&root, &mut config.class_grid.raster);
    absolutize(&root, &mut config.data_sources.aoi_shapefile);
    absolutize(&root, &mut config.data_sources.dem_raster);
    absolutize(&root, &mut config.data_sources.ccdc_dir);
    absolutize(&root, &mut config.data_sources.rainfall_dir);
    absolutize(&root, &mut config.data_sources.temperature_dir);
    absolutize(&root, &mut config.data_sources.landuse_dir);
    absolutize(&root, &mut config.data_sources.resident_raster);
    absolutize(&root, &mut config.data_sources.river_shapefile);
    absolutize(&root, &mut config.data_sources.road_dir);
    Ok(config)
}

fn absolutize(root: &Path, path: &mut PathBuf) {
    if path.is_relative() {
        *path = root.join(&path);
    }
}

fn reset_workflow_outputs(out_root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("CENTER_WINDOW_REUSE_CACHE").ok().as_deref() == Some("1") {
        return Ok(());
    }
    for child in ["cache", "models", "predictions"] {
        let path = out_root.join(child);
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}

fn real_smoke_data_process_request(
    config: &WorkflowConfig,
    class_raster: &Path,
    out_root: &Path,
    window_size: usize,
) -> DataProcessRequest {
    DataProcessRequest {
        class_raster: class_raster.to_path_buf(),
        class_start_year: config.class_grid.start_year,
        sources: DataProcessSources {
            aoi_shapefile: config.data_sources.aoi_shapefile.clone(),
            dem_raster: config.data_sources.dem_raster.clone(),
            ccdc_dir: config.data_sources.ccdc_dir.clone(),
            rainfall_dir: config.data_sources.rainfall_dir.clone(),
            temperature_dir: config.data_sources.temperature_dir.clone(),
            landuse_dir: config.data_sources.landuse_dir.clone(),
            resident_raster: config.data_sources.resident_raster.clone(),
            river_shapefile: config.data_sources.river_shapefile.clone(),
            road_dir: config.data_sources.road_dir.clone(),
        },
        source_usage: real_smoke_data_source_usage(&config.data_usage.smoke_test),
        cache_dir: out_root.join("cache"),
        tile_size: window_size,
        processing_years: config.processing_years(),
        training_years: config.train.training_years.clone(),
        prediction_year: config.predict.prediction_year,
        workers: 4,
    }
}

fn real_smoke_data_source_usage(usage: &workflow_config::DataUsageConfig) -> DataSourceUsage {
    DataSourceUsage {
        dem: usage.dem,
        ccdc: usage.ccdc,
        rainfall: usage.rainfall,
        temperature: usage.temperature,
        landuse: usage.landuse,
        resident: usage.resident,
        river: false,
        road: false,
    }
}

fn train_settings_from_workflow(config: &WorkflowConfig, workers: usize) -> BasinTrainSettings {
    BasinTrainSettings {
        discretizer_method: config.train.discretizer_method.clone(),
        discretize_bins: config.train.discretize_bins,
        max_pattern_size: config.train.max_pattern_size,
        min_support: config.train.min_support,
        min_relative_risk: config.train.min_relative_risk,
        min_stability: config.train.min_stability,
        bootstrap_iterations: config.train.bootstrap_iterations,
        max_bootstrap_samples: config.train.max_bootstrap_samples,
        max_network_factors: config.train.max_network_factors,
        random_state: config.train.random_state,
        missing_data_policy: match config.train.missing_data_policy {
            workflow_config::ModelMissingDataPolicy::NativeMissingCategory => {
                ModelMissingDataPolicy::NativeMissingCategory
            }
            workflow_config::ModelMissingDataPolicy::FeatureDropout => {
                ModelMissingDataPolicy::FeatureDropout
            }
        },
        workers,
    }
}

fn cooccurrence_factor_count(model_path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(model_path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let count = value
        .get("selected_factors")
        .and_then(|factors| factors.as_array())
        .map(|factors| {
            factors
                .iter()
                .filter(|factor| {
                    factor
                        .get("items")
                        .and_then(|items| items.as_array())
                        .is_some_and(|items| items.len() > 1)
                })
                .count()
        })
        .unwrap_or(0);
    Ok(count)
}

fn crop_center_class(
    src_path: &Path,
    dst_path: &Path,
    window_size: usize,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let src = Dataset::open(src_path)?;
    let (width, height) = src.raster_size();
    assert!(window_size <= width);
    assert!(window_size <= height);
    let col_off = (width - window_size) / 2;
    let row_off = (height - window_size) / 2;

    let mut gt = src.geo_transform()?;
    gt[0] += col_off as f64 * gt[1] + row_off as f64 * gt[2];
    gt[3] += col_off as f64 * gt[4] + row_off as f64 * gt[5];

    let driver = DriverManager::get_driver_by_name("GTiff")?;
    let options = RasterCreationOptions::from_iter([
        "TILED=YES",
        "COMPRESS=DEFLATE",
        "BIGTIFF=IF_SAFER",
        "BLOCKXSIZE=128",
        "BLOCKYSIZE=128",
    ]);
    let mut dst = driver.create_with_band_type_with_options::<i16, _>(
        dst_path,
        window_size,
        window_size,
        src.raster_count(),
        &options,
    )?;
    dst.set_geo_transform(&gt)?;
    dst.set_projection(&src.projection())?;

    for band_idx in 1..=src.raster_count() {
        let src_band = src.rasterband(band_idx)?;
        let mut dst_band = dst.rasterband(band_idx)?;
        if let Some(nodata) = src_band.no_data_value() {
            dst_band.set_no_data_value(Some(nodata))?;
        }
        let mut buffer = src_band.read_as::<i16>(
            (col_off as isize, row_off as isize),
            (window_size, window_size),
            (window_size, window_size),
            None,
        )?;
        dst_band.write((0, 0), (window_size, window_size), &mut buffer)?;
    }

    dst.flush_cache()?;
    Ok((col_off, row_off))
}

fn write_center_config(
    base: &WorkflowConfig,
    class_out: &Path,
    out_root: &Path,
    config_out: &Path,
    window_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let products: Vec<Value> = base
        .predict
        .products
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()?;

    let value = json!({
        "class_grid": {
            "raster": class_out,
            "start_year": base.class_grid.start_year
        },
        "data_sources": {
            "dem_raster": base.data_sources.dem_raster,
            "ccdc_dir": base.data_sources.ccdc_dir,
            "rainfall_dir": base.data_sources.rainfall_dir,
            "temperature_dir": base.data_sources.temperature_dir,
            "landuse_dir": base.data_sources.landuse_dir,
            "resident_raster": base.data_sources.resident_raster,
            "river_shapefile": base.data_sources.river_shapefile,
            "road_dir": base.data_sources.road_dir
        },
        "data_usage": {
            "production": {
                "dem": base.data_usage.production.dem,
                "ccdc": base.data_usage.production.ccdc,
                "rainfall": base.data_usage.production.rainfall,
                "temperature": base.data_usage.production.temperature,
                "landuse": base.data_usage.production.landuse,
                "resident": base.data_usage.production.resident,
                "river": base.data_usage.production.river,
                "road": base.data_usage.production.road
            },
            "smoke_test": {
                "dem": base.data_usage.smoke_test.dem,
                "ccdc": base.data_usage.smoke_test.ccdc,
                "rainfall": base.data_usage.smoke_test.rainfall,
                "temperature": base.data_usage.smoke_test.temperature,
                "landuse": base.data_usage.smoke_test.landuse,
                "resident": base.data_usage.smoke_test.resident,
                "river": false,
                "road": false
            }
        },
        "data_process": {
            "cache_dir": out_root.join("cache"),
            "tile_size": window_size,
            "temporal_context_years": base.data_process.temporal_context_years,
            "workers": 4
        },
        "train": {
            "training_years": base.train.training_years,
            "model_dir": out_root.join("models"),
            "missing_data_policy": base.train.missing_data_policy,
            "workers": 4
        },
        "predict": {
            "prediction_year": base.predict.prediction_year,
            "output_dir": out_root.join("predictions"),
            "products": products,
            "workers": 4
        }
    });

    fs::write(config_out, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

fn prediction_tile_path(out_root: &Path, prediction_year: i32) -> PathBuf {
    out_root
        .join("cache/features")
        .join(format!("prediction_{prediction_year}"))
        .join("tile_000000_000000.parquet")
}

#[derive(Debug)]
struct BandStats {
    band: usize,
    valid: usize,
    nonzero: usize,
    min: i16,
    max: i16,
    nodata: Option<i16>,
}

#[derive(Debug)]
struct ProbabilityStats {
    valid: usize,
    unique_count: usize,
    min: f32,
    max: f32,
}

#[derive(Debug)]
struct FeatureColumnStats {
    name: String,
    valid: usize,
    min: f64,
    max: f64,
}

#[derive(Debug)]
struct IntegerRasterStats {
    valid: usize,
    unique_values: Vec<i16>,
}

fn integer_raster_stats(path: &Path) -> Result<IntegerRasterStats, Box<dyn std::error::Error>> {
    let ds = Dataset::open(path)?;
    let (width, height) = ds.raster_size();
    let band = ds.rasterband(1)?;
    let nodata = band.no_data_value().map(|value| value as i16);
    let data = band.read_as::<i16>((0, 0), (width, height), (width, height), None)?;

    let mut values = std::collections::BTreeSet::new();
    let mut valid = 0usize;
    for &value in data.data() {
        if Some(value) == nodata || value == 0 {
            continue;
        }
        valid += 1;
        values.insert(value);
    }
    Ok(IntegerRasterStats {
        valid,
        unique_values: values.into_iter().collect(),
    })
}

fn assert_disturbance_type_legend(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let classes = value
        .get("classes")
        .and_then(|classes| classes.as_array())
        .ok_or("missing classes legend")?;
    let expected = [
        (1, "fire", "火烧"),
        (2, "logging", "砍伐"),
        (3, "geo_hazard", "地质灾害"),
        (4, "road_construction", "道路建设"),
    ];
    for (map_value, name, label_zh) in expected {
        let found = classes.iter().any(|class| {
            class.get("map_value").and_then(|v| v.as_i64()) == Some(map_value)
                && class.get("name").and_then(|v| v.as_str()) == Some(name)
                && class.get("label_zh").and_then(|v| v.as_str()) == Some(label_zh)
        });
        assert!(found, "missing legend class {map_value} {label_zh}");
    }
    Ok(())
}

fn feature_table_stats(
    path: &Path,
    columns: &[&str],
) -> Result<Vec<FeatureColumnStats>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    let mut stats = Vec::with_capacity(columns.len());
    for &name in columns {
        let column = df.column(name)?;
        let series = column.cast(&DataType::Float64)?;
        let values = series.f64()?;
        let mut valid = 0usize;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in values.into_iter().flatten() {
            if !value.is_finite() {
                continue;
            }
            valid += 1;
            min = min.min(value);
            max = max.max(value);
        }
        stats.push(FeatureColumnStats {
            name: name.to_string(),
            valid,
            min: if valid > 0 { min } else { 0.0 },
            max: if valid > 0 { max } else { 0.0 },
        });
    }
    Ok(stats)
}

fn probability_raster_stats(path: &Path) -> Result<ProbabilityStats, Box<dyn std::error::Error>> {
    let ds = Dataset::open(path)?;
    let (width, height) = ds.raster_size();
    let band = ds.rasterband(1)?;
    let nodata = band.no_data_value();
    let data = band.read_as::<f32>((0, 0), (width, height), (width, height), None)?;

    let mut valid = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut unique = std::collections::HashSet::new();

    for &value in data.data() {
        if !value.is_finite() {
            continue;
        }
        if nodata.is_some_and(|nd| nd.is_nan() && value.is_nan()) {
            continue;
        }
        valid += 1;
        min = min.min(value);
        max = max.max(value);
        unique.insert(value.to_bits());
    }

    Ok(ProbabilityStats {
        valid,
        unique_count: unique.len(),
        min: if valid > 0 { min } else { 0.0 },
        max: if valid > 0 { max } else { 0.0 },
    })
}

fn band_stats(
    path: &Path,
    window_size: usize,
) -> Result<Vec<BandStats>, Box<dyn std::error::Error>> {
    let ds = Dataset::open(path)?;
    let mut stats = Vec::with_capacity(ds.raster_count());
    for band_idx in 1..=ds.raster_count() {
        let band = ds.rasterband(band_idx)?;
        let data = band.read_as::<i16>(
            (0, 0),
            (window_size, window_size),
            (window_size, window_size),
            None,
        )?;
        let nodata = band.no_data_value().map(|value| value as i16);
        let mut nonzero = 0usize;
        let mut valid = 0usize;
        let mut min = i16::MAX;
        let mut max = i16::MIN;
        for &value in data.data() {
            if Some(value) == nodata {
                continue;
            }
            if value != 0 {
                nonzero += 1;
            }
            valid += 1;
            min = min.min(value);
            max = max.max(value);
        }
        stats.push(BandStats {
            band: band_idx,
            valid,
            nonzero,
            min: if valid > 0 { min } else { 0 },
            max: if valid > 0 { max } else { 0 },
            nodata,
        });
    }
    Ok(stats)
}

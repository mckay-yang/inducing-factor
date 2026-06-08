use std::path::Path;
use std::time::Instant;

use data_process::{
    run_class_basin_data_process, DataProcessPlan, DataProcessRequest, DataProcessSources,
    DataSourceUsage,
};
use gdal::GeoGrid;
use predict::{BasinPredictRequest, PredictPlan, PredictionProduct};
use tracing::error;
use train::{BasinTrainSettings, ModelMissingDataPolicy, TrainPlan};
use workflow_config::{OutputProduct, WorkflowConfig};

fn setup_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cli=info".into()),
        )
        .with_target(false)
        .try_init()
        .ok();
}

fn load_workflow_config(path: &Path) -> Option<WorkflowConfig> {
    match WorkflowConfig::from_json_file(path) {
        Ok(config) => Some(config),
        Err(e) => {
            error!("Failed to load workflow config {:?}: {}", path, e);
            None
        }
    }
}

fn data_process_request_from_workflow(config: &WorkflowConfig) -> DataProcessRequest {
    DataProcessRequest {
        class_raster: config.class_grid.raster.clone(),
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
        source_usage: data_source_usage_from_workflow(&config.data_usage.production),
        cache_dir: config.data_process.cache_dir.clone(),
        tile_size: config.data_process.tile_size,
        processing_years: config.processing_years(),
        training_years: config.train.training_years.clone(),
        prediction_year: config.predict.prediction_year,
        workers: config.data_process.workers,
    }
}

fn data_source_usage_from_workflow(usage: &workflow_config::DataUsageConfig) -> DataSourceUsage {
    DataSourceUsage {
        dem: usage.dem,
        ccdc: usage.ccdc,
        rainfall: usage.rainfall,
        temperature: usage.temperature,
        landuse: usage.landuse,
        resident: usage.resident,
        river: usage.river,
        road: usage.road,
    }
}

fn train_plan_from_workflow(config: &WorkflowConfig) -> TrainPlan {
    TrainPlan {
        training_years: config.train.training_years.clone(),
        feature_path: config
            .data_process
            .cache_dir
            .join("features")
            .join("training_features.parquet"),
        model_path: config.train.model_dir.join("basin_model.json"),
        missing_data_policy: train_missing_policy(&config.train.missing_data_policy),
        workers: config.train.workers,
    }
}

fn train_settings_from_workflow(config: &WorkflowConfig) -> BasinTrainSettings {
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
        missing_data_policy: train_missing_policy(&config.train.missing_data_policy),
        workers: config.train.workers,
    }
}

fn train_missing_policy(
    policy: &workflow_config::ModelMissingDataPolicy,
) -> ModelMissingDataPolicy {
    match policy {
        workflow_config::ModelMissingDataPolicy::NativeMissingCategory => {
            ModelMissingDataPolicy::NativeMissingCategory
        }
        workflow_config::ModelMissingDataPolicy::FeatureDropout => {
            ModelMissingDataPolicy::FeatureDropout
        }
    }
}

fn predict_plan_from_workflow(config: &WorkflowConfig) -> PredictPlan {
    PredictPlan {
        prediction_year: config.predict.prediction_year,
        model_path: config.train.model_dir.join("basin_model.json"),
        feature_dir: config
            .data_process
            .cache_dir
            .join("features")
            .join(format!("prediction_{}", config.predict.prediction_year)),
        output_dir: config.predict.output_dir.clone(),
        products: config
            .predict
            .products
            .iter()
            .map(prediction_product)
            .collect(),
        workers: config.predict.workers,
    }
}

fn predict_request_from_workflow(config: &WorkflowConfig) -> BasinPredictRequest {
    let plan = predict_plan_from_workflow(config);
    BasinPredictRequest {
        class_raster: config.class_grid.raster.clone(),
        class_start_year: config.class_grid.start_year,
        model_path: plan.model_path,
        feature_dir: plan.feature_dir,
        output_dir: plan.output_dir,
        prediction_year: plan.prediction_year,
        products: plan.products,
        workers: plan.workers,
    }
}

fn prediction_product(product: &OutputProduct) -> PredictionProduct {
    match product {
        OutputProduct::Probability => PredictionProduct::Probability,
        OutputProduct::DominantFactor => PredictionProduct::DominantFactor,
        OutputProduct::DisturbanceType => PredictionProduct::DisturbanceType,
        OutputProduct::DisturbanceTypeProbability => PredictionProduct::DisturbanceTypeProbability,
        OutputProduct::FactorActivation => PredictionProduct::FactorActivation,
        OutputProduct::MissingDataMask => PredictionProduct::MissingDataMask,
    }
}

pub fn run_plan_workflow(config_path: &Path) {
    setup_logging();

    let Some(config) = load_workflow_config(config_path) else {
        return;
    };

    let class_grid = match GeoGrid::open(&config.class_grid.raster, config.class_grid.start_year) {
        Ok(grid) => grid,
        Err(e) => {
            error!(
                "Failed to open CLASS grid {:?}: {}",
                config.class_grid.raster, e
            );
            return;
        }
    };

    let data_request = data_process_request_from_workflow(&config);
    let data_plan = DataProcessPlan::from_request(&data_request, class_grid);
    let train_plan = train_plan_from_workflow(&config);
    let predict_plan = predict_plan_from_workflow(&config);

    eprintln!(
        "[workflow] class grid: {}x{}, years {:?}",
        data_plan.class_grid.width,
        data_plan.class_grid.height,
        data_plan.class_grid.years()
    );
    eprintln!(
        "[workflow] data_process years: {:?}, tiles: {}, tile_size: {}, workers: {}",
        data_plan.processing_years,
        data_plan.tiles.len(),
        data_plan.tile_size,
        data_plan.workers
    );
    eprintln!(
        "[workflow] train years: {:?}, features: {}, model: {}, missing_policy: {:?}, workers: {}",
        train_plan.training_years,
        train_plan.feature_path.display(),
        train_plan.model_path.display(),
        train_plan.missing_data_policy,
        train_plan.workers
    );
    eprintln!(
        "[workflow] predict year: {}, feature_dir: {}, output_dir: {}, products: {:?}, workers: {}",
        predict_plan.prediction_year,
        predict_plan.feature_dir.display(),
        predict_plan.output_dir.display(),
        predict_plan.products,
        predict_plan.workers
    );
}

pub fn run_data_process_workflow(config_path: &Path) {
    setup_logging();

    let Some(config) = load_workflow_config(config_path) else {
        return;
    };

    let request = data_process_request_from_workflow(&config);
    match run_class_basin_data_process(&request) {
        Ok(report) => {
            eprintln!(
                "[data-process] complete: class={}x{}, processing_years={:?}, training_years={:?}, prediction_year={}, prediction_tiles={}, cache={}",
                report.class_width,
                report.class_height,
                report.processing_years,
                report.training_years,
                report.prediction_year,
                report.prediction_tiles,
                report.cache_dir.display()
            );
        }
        Err(e) => {
            error!("[data-process] failed: {}", e);
        }
    }
}

pub fn run_train_workflow(config_path: &Path) {
    setup_logging();
    let Some(config) = load_workflow_config(config_path) else {
        return;
    };
    let start_time = Instant::now();

    let train_plan = train_plan_from_workflow(&config);
    let settings = train_settings_from_workflow(&config);
    match train::train_basin_with_settings(
        &train_plan.feature_path,
        &config.train.model_dir,
        &settings,
    ) {
        Ok(report) => {
            eprintln!(
                "[train-workflow] complete: model={}, rows={}, events={}, background={}, factors={}, workers={}, elapsed={:.2?}",
                report.model_path.display(),
                report.n_rows,
                report.n_events,
                report.n_background,
                report.n_factors,
                report.workers,
                start_time.elapsed()
            );
        }
        Err(e) => {
            error!("[train-workflow] failed: {}", e);
        }
    }
}

pub fn run_predict_workflow(config_path: &Path) {
    setup_logging();
    let Some(config) = load_workflow_config(config_path) else {
        return;
    };
    let start_time = Instant::now();
    let request = predict_request_from_workflow(&config);

    match predict::predict_basin(&request) {
        Ok(report) => {
            eprintln!(
                "[predict-workflow] complete: tiles={}, pixels={}, workers={}, probability={:?}, dominant_factor={:?}, disturbance_type={:?}, type_probabilities={:?}, missing_mask={:?}, elapsed={:.2?}",
                report.tiles_processed,
                report.pixels_processed,
                report.workers,
                report.probability_path,
                report.dominant_factor_path,
                report.disturbance_type_path,
                report.disturbance_type_probability_paths,
                report.missing_data_mask_path,
                start_time.elapsed()
            );
        }
        Err(e) => {
            error!("[predict-workflow] failed: {}", e);
        }
    }
}

pub fn run_workflow(config_path: &Path, skip_data_process: bool) {
    setup_logging();
    let start_time = Instant::now();
    let Some(config) = load_workflow_config(config_path) else {
        return;
    };

    if !skip_data_process {
        let request = data_process_request_from_workflow(&config);
        match run_class_basin_data_process(&request) {
            Ok(report) => {
                eprintln!(
                    "[run-workflow][data-process] complete: prediction_tiles={}, cache={}",
                    report.prediction_tiles,
                    report.cache_dir.display()
                );
            }
            Err(e) => {
                error!("[run-workflow][data-process] failed: {}", e);
                return;
            }
        }
    }

    let train_plan = train_plan_from_workflow(&config);
    let settings = train_settings_from_workflow(&config);
    match train::train_basin_with_settings(
        &train_plan.feature_path,
        &config.train.model_dir,
        &settings,
    ) {
        Ok(report) => {
            eprintln!(
                "[run-workflow][train] complete: model={}, rows={}, factors={}, workers={}",
                report.model_path.display(),
                report.n_rows,
                report.n_factors,
                report.workers
            );
        }
        Err(e) => {
            error!("[run-workflow][train] failed: {}", e);
            return;
        }
    }

    let request = predict_request_from_workflow(&config);
    match predict::predict_basin(&request) {
        Ok(report) => {
            eprintln!(
                "[run-workflow][predict] complete: tiles={}, pixels={}, workers={}, probability={:?}, dominant_factor={:?}, disturbance_type={:?}, type_probabilities={:?}, missing_mask={:?}",
                report.tiles_processed,
                report.pixels_processed,
                report.workers,
                report.probability_path,
                report.dominant_factor_path,
                report.disturbance_type_path,
                report.disturbance_type_probability_paths,
                report.missing_data_mask_path
            );
        }
        Err(e) => {
            error!("[run-workflow][predict] failed: {}", e);
            return;
        }
    }

    eprintln!(
        "[run-workflow] complete: config={}, elapsed={:.2?}",
        config_path.display(),
        start_time.elapsed()
    );
}

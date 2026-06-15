//! Prediction stage boundary.
//!
//! Prediction reads model artifacts through `model` and writes one or more
//! geospatial products.  Additional products such as dominant factor maps are
//! represented in config before their concrete raster writers are added.

use gdal::GeoGrid;
use gdal_crate::raster::{Buffer, RasterCreationOptions};
use gdal_crate::{Dataset, DriverManager};
use model::BasinModelArtifact;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PredictionProduct {
    Probability,
    DominantFactor,
    DisturbanceType,
    DisturbanceTypeProbability,
    FactorActivation,
    MissingDataMask,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictPlan {
    pub prediction_year: i32,
    pub model_path: PathBuf,
    pub feature_dir: PathBuf,
    pub output_dir: PathBuf,
    pub products: Vec<PredictionProduct>,
    pub workers: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum PredictError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("GDAL error: {0}")]
    Gdal(String),
    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("model artifact error: {0}")]
    Model(#[from] model::ModelArtifactError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("thread pool error: {0}")]
    ThreadPool(String),
    #[error("no parquet tile files found in {0}")]
    NoTiles(PathBuf),
    #[error("missing required column {0}")]
    MissingColumn(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BasinPredictRequest {
    pub class_raster: PathBuf,
    pub class_start_year: i32,
    pub model_path: PathBuf,
    pub feature_dir: PathBuf,
    pub output_dir: PathBuf,
    pub prediction_year: i32,
    pub products: Vec<PredictionProduct>,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BasinPredictReport {
    pub probability_path: Option<PathBuf>,
    pub dominant_factor_path: Option<PathBuf>,
    pub disturbance_type_path: Option<PathBuf>,
    pub disturbance_type_probability_paths: Vec<PathBuf>,
    pub disturbance_type_dominant_factor_paths: Vec<PathBuf>,
    pub missing_data_mask_path: Option<PathBuf>,
    pub tiles_processed: usize,
    pub pixels_processed: usize,
    pub workers: usize,
}

pub fn predict_basin(request: &BasinPredictRequest) -> Result<BasinPredictReport, PredictError> {
    fs::create_dir_all(&request.output_dir)?;

    let model = BasinModelArtifact::read_json(&request.model_path)?;
    let class_grid = GeoGrid::open(&request.class_raster, request.class_start_year)
        .map_err(|e| PredictError::Gdal(e.to_string()))?;
    status!(
        "[predict] start: model={}, feature_dir={}, output_dir={}, year={}",
        request.model_path.display(),
        request.feature_dir.display(),
        request.output_dir.display(),
        request.prediction_year
    );

    let mut tile_paths: Vec<PathBuf> = fs::read_dir(&request.feature_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().is_some_and(|ext| ext == "parquet") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    tile_paths.sort();
    if tile_paths.is_empty() {
        return Err(PredictError::NoTiles(request.feature_dir.clone()));
    }
    status!("[predict] feature tiles discovered: {}", tile_paths.len());

    let wants_probability =
        request.products.contains(&PredictionProduct::Probability) || request.products.is_empty();
    let wants_dominant = request
        .products
        .contains(&PredictionProduct::DominantFactor);
    let requested_disturbance_type = request
        .products
        .contains(&PredictionProduct::DisturbanceType);
    let requested_type_probability = request
        .products
        .contains(&PredictionProduct::DisturbanceTypeProbability);
    let has_type_model = model
        .class_engine
        .as_ref()
        .map(|engine| !engine.classes.is_empty())
        .unwrap_or(false);
    if (requested_disturbance_type || requested_type_probability) && !has_type_model {
        status!(
            "[predict] disturbance type products requested but no trained class_engine is present; skip type rasters instead of writing empty placeholders"
        );
    }
    let wants_disturbance_type = requested_disturbance_type && has_type_model;
    let wants_type_probability = requested_type_probability && has_type_model;
    let wants_missing = request
        .products
        .contains(&PredictionProduct::MissingDataMask);

    let probability_path = wants_probability.then(|| {
        request
            .output_dir
            .join(format!("probability_{}.tif", request.prediction_year))
    });
    let dominant_factor_path = wants_dominant.then(|| {
        request
            .output_dir
            .join(format!("dominant_factor_{}.tif", request.prediction_year))
    });
    let disturbance_type_path = wants_disturbance_type.then(|| {
        request
            .output_dir
            .join(format!("disturbance_type_{}.tif", request.prediction_year))
    });
    let disturbance_type_probability_paths: Vec<PathBuf> = if wants_type_probability {
        disturbance_type_specs()
            .iter()
            .map(|spec| {
                request.output_dir.join(format!(
                    "disturbance_type_{}_probability_{}.tif",
                    spec.name, request.prediction_year
                ))
            })
            .collect()
    } else {
        Vec::new()
    };
    let disturbance_type_dominant_factor_paths: Vec<PathBuf> =
        if wants_type_probability && wants_dominant {
            disturbance_type_specs()
                .iter()
                .map(|spec| {
                    request.output_dir.join(format!(
                        "disturbance_type_{}_dominant_factor_{}.tif",
                        spec.name, request.prediction_year
                    ))
                })
                .collect()
        } else {
            Vec::new()
        };
    let missing_data_mask_path = wants_missing.then(|| {
        request
            .output_dir
            .join(format!("missing_data_mask_{}.tif", request.prediction_year))
    });
    let requested_outputs = requested_output_paths(
        probability_path.as_ref(),
        dominant_factor_path.as_ref(),
        disturbance_type_path.as_ref(),
        &disturbance_type_probability_paths,
        &disturbance_type_dominant_factor_paths,
        missing_data_mask_path.as_ref(),
    );
    let workers = effective_workers(request.workers);
    if std::env::var("PREDICT_SKIP_EXISTING_OUTPUTS")
        .ok()
        .as_deref()
        == Some("1")
        && !requested_outputs.is_empty()
        && requested_outputs.iter().all(|path| file_is_nonempty(path))
    {
        status!(
            "[predict] all requested outputs exist, skip: {} files",
            requested_outputs.len()
        );
        return Ok(BasinPredictReport {
            probability_path,
            dominant_factor_path,
            disturbance_type_path,
            disturbance_type_probability_paths,
            disturbance_type_dominant_factor_paths,
            missing_data_mask_path,
            tiles_processed: 0,
            pixels_processed: 0,
            workers,
        });
    }
    status!(
        "[predict] products: probability={}, dominant_factor={}, disturbance_type={}, type_probability={}, missing_mask={}",
        wants_probability,
        wants_dominant,
        wants_disturbance_type,
        wants_type_probability,
        wants_missing
    );

    let mut probability_ds = match &probability_path {
        Some(path) => Some(create_tiff::<f32>(path, &class_grid, Some(f64::NAN))?),
        None => None,
    };
    let mut dominant_ds = match &dominant_factor_path {
        Some(path) => Some(create_tiff::<i16>(path, &class_grid, Some(0.0))?),
        None => None,
    };
    let mut disturbance_type_ds = match &disturbance_type_path {
        Some(path) => Some(create_tiff::<i16>(path, &class_grid, Some(0.0))?),
        None => None,
    };
    let mut disturbance_type_probability_ds: Vec<Dataset> = disturbance_type_probability_paths
        .iter()
        .map(|path| create_tiff::<f32>(path, &class_grid, Some(f64::NAN)))
        .collect::<Result<_, _>>()?;
    let mut disturbance_type_dominant_ds: Vec<Dataset> = disturbance_type_dominant_factor_paths
        .iter()
        .map(|path| create_tiff::<i16>(path, &class_grid, Some(0.0)))
        .collect::<Result<_, _>>()?;
    let mut missing_ds = match &missing_data_mask_path {
        Some(path) => Some(create_tiff::<u8>(path, &class_grid, Some(255.0))?),
        None => None,
    };

    if wants_dominant {
        write_factor_legend(
            &request.output_dir.join(format!(
                "dominant_factor_{}_legend.json",
                request.prediction_year
            )),
            &model,
        )?;
        if let Some(model_dir) = request.model_path.parent() {
            write_factor_legend(&model_dir.join("dominant_factor_mapping.json"), &model)?;
        }
    }
    if wants_disturbance_type || wants_type_probability {
        write_disturbance_type_legend(&request.output_dir.join(format!(
            "disturbance_type_{}_legend.json",
            request.prediction_year
        )))?;
    }

    let metadata_cols = ["row", "col"];
    let pred_feature_cols: Vec<String> = model
        .feature_columns
        .iter()
        .filter(|c| !metadata_cols.contains(&c.as_str()))
        .cloned()
        .collect();

    let mut tiles_processed = 0usize;
    let mut pixels_processed = 0usize;
    let batch_size = workers.saturating_mul(2).max(1);
    let total_batches = (tile_paths.len() + batch_size - 1) / batch_size;
    status!(
        "[predict] processing tiles: workers={}, batch_size={}, batches={}",
        workers,
        batch_size,
        total_batches
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| PredictError::ThreadPool(e.to_string()))?;

    for (batch_idx, batch) in tile_paths.chunks(batch_size).enumerate() {
        let batch_tiles: Vec<Result<Option<TilePredictions>, PredictError>> = pool.install(|| {
            batch
                .par_iter()
                .map(|tile_path| process_tile_path(tile_path, &pred_feature_cols, &model))
                .collect()
        });

        for tile in batch_tiles {
            let Some(tile) = tile? else {
                continue;
            };
            pixels_processed += tile.rows.len();

            if let Some(ref mut ds) = probability_ds {
                write_tile_window(ds, &tile, &tile.probabilities)?;
            }
            if let Some(ref mut ds) = dominant_ds {
                write_tile_window(ds, &tile, &tile.dominant_factors)?;
            }
            if let Some(ref mut ds) = disturbance_type_ds {
                write_tile_window(ds, &tile, &tile.disturbance_types)?;
            }
            for (idx, ds) in disturbance_type_probability_ds.iter_mut().enumerate() {
                write_tile_window(ds, &tile, &tile.disturbance_type_probabilities[idx])?;
            }
            for (idx, ds) in disturbance_type_dominant_ds.iter_mut().enumerate() {
                write_tile_window(ds, &tile, &tile.disturbance_type_dominant_factors[idx])?;
            }
            if let Some(ref mut ds) = missing_ds {
                write_tile_window(ds, &tile, &tile.missing_mask)?;
            }

            tiles_processed += 1;
        }
        if tiles_processed == 1 || (batch_idx + 1) % 10 == 0 || batch_idx + 1 == total_batches {
            status!(
                "[predict] progress: batch {}/{} ({:.1}%), tiles={}, pixels={}",
                batch_idx + 1,
                total_batches,
                if total_batches > 0 {
                    100.0 * (batch_idx + 1) as f64 / total_batches as f64
                } else {
                    100.0
                },
                tiles_processed,
                pixels_processed
            );
        }
    }

    drop(probability_ds);
    drop(dominant_ds);
    drop(disturbance_type_ds);
    drop(disturbance_type_probability_ds);
    drop(disturbance_type_dominant_ds);
    drop(missing_ds);

    status!(
        "[predict] complete: tiles={}, pixels={}, output_dir={}",
        tiles_processed,
        pixels_processed,
        request.output_dir.display()
    );

    Ok(BasinPredictReport {
        probability_path,
        dominant_factor_path,
        disturbance_type_path,
        disturbance_type_probability_paths,
        disturbance_type_dominant_factor_paths,
        missing_data_mask_path,
        tiles_processed,
        pixels_processed,
        workers,
    })
}

fn requested_output_paths<'a>(
    probability_path: Option<&'a PathBuf>,
    dominant_factor_path: Option<&'a PathBuf>,
    disturbance_type_path: Option<&'a PathBuf>,
    disturbance_type_probability_paths: &'a [PathBuf],
    disturbance_type_dominant_factor_paths: &'a [PathBuf],
    missing_data_mask_path: Option<&'a PathBuf>,
) -> Vec<&'a PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = probability_path {
        paths.push(path);
    }
    if let Some(path) = dominant_factor_path {
        paths.push(path);
    }
    if let Some(path) = disturbance_type_path {
        paths.push(path);
    }
    paths.extend(disturbance_type_probability_paths.iter());
    paths.extend(disturbance_type_dominant_factor_paths.iter());
    if let Some(path) = missing_data_mask_path {
        paths.push(path);
    }
    paths
}

fn file_is_nonempty(path: &Path) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
}

fn process_tile_path(
    tile_path: &Path,
    pred_feature_cols: &[String],
    model: &BasinModelArtifact,
) -> Result<Option<TilePredictions>, PredictError> {
    let file = fs::File::open(tile_path)?;
    let df = ParquetReader::new(file).finish()?;
    if df.height() == 0 {
        return Ok(None);
    }
    prepare_tile_predictions(&df, pred_feature_cols, model).map(Some)
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

#[derive(Debug)]
struct TilePredictions {
    row_off: usize,
    col_off: usize,
    width: usize,
    height: usize,
    rows: Vec<usize>,
    probabilities: Vec<f32>,
    dominant_factors: Vec<i16>,
    disturbance_types: Vec<i16>,
    disturbance_type_probabilities: [Vec<f32>; 4],
    disturbance_type_dominant_factors: [Vec<i16>; 4],
    missing_mask: Vec<u8>,
}

fn prepare_tile_predictions(
    df: &DataFrame,
    pred_feature_cols: &[String],
    model: &BasinModelArtifact,
) -> Result<TilePredictions, PredictError> {
    let rows_series = df
        .column("row")
        .map_err(|_| PredictError::MissingColumn("row".into()))?
        .i32()?;
    let cols_series = df
        .column("col")
        .map_err(|_| PredictError::MissingColumn("col".into()))?
        .i32()?;

    let mut rows = Vec::with_capacity(df.height());
    let mut cols = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        let row = rows_series
            .get(i)
            .ok_or_else(|| PredictError::MissingColumn("row contains null".into()))?;
        let col = cols_series
            .get(i)
            .ok_or_else(|| PredictError::MissingColumn("col contains null".into()))?;
        rows.push(row as usize);
        cols.push(col as usize);
    }

    let row_off = *rows.iter().min().unwrap_or(&0);
    let col_off = *cols.iter().min().unwrap_or(&0);
    let row_max = *rows.iter().max().unwrap_or(&row_off);
    let col_max = *cols.iter().max().unwrap_or(&col_off);
    let width = col_max - col_off + 1;
    let height = row_max - row_off + 1;
    let n_cells = width * height;

    let feature_columns_data: Vec<Vec<f64>> = pred_feature_cols
        .iter()
        .map(|name| collect_numeric_feature_column(&df, name, df.height()))
        .collect();
    let mut feature_data: Vec<Vec<f64>> = Vec::with_capacity(df.height());
    let mut missing_by_row: Vec<bool> = Vec::with_capacity(df.height());
    for row_idx in 0..df.height() {
        let mut row = Vec::with_capacity(pred_feature_cols.len());
        let mut missing = false;
        for column in &feature_columns_data {
            let val = column[row_idx];
            if !val.is_finite() {
                missing = true;
            }
            row.push(val);
        }
        feature_data.push(row);
        missing_by_row.push(missing);
    }

    let mut probability_tile = vec![f32::NAN; n_cells];
    let mut dominant_tile = vec![0i16; n_cells];
    let mut disturbance_type_tile = vec![0i16; n_cells];
    let mut disturbance_type_probability_tiles: [Vec<f32>; 4] =
        std::array::from_fn(|_| vec![f32::NAN; n_cells]);
    let mut disturbance_type_dominant_tiles: [Vec<i16>; 4] =
        std::array::from_fn(|_| vec![0i16; n_cells]);
    let mut missing_tile = vec![255u8; n_cells];

    if model.occurrence_engine.is_numeric_backend() {
        let (probabilities, dominant_factors) = model
            .occurrence_engine
            .predict_numeric_rows_with_dominant(&feature_data);
        let class_outputs = model
            .class_engine
            .as_ref()
            .filter(|engine| engine.is_numeric_backend())
            .map(|engine| engine.predict_numeric_proba_with_dominant(&feature_data));
        for i in 0..df.height() {
            let local_row = rows[i] - row_off;
            let local_col = cols[i] - col_off;
            let idx = local_row * width + local_col;
            let disturbance_probability = probabilities.get(i).copied().unwrap_or(f64::NAN);
            probability_tile[idx] = disturbance_probability as f32;
            dominant_tile[idx] = dominant_factors.get(i).copied().unwrap_or(0);
            if let Some((class_probabilities, type_dominant)) = class_outputs.as_ref() {
                let (class_value, joint_probabilities) = numeric_disturbance_type_prediction(
                    class_probabilities.get(i).map(Vec::as_slice).unwrap_or(&[]),
                    disturbance_probability,
                );
                disturbance_type_tile[idx] = class_value;
                for class_idx in 0..4 {
                    disturbance_type_probability_tiles[class_idx][idx] =
                        joint_probabilities[class_idx];
                    disturbance_type_dominant_tiles[class_idx][idx] =
                        type_dominant.get(i).map(|row| row[class_idx]).unwrap_or(0);
                }
            } else {
                disturbance_type_tile[idx] = 0;
                for class_idx in 0..4 {
                    disturbance_type_probability_tiles[class_idx][idx] = f32::NAN;
                    disturbance_type_dominant_tiles[class_idx][idx] = 0;
                }
            }
            missing_tile[idx] = if missing_by_row[i] { 1 } else { 0 };
        }
    } else {
        let labels = model
            .discretizer
            .transform(pred_feature_cols, &feature_data, true);
        let label_matrix = LabelMatrix::from_labels(pred_feature_cols, &labels);
        let compiled_factors =
            CompiledFactors::compile(pred_feature_cols, &label_matrix, &model.selected_factors);

        for i in 0..df.height() {
            let local_row = rows[i] - row_off;
            let local_col = cols[i] - col_off;
            let idx = local_row * width + local_col;
            let activations = compiled_factors.activations_for_row(&label_matrix, i);
            probability_tile[idx] = occurrence_probability(
                &activations,
                &model.occurrence_engine.weights,
                model.occurrence_engine.intercept,
            ) as f32;
            dominant_tile[idx] = dominant_factor_index_from_activations(
                &activations,
                &model.occurrence_engine.weights,
                &model.selected_factors,
            );
            let (class_value, class_probabilities) =
                disturbance_type_prediction_from_activations(&activations, model);
            disturbance_type_tile[idx] = class_value;
            for class_idx in 0..4 {
                disturbance_type_probability_tiles[class_idx][idx] = class_probabilities[class_idx];
                disturbance_type_dominant_tiles[class_idx][idx] = 0;
            }
            missing_tile[idx] = if missing_by_row[i] { 1 } else { 0 };
        }
    }

    Ok(TilePredictions {
        row_off,
        col_off,
        width,
        height,
        rows,
        probabilities: probability_tile,
        dominant_factors: dominant_tile,
        disturbance_types: disturbance_type_tile,
        disturbance_type_probabilities: disturbance_type_probability_tiles,
        disturbance_type_dominant_factors: disturbance_type_dominant_tiles,
        missing_mask: missing_tile,
    })
}

fn numeric_disturbance_type_prediction(
    conditional_probabilities: &[(i64, f64)],
    disturbance_probability: f64,
) -> (i16, [f32; 4]) {
    let mut probabilities = [f32::NAN; 4];
    let mut best_class = 0i16;
    let mut best_probability = f64::NEG_INFINITY;
    for &(class_value, conditional_probability) in conditional_probabilities {
        if (1..=4).contains(&class_value) {
            let joint = disturbance_probability * conditional_probability;
            probabilities[class_value as usize - 1] = joint as f32;
            if joint > best_probability {
                best_probability = joint;
                best_class = class_value as i16;
            }
        }
    }
    (best_class, probabilities)
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

struct LabelMatrix {
    codes: Vec<Vec<u16>>,
    dictionaries: Vec<Vec<String>>,
}

impl LabelMatrix {
    fn from_labels(feature_names: &[String], labels: &[Vec<String>]) -> Self {
        let n_features = feature_names.len();
        let mut dictionaries: Vec<Vec<String>> = vec![Vec::new(); n_features];
        let mut codes: Vec<Vec<u16>> = vec![vec![0; n_features]; labels.len()];

        for feature_idx in 0..n_features {
            let mut values = Vec::<String>::new();
            for (row_idx, row) in labels.iter().enumerate() {
                let value = row
                    .get(feature_idx)
                    .cloned()
                    .unwrap_or_else(|| "missing".to_string());
                let code = match values.iter().position(|existing| existing == &value) {
                    Some(idx) => idx as u16,
                    None => {
                        values.push(value);
                        (values.len() - 1) as u16
                    }
                };
                codes[row_idx][feature_idx] = code;
            }
            dictionaries[feature_idx] = values;
        }

        Self {
            codes,
            dictionaries,
        }
    }
}

#[derive(Debug)]
struct CompiledFactor {
    items: Vec<(usize, u16)>,
}

struct CompiledFactors {
    factors: Vec<CompiledFactor>,
}

impl CompiledFactors {
    fn compile(
        feature_names: &[String],
        labels: &LabelMatrix,
        factors: &[factor_core::types::InducingFactor],
    ) -> Self {
        let mut compiled = Vec::with_capacity(factors.len());
        for factor in factors {
            let mut items = Vec::with_capacity(factor.items.len());
            for item in &factor.items {
                if let Some((feature, state)) = item.split_once('=') {
                    if let Some(feature_idx) = feature_names.iter().position(|name| name == feature)
                    {
                        let state_code = labels.dictionaries[feature_idx]
                            .iter()
                            .position(|value| value == state)
                            .map(|idx| idx as u16)
                            .unwrap_or(u16::MAX);
                        items.push((feature_idx, state_code));
                    }
                }
            }
            compiled.push(CompiledFactor { items });
        }
        Self { factors: compiled }
    }

    fn activations_for_row(&self, labels: &LabelMatrix, row_idx: usize) -> Vec<i32> {
        let mut activations = Vec::with_capacity(self.factors.len());
        for factor in &self.factors {
            let mut active = true;
            for &(feature_idx, state_code) in &factor.items {
                if labels
                    .codes
                    .get(row_idx)
                    .and_then(|row| row.get(feature_idx))
                    .copied()
                    != Some(state_code)
                {
                    active = false;
                    break;
                }
            }
            activations.push(if active { 1 } else { 0 });
        }
        activations
    }
}

fn disturbance_type_prediction_from_activations(
    activations: &[i32],
    model: &BasinModelArtifact,
) -> (i16, [f32; 4]) {
    let Some(engine) = model.class_engine.as_ref() else {
        return (0, [f32::NAN; 4]);
    };
    if engine.classes.is_empty() {
        return (0, [f32::NAN; 4]);
    }

    if engine.classes.len() == 1 {
        let class_value = engine.classes[0];
        let mut probabilities = [f32::NAN; 4];
        if (1..=4).contains(&class_value) {
            probabilities[class_value as usize - 1] = 1.0;
            return (class_value as i16, probabilities);
        }
        return (0, probabilities);
    }

    let mut logits = Vec::with_capacity(engine.classes.len());
    for class_idx in 0..engine.classes.len() {
        let mut logit = engine.intercepts.get(class_idx).copied().unwrap_or(0.0);
        if let Some(weights) = engine.weights.get(class_idx) {
            for (idx, &active) in activations.iter().enumerate() {
                logit += weights.get(idx).copied().unwrap_or(0.0) * active as f64;
            }
        }
        logits.push(logit);
    }

    let probabilities_by_class = softmax_probabilities(&logits);
    disturbance_type_prediction(&engine.classes, &probabilities_by_class)
}

fn disturbance_type_prediction(classes: &[i64], class_probs: &[f64]) -> (i16, [f32; 4]) {
    let mut probabilities = [f32::NAN; 4];
    let mut best_class = 0i16;
    let mut best_prob = f64::NEG_INFINITY;
    for (&class_value, &probability) in classes.iter().zip(class_probs.iter()) {
        if (1..=4).contains(&class_value) {
            probabilities[class_value as usize - 1] = probability as f32;
            if probability > best_prob {
                best_prob = probability;
                best_class = class_value as i16;
            }
        }
    }
    (best_class, probabilities)
}

fn occurrence_probability(activations: &[i32], weights: &[f64], intercept: f64) -> f64 {
    let mut logit = intercept;
    for (idx, &active) in activations.iter().enumerate() {
        logit += weights.get(idx).copied().unwrap_or(0.0) * active as f64;
    }
    sigmoid(logit.clamp(-30.0, 30.0))
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let neg = (-value).exp();
        1.0 / (1.0 + neg)
    } else {
        let pos = value.exp();
        pos / (1.0 + pos)
    }
}

fn softmax_probabilities(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_values = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &logit in logits {
        let value = (logit - max).exp();
        exp_values.push(value);
        sum += value;
    }
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exp_values.into_iter().map(|value| value / sum).collect()
}

fn dominant_factor_index_from_activations(
    activations: &[i32],
    weights: &[f64],
    factors: &[factor_core::types::InducingFactor],
) -> i16 {
    let mut best_idx: Option<usize> = None;
    let mut best_weight = f64::NEG_INFINITY;
    for (idx, active) in activations.iter().enumerate() {
        if *active == 0 {
            continue;
        }
        let weight = weights
            .get(idx)
            .copied()
            .unwrap_or_else(|| factors.get(idx).map(|f| f.score).unwrap_or(0.0))
            .abs();
        if weight > best_weight {
            best_weight = weight;
            best_idx = Some(idx);
        }
    }
    best_idx.map(|idx| idx as i16 + 1).unwrap_or(0)
}

fn create_tiff<T: gdal_crate::raster::GdalType + Copy>(
    path: &Path,
    class_grid: &GeoGrid,
    nodata: Option<f64>,
) -> Result<Dataset, PredictError> {
    let driver = DriverManager::get_driver_by_name("GTiff")
        .map_err(|e| PredictError::Gdal(format!("GTiff driver: {e}")))?;
    let options = RasterCreationOptions::from_iter([
        "TILED=YES",
        "COMPRESS=LZW",
        "BIGTIFF=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
    ]);
    let mut ds = driver
        .create_with_band_type_with_options::<T, _>(
            path,
            class_grid.width,
            class_grid.height,
            1,
            &options,
        )
        .map_err(|e| PredictError::Gdal(format!("create {}: {e}", path.display())))?;
    ds.set_geo_transform(&class_grid.geo_transform)
        .map_err(|e| PredictError::Gdal(format!("set geotransform: {e}")))?;
    ds.set_projection(&class_grid.crs_wkt)
        .map_err(|e| PredictError::Gdal(format!("set projection: {e}")))?;
    if let Some(value) = nodata {
        let mut band = ds
            .rasterband(1)
            .map_err(|e| PredictError::Gdal(format!("rasterband: {e}")))?;
        band.set_no_data_value(Some(value))
            .map_err(|e| PredictError::Gdal(format!("set nodata: {e}")))?;
    }
    Ok(ds)
}

fn write_tile_window<T: gdal_crate::raster::GdalType + Copy>(
    ds: &mut Dataset,
    tile: &TilePredictions,
    data: &[T],
) -> Result<(), PredictError> {
    let mut band = ds
        .rasterband(1)
        .map_err(|e| PredictError::Gdal(format!("rasterband: {e}")))?;
    let mut buffer = Buffer::new((tile.width, tile.height), data.to_vec());
    band.write(
        (tile.col_off as isize, tile.row_off as isize),
        (tile.width, tile.height),
        &mut buffer,
    )
    .map_err(|e| {
        PredictError::Gdal(format!(
            "write tile row={} col={} size={}x{}: {e}",
            tile.row_off, tile.col_off, tile.width, tile.height
        ))
    })?;
    Ok(())
}

fn write_factor_legend(path: &Path, model: &BasinModelArtifact) -> Result<(), PredictError> {
    let factors: Vec<serde_json::Value> = if model.occurrence_engine.is_numeric_backend() {
        let mut factors = Vec::new();
        for (idx, name) in model
            .occurrence_engine
            .numeric_feature_columns
            .iter()
            .enumerate()
        {
            factors.push(serde_json::json!({
                "map_value": idx + 1,
                "kind": "main_effect",
                "name": name,
                "activation_midpoint": model.occurrence_engine.activation_midpoints.get(idx),
                "learned_threshold": model.occurrence_engine.learned_thresholds.get(idx),
                "interpretation_limit": "model contribution only; not causal",
            }));
        }
        let offset = model.occurrence_engine.numeric_feature_columns.len();
        for (idx, term) in model.occurrence_engine.interactions.iter().enumerate() {
            factors.push(serde_json::json!({
                "map_value": offset + idx + 1,
                "kind": "interaction",
                "factor_id": term.id,
                "name": term.feature_names.join(" * "),
                "order": term.order,
                "feature_indices": term.feature_indices,
                "weight": term.weight,
                "support_pos": term.support_pos,
                "support_neg": term.support_neg,
                "lift": term.lift,
                "score": term.score,
                "activation_midpoints": &term.activation_midpoints,
                "learned_thresholds": &term.learned_thresholds,
                "explanation_text": &term.explanation_text,
                "interpretation_limit": "highest positive model contribution / correlation; not causal",
            }));
        }
        factors
    } else {
        model
            .selected_factors
            .iter()
            .enumerate()
            .map(|(idx, factor)| {
                serde_json::json!({
                    "map_value": idx + 1,
                    "factor_id": factor.factor_id,
                    "name": factor.name,
                    "semantics": factor.semantics,
                    "items": factor.items,
                    "score": factor.score,
                    "relative_risk": factor.relative_risk,
                })
            })
            .collect()
    };
    let raw = serde_json::to_string_pretty(&serde_json::json!({ "factors": factors }))?;
    fs::write(path, raw)?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct DisturbanceTypeSpec {
    value: i16,
    name: &'static str,
    label_zh: &'static str,
}

fn disturbance_type_specs() -> [DisturbanceTypeSpec; 4] {
    [
        DisturbanceTypeSpec {
            value: 1,
            name: "fire",
            label_zh: "火烧",
        },
        DisturbanceTypeSpec {
            value: 2,
            name: "logging",
            label_zh: "砍伐",
        },
        DisturbanceTypeSpec {
            value: 3,
            name: "geo_hazard",
            label_zh: "地质灾害",
        },
        DisturbanceTypeSpec {
            value: 4,
            name: "road_construction",
            label_zh: "道路建设",
        },
    ]
}

fn write_disturbance_type_legend(path: &Path) -> Result<(), PredictError> {
    let classes: Vec<serde_json::Value> = disturbance_type_specs()
        .iter()
        .map(|spec| {
            serde_json::json!({
                "map_value": spec.value,
                "name": spec.name,
                "label_zh": spec.label_zh,
            })
        })
        .collect();
    let raw = serde_json::to_string_pretty(&serde_json::json!({ "classes": classes }))?;
    fs::write(path, raw)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use factor_core::types::InducingFactor;

    fn factor(id: &str, item: &str, score: f64) -> InducingFactor {
        InducingFactor {
            factor_id: id.into(),
            name: id.into(),
            semantics: id.into(),
            items: vec![item.into()],
            support_event: 0.0,
            support_background: 0.0,
            relative_risk: 1.0,
            stability: 1.0,
            score,
            count_event: 0,
            count_background: 0,
            n_event: 0,
            n_background: 0,
            rr_ci_lower: 0.0,
            rr_ci_upper: 0.0,
            p_value: 1.0,
            fdr_significant: false,
            likelihood_ratio: 1.0,
            true_relative_risk: 1.0,
        }
    }

    #[test]
    fn dominant_factor_prefers_active_weight() {
        let factors = vec![
            factor("IF_001", "rainfall=missing", 1.0),
            factor("IF_002", "road=near", 1.0),
        ];
        let idx = dominant_factor_index_from_activations(&[1, 1], &[0.1, 2.0], &factors);
        assert_eq!(idx, 2);
    }

    #[test]
    fn request_keeps_configured_workers() {
        let request = BasinPredictRequest {
            class_raster: PathBuf::from("class.tif"),
            class_start_year: 2016,
            model_path: PathBuf::from("model.json"),
            feature_dir: PathBuf::from("features"),
            output_dir: PathBuf::from("predictions"),
            prediction_year: 2025,
            products: vec![PredictionProduct::Probability],
            workers: 3,
        };
        assert_eq!(request.workers, 3);
    }

    #[test]
    fn zero_workers_means_auto_at_least_one() {
        assert!(effective_workers(0) >= 1);
        assert_eq!(effective_workers(2), 2);
    }
}

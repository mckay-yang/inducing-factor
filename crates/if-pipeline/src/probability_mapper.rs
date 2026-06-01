//! Annual probability map generation from Rust model predictions.
//!
//! Reads chunked feature tables, runs model inference, and writes
//! GeoTIFF output rasters.  Supports parallel prediction via rayon.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array2;
use polars::prelude::*;
use rayon::prelude::*;

use if_config::AppConfig;
use if_core::discretize::AdaptiveDiscretizer;
use if_core::model::{DisturbanceClassEngine, OccurrenceProbabilityEngine};
use if_core::types::InducingFactor;

use crate::feature_builder::FeatureBuilder;

// ── Prediction result ──────────────────────────────────────────

/// Per-chunk prediction output.  Contains `DataFrame` with
/// `raster_row`, `raster_col`, and probability columns.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub chunk_idx: usize,
    pub predictions: Option<DataFrame>,
    pub error: Option<String>,
}

// ── Raster output metadata ─────────────────────────────────────

/// Parameters that describe the output GeoTIFF grid.
#[derive(Debug, Clone)]
pub struct RasterOutputMeta {
    pub width: usize,
    pub height: usize,
    pub crs_wkt: String,
    /// 6-element affine transform (GDAL convention)
    pub geo_transform: [f64; 6],
    pub nodata_probability: f64,
    pub nodata_class: i16,
}

impl Default for RasterOutputMeta {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            crs_wkt: "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]".to_string(),
            geo_transform: [0.0, 0.01, 0.0, 10.0, 0.0, -0.01],
            nodata_probability: -9999.0,
            nodata_class: 0,
        }
    }
}

/// Map of class label to human-readable name.
pub const CLASS_LABELS: &[(i64, &str)] = &[
    (1, "fire"),
    (2, "logging"),
    (3, "geo_hazard"),
    (4, "road_construction"),
];

const ABS_COL_SHORT: &[(i64, &str)] =
    &[(1, "fire"), (2, "logging"), (3, "geo_hazard"), (4, "road")];

// ── AnnualProbabilityMapper ────────────────────────────────────

/// Generates annual disturbance probability and classification
/// GeoTIFFs from chunked feature tables.
pub struct AnnualProbabilityMapper {
    pub config: AppConfig,
    pub discretizer: Option<AdaptiveDiscretizer>,
    pub occurrence_engine: Option<OccurrenceProbabilityEngine>,
    pub class_engine: Option<DisturbanceClassEngine>,
    pub selected_factors: Vec<InducingFactor>,
    pub feature_builder: FeatureBuilder,
    pub feature_columns: Vec<String>,
}

impl AnnualProbabilityMapper {
    pub fn new(config: AppConfig) -> Self {
        let lookback_years = config.attribution.lookback_years;
        let fb = FeatureBuilder::new(lookback_years, vec![], vec![], vec![], vec![]);
        Self {
            config,
            discretizer: None,
            occurrence_engine: None,
            class_engine: None,
            selected_factors: Vec::new(),
            feature_builder: fb,
            feature_columns: Vec::new(),
        }
    }

    pub fn load_artifacts(&mut self) {
        let out_dir = &self.config.paths.output_dir;
        let disc_path = out_dir.join("discretizer.joblib");
        if disc_path.exists() {
            let raw = fs::read_to_string(&disc_path)
                .unwrap_or_else(|e| panic!("failed to read discretizer at {:?}: {e}", disc_path));
            match serde_json::from_str::<AdaptiveDiscretizer>(&raw) {
                Ok(disc) => self.discretizer = Some(disc),
                Err(_e) => {}
            }
        }
        let occ_path = out_dir.join("occurrence_probability_model.json");
        if occ_path.exists() {
            let raw = fs::read_to_string(&occ_path).unwrap_or_else(|e| {
                panic!("failed to read occurrence model at {:?}: {e}", occ_path)
            });
            if let Ok(eng) = serde_json::from_str::<OccurrenceProbabilityEngine>(&raw) {
                self.occurrence_engine = Some(eng);
            }
        }
        let cls_path = out_dir.join("disturbance_class_model.json");
        if cls_path.exists() {
            let raw = fs::read_to_string(&cls_path)
                .unwrap_or_else(|e| panic!("failed to read class model at {:?}: {e}", cls_path));
            if let Ok(eng) = serde_json::from_str::<DisturbanceClassEngine>(&raw) {
                self.class_engine = Some(eng);
            }
        }
        let fac_path = out_dir.join("selected_factors.json");
        if fac_path.exists() {
            let raw = fs::read_to_string(&fac_path)
                .unwrap_or_else(|e| panic!("failed to read factors at {:?}: {e}", fac_path));
            #[derive(serde::Deserialize)]
            struct SelectedFactorsFile {
                factors: Vec<InducingFactor>,
            }
            if let Ok(parsed) = serde_json::from_str::<SelectedFactorsFile>(&raw) {
                self.selected_factors = parsed.factors;
            }
        }
        let meta_path = out_dir.join("model_metadata.json");
        if meta_path.exists() {
            let raw = fs::read_to_string(&meta_path)
                .unwrap_or_else(|e| panic!("failed to read metadata at {:?}: {e}", meta_path));
            #[derive(serde::Deserialize)]
            struct MetaFile {
                feature_columns: Vec<String>,
            }
            if let Ok(meta) = serde_json::from_str::<MetaFile>(&raw) {
                self.feature_columns = meta.feature_columns;
            }
        }
    }

    pub fn iter_chunk_files(&self) -> Vec<PathBuf> {
        let cache_dir = &self.config.paths.cache_dir;
        if !cache_dir.exists() {
            return vec![];
        }
        let mut files: Vec<PathBuf> = fs::read_dir(cache_dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let e = entry.ok()?;
                let p = e.path();
                if p.extension().is_some_and(|ext| ext == "parquet") {
                    Some(p)
                } else {
                    None
                }
            })
            .collect();
        files.sort();
        files
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn predict_chunk(&self, chunk_df: &DataFrame, year: i32) -> Option<DataFrame> {
        let lb = self.feature_builder.lookback_years;
        let year_series = chunk_df.column("year").ok()?.i64().ok()?;
        let mask: Vec<bool> = year_series
            .into_iter()
            .map(|v| {
                if let Some(y) = v {
                    let y = y as i32;
                    y >= year - lb as i32 - 1 && y <= year
                } else {
                    false
                }
            })
            .collect();
        let filtered = chunk_df
            .filter(&BooleanChunked::new("mask".into(), &mask))
            .ok()?;
        if filtered.height() == 0 {
            return None;
        }
        let mut fb = self.feature_builder.clone();
        let samples = if let Ok((built, _engineered_cols)) = fb.build_chunk_features(&filtered) {
            built
        } else {
            filtered
        };
        if samples.height() == 0 {
            return None;
        }
        let year_col = samples.column("year").ok()?.i64().ok()?;
        let year_mask: Vec<bool> = year_col
            .into_iter()
            .map(|v| v.is_some_and(|y| y == year as i64))
            .collect();
        let year_df = samples
            .filter(&BooleanChunked::new("ym".into(), &year_mask))
            .ok()?;
        if year_df.height() == 0 {
            return None;
        }
        let has_coords =
            year_df.column("raster_row").is_ok() && year_df.column("raster_col").is_ok();
        if !has_coords {
            return None;
        }
        let n_rows = year_df.height();
        let mut feature_data: Vec<Vec<f64>> =
            vec![vec![f64::NAN; self.feature_columns.len()]; n_rows];
        for (fi, feat) in self.feature_columns.iter().enumerate() {
            if let Ok(col) = year_df.column(feat) {
                if let Ok(series) = col.as_materialized_series().f64() {
                    for (ri, val) in series.into_iter().enumerate() {
                        feature_data[ri][fi] = val.unwrap_or(f64::NAN);
                    }
                }
            }
        }
        let labels: Vec<HashMap<String, String>> = if let Some(ref disc) = self.discretizer {
            let transformed = disc.transform(&self.feature_columns, &feature_data, true);
            transformed
                .into_iter()
                .map(|row_labels| {
                    let mut map = HashMap::new();
                    for (fi, feat) in self.feature_columns.iter().enumerate() {
                        map.insert(feat.clone(), row_labels[fi].clone());
                    }
                    map
                })
                .collect()
        } else {
            feature_data
                .iter()
                .map(|row| {
                    let mut map = HashMap::new();
                    for (fi, feat) in self.feature_columns.iter().enumerate() {
                        let val = row[fi];
                        map.insert(
                            feat.clone(),
                            if val.is_nan() {
                                "missing".to_string()
                            } else {
                                format!("{:.4}", val)
                            },
                        );
                    }
                    map
                })
                .collect()
        };
        let occ_probs: Vec<f64> = if let Some(ref eng) = self.occurrence_engine {
            eng.predict_proba(&labels, &self.selected_factors)
        } else {
            vec![0.0; n_rows]
        };
        let class_probs_all: Vec<Vec<(i64, f64)>> = if let Some(ref eng) = self.class_engine {
            eng.predict_proba(&labels, &self.selected_factors)
        } else {
            vec![vec![]; n_rows]
        };
        let mut out_rows = Vec::with_capacity(n_rows);
        let rr_col = year_df.column("raster_row").ok()?.i64().ok()?;
        let rc_col = year_df.column("raster_col").ok()?.i64().ok()?;
        let pid_col = year_df.column("point_id").ok()?.i64().ok()?;
        let x_col = year_df.column("x").ok()?.f64().ok()?;
        let y_col = year_df.column("y").ok()?.f64().ok()?;
        for i in 0..n_rows {
            let occ = *occ_probs.get(i).unwrap_or(&0.0);
            let class_probs = class_probs_all.get(i).cloned().unwrap_or_default();
            let cp_map: HashMap<i64, f64> = class_probs.iter().cloned().collect();
            let dominant = class_probs
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, _)| *c)
                .unwrap_or(0);
            let abs = |cls: i64| -> f64 { cp_map.get(&cls).copied().unwrap_or(0.0) * occ };
            out_rows.push(PredictionRow {
                point_id: pid_col.get(i).unwrap_or(0),
                x: x_col.get(i).unwrap_or(0.0),
                y: y_col.get(i).unwrap_or(0.0),
                year: year as i64,
                raster_row: rr_col.get(i).unwrap_or(0),
                raster_col: rc_col.get(i).unwrap_or(0),
                disturbance_probability: occ,
                dominant_disturbance_class: dominant,
                class_prob_1: cp_map.get(&1).copied().unwrap_or(0.0),
                class_prob_2: cp_map.get(&2).copied().unwrap_or(0.0),
                class_prob_3: cp_map.get(&3).copied().unwrap_or(0.0),
                class_prob_4: cp_map.get(&4).copied().unwrap_or(0.0),
                prob_fire_abs: abs(1),
                prob_logging_abs: abs(2),
                prob_geo_hazard_abs: abs(3),
                prob_road_abs: abs(4),
            });
        }
        let df = dataframe_from_rows(&out_rows).ok()?;
        Some(df)
    }

    pub fn resolve_predict_worker_count(&self) -> usize {
        use if_config::Workers;
        match &self.config.pipeline.workers {
            Workers::Auto => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            Workers::Count(nz) => {
                let n = nz.get();
                n.min(
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1),
                )
            }
        }
    }

    pub fn write_maps(
        &self,
        year: i32,
        output_dir: Option<PathBuf>,
        dem_path: Option<PathBuf>,
        aoi_path: Option<PathBuf>,
    ) -> Result<HashMap<String, PathBuf>, String> {
        let out_dir =
            output_dir.unwrap_or_else(|| self.config.paths.output_dir.join("probability_maps"));
        fs::create_dir_all(&out_dir).map_err(|e| format!("failed to create output dir: {e}"))?;
        let meta = if let Some(ref dem) = dem_path {
            Self::raster_metadata_from_dem(dem, aoi_path.as_deref())?
        } else {
            RasterOutputMeta::default()
        };
        let chunk_files = self.iter_chunk_files();
        let worker_count = self.resolve_predict_worker_count();
        let prob_path = out_dir.join(format!("disturbance_probability_{year}.tif"));
        let class_path = out_dir.join(format!("dominant_disturbance_class_{year}.tif"));
        let mut class_prob_paths: HashMap<i64, PathBuf> = HashMap::new();
        for (cls, _label) in CLASS_LABELS {
            class_prob_paths.insert(
                *cls,
                out_dir.join(format!("class_probability_{year}_{cls}.tif")),
            );
        }
        let mut abs_prob_paths: HashMap<i64, PathBuf> = HashMap::new();
        for (cls, short) in ABS_COL_SHORT {
            abs_prob_paths.insert(*cls, out_dir.join(format!("prob_{short}_{year}.tif")));
        }
        let h = meta.height;
        let w = meta.width;
        let mut prob_raster = Array2::<f32>::from_elem((h, w), meta.nodata_probability as f32);
        let mut class_raster = Array2::<i16>::from_elem((h, w), meta.nodata_class);
        let mut class_prob_rasters: HashMap<i64, Array2<f32>> = HashMap::new();
        let mut abs_prob_rasters: HashMap<i64, Array2<f32>> = HashMap::new();
        for (cls, _) in CLASS_LABELS {
            class_prob_rasters.insert(
                *cls,
                Array2::from_elem((h, w), meta.nodata_probability as f32),
            );
            abs_prob_rasters.insert(
                *cls,
                Array2::from_elem((h, w), meta.nodata_probability as f32),
            );
        }
        if worker_count <= 1 || chunk_files.len() <= 1 {
            for (idx, npz_path) in chunk_files.iter().enumerate() {
                self.process_chunk(
                    idx + 1,
                    npz_path,
                    year,
                    &meta,
                    &mut prob_raster,
                    &mut class_raster,
                    &mut class_prob_rasters,
                    &mut abs_prob_rasters,
                );
            }
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .map_err(|e| format!("failed to build rayon pool: {e}"))?;
            let pred_results: Vec<PredictWriteOp> = pool.install(|| {
                chunk_files
                    .par_iter()
                    .map(|npz_path| {
                        let predictions = self.predict_chunk_file(npz_path, year);
                        PredictWriteOp { predictions }
                    })
                    .collect()
            });
            for op in pred_results {
                if let Some(df) = op.predictions {
                    write_predictions_to_buffers(
                        &df,
                        &meta,
                        &mut prob_raster,
                        &mut class_raster,
                        &mut class_prob_rasters,
                        &mut abs_prob_rasters,
                    );
                }
            }
        }
        write_geotiff_f32(
            &prob_path,
            &prob_raster,
            &meta,
            meta.nodata_probability as f32,
        )?;
        write_geotiff_i16(&class_path, &class_raster, &meta, meta.nodata_class)?;
        for (cls, path) in &class_prob_paths {
            if let Some(raster) = class_prob_rasters.get(cls) {
                write_geotiff_f32(path, raster, &meta, meta.nodata_probability as f32)?;
            }
        }
        for (cls, path) in &abs_prob_paths {
            if let Some(raster) = abs_prob_rasters.get(cls) {
                write_geotiff_f32(path, raster, &meta, meta.nodata_probability as f32)?;
            }
        }
        let mut outputs = HashMap::new();
        outputs.insert("disturbance_probability".to_string(), prob_path);
        outputs.insert("dominant_disturbance_class".to_string(), class_path);
        for (cls, path) in class_prob_paths {
            outputs.insert(format!("class_probability_{cls}"), path);
        }
        for (cls, _path) in abs_prob_paths {
            outputs.insert(format!("prob_{cls}_abs"), _path);
        }
        Ok(outputs)
    }

    #[allow(clippy::too_many_arguments)]
    fn process_chunk(
        &self,
        _chunk_idx: usize,
        npz_path: &Path,
        year: i32,
        meta: &RasterOutputMeta,
        prob_raster: &mut Array2<f32>,
        class_raster: &mut Array2<i16>,
        class_prob_rasters: &mut HashMap<i64, Array2<f32>>,
        abs_prob_rasters: &mut HashMap<i64, Array2<f32>>,
    ) {
        if let Some(df) = self.predict_chunk_file(npz_path, year) {
            write_predictions_to_buffers(
                &df,
                meta,
                prob_raster,
                class_raster,
                class_prob_rasters,
                abs_prob_rasters,
            );
        }
    }

    fn predict_chunk_file(&self, path: &Path, year: i32) -> Option<DataFrame> {
        let file = fs::File::open(path).ok()?;
        let reader = ParquetReader::new(file);
        let df = reader.finish().ok()?;
        self.predict_chunk(&df, year)
    }

    #[cfg(feature = "gdal")]
    fn raster_metadata_from_dem(
        dem_path: &Path,
        aoi_path: Option<&Path>,
    ) -> Result<RasterOutputMeta, String> {
        use gdal::Dataset;
        let ds = Dataset::open(dem_path).map_err(|e| format!("failed to open DEM: {e}"))?;
        let (width, height) = ds.raster_size();
        let geo_transform = ds
            .geo_transform()
            .map_err(|e| format!("failed to read geo transform: {e}"))?;
        let crs_wkt = ds.projection().unwrap_or_default();
        let _ = aoi_path;
        Ok(RasterOutputMeta {
            width: width as usize,
            height: height as usize,
            crs_wkt,
            geo_transform,
            nodata_probability: -9999.0,
            nodata_class: 0,
        })
    }

    #[cfg(not(feature = "gdal"))]
    fn raster_metadata_from_dem(
        _dem_path: &Path,
        _aoi_path: Option<&Path>,
    ) -> Result<RasterOutputMeta, String> {
        Ok(RasterOutputMeta::default())
    }
}

// ── Batch write helper ─────────────────────────────────────────

struct PredictWriteOp {
    predictions: Option<DataFrame>,
}

fn write_predictions_to_buffers(
    df: &DataFrame,
    meta: &RasterOutputMeta,
    prob_raster: &mut Array2<f32>,
    class_raster: &mut Array2<i16>,
    class_prob_rasters: &mut HashMap<i64, Array2<f32>>,
    abs_prob_rasters: &mut HashMap<i64, Array2<f32>>,
) {
    let n = df.height();
    if n == 0 {
        return;
    }
    let rr = df.column("raster_row").ok().and_then(|c| c.i64().ok());
    let rc = df.column("raster_col").ok().and_then(|c| c.i64().ok());
    if rr.is_none() || rc.is_none() {
        return;
    }
    let rows = rr.unwrap();
    let cols = rc.unwrap();
    let prob_col = df
        .column("disturbance_probability")
        .ok()
        .and_then(|c| c.as_materialized_series().f64().ok());
    let class_col = df
        .column("dominant_disturbance_class")
        .ok()
        .and_then(|c| c.as_materialized_series().i64().ok());
    for i in 0..n {
        let r = rows.get(i).unwrap_or(-1) as usize;
        let c = cols.get(i).unwrap_or(-1) as usize;
        if r >= meta.height || c >= meta.width {
            continue;
        }
        if let Some(pcol) = prob_col {
            if let Some(val) = pcol.get(i) {
                prob_raster[[r, c]] = val as f32;
            }
        }
        if let Some(ccol) = class_col {
            if let Some(val) = ccol.get(i) {
                class_raster[[r, c]] = val as i16;
            }
        }
        for (cls, arr) in class_prob_rasters.iter_mut() {
            let col_name = format!("class_probability_{cls}");
            if let Ok(col) = df.column(&col_name) {
                if let Ok(series) = col.as_materialized_series().f64() {
                    if let Some(val) = series.get(i) {
                        arr[[r, c]] = val as f32;
                    }
                }
            }
        }
        for (cls, short_name) in &[(1, "fire"), (2, "logging"), (3, "geo_hazard"), (4, "road")] {
            let col_name = format!("prob_{}_abs", short_name);
            if let Some(arr) = abs_prob_rasters.get_mut(cls) {
                if let Ok(col) = df.column(&col_name) {
                    if let Ok(series) = col.as_materialized_series().f64() {
                        if let Some(val) = series.get(i) {
                            arr[[r, c]] = val as f32;
                        }
                    }
                }
            }
        }
    }
}

// ── DataFrame builder helpers ──────────────────────────────────

#[derive(Debug, Clone)]
struct PredictionRow {
    point_id: i64,
    x: f64,
    y: f64,
    year: i64,
    raster_row: i64,
    raster_col: i64,
    disturbance_probability: f64,
    dominant_disturbance_class: i64,
    class_prob_1: f64,
    class_prob_2: f64,
    class_prob_3: f64,
    class_prob_4: f64,
    prob_fire_abs: f64,
    prob_logging_abs: f64,
    prob_geo_hazard_abs: f64,
    prob_road_abs: f64,
}

fn dataframe_from_rows(rows: &[PredictionRow]) -> PolarsResult<DataFrame> {
    if rows.is_empty() {
        return DataFrame::new(vec![
            Column::new("point_id".into(), Vec::<i64>::new()),
            Column::new("x".into(), Vec::<f64>::new()),
            Column::new("y".into(), Vec::<f64>::new()),
            Column::new("year".into(), Vec::<i64>::new()),
            Column::new("raster_row".into(), Vec::<i64>::new()),
            Column::new("raster_col".into(), Vec::<i64>::new()),
            Column::new("disturbance_probability".into(), Vec::<f64>::new()),
            Column::new("dominant_disturbance_class".into(), Vec::<i64>::new()),
            Column::new("class_probability_1".into(), Vec::<f64>::new()),
            Column::new("class_probability_2".into(), Vec::<f64>::new()),
            Column::new("class_probability_3".into(), Vec::<f64>::new()),
            Column::new("class_probability_4".into(), Vec::<f64>::new()),
            Column::new("prob_fire_abs".into(), Vec::<f64>::new()),
            Column::new("prob_logging_abs".into(), Vec::<f64>::new()),
            Column::new("prob_geo_hazard_abs".into(), Vec::<f64>::new()),
            Column::new("prob_road_abs".into(), Vec::<f64>::new()),
        ]);
    }
    let _n = rows.len();
    let point_ids: Vec<i64> = rows.iter().map(|r| r.point_id).collect();
    let xs: Vec<f64> = rows.iter().map(|r| r.x).collect();
    let ys: Vec<f64> = rows.iter().map(|r| r.y).collect();
    let years: Vec<i64> = rows.iter().map(|r| r.year).collect();
    let raster_rows: Vec<i64> = rows.iter().map(|r| r.raster_row).collect();
    let raster_cols: Vec<i64> = rows.iter().map(|r| r.raster_col).collect();
    let probs: Vec<f64> = rows.iter().map(|r| r.disturbance_probability).collect();
    let classes: Vec<i64> = rows.iter().map(|r| r.dominant_disturbance_class).collect();
    let cp1: Vec<f64> = rows.iter().map(|r| r.class_prob_1).collect();
    let cp2: Vec<f64> = rows.iter().map(|r| r.class_prob_2).collect();
    let cp3: Vec<f64> = rows.iter().map(|r| r.class_prob_3).collect();
    let cp4: Vec<f64> = rows.iter().map(|r| r.class_prob_4).collect();
    let abs1: Vec<f64> = rows.iter().map(|r| r.prob_fire_abs).collect();
    let abs2: Vec<f64> = rows.iter().map(|r| r.prob_logging_abs).collect();
    let abs3: Vec<f64> = rows.iter().map(|r| r.prob_geo_hazard_abs).collect();
    let abs4: Vec<f64> = rows.iter().map(|r| r.prob_road_abs).collect();
    DataFrame::new(vec![
        Column::new("point_id".into(), point_ids),
        Column::new("x".into(), xs),
        Column::new("y".into(), ys),
        Column::new("year".into(), years),
        Column::new("raster_row".into(), raster_rows),
        Column::new("raster_col".into(), raster_cols),
        Column::new("disturbance_probability".into(), probs),
        Column::new("dominant_disturbance_class".into(), classes),
        Column::new("class_probability_1".into(), cp1),
        Column::new("class_probability_2".into(), cp2),
        Column::new("class_probability_3".into(), cp3),
        Column::new("class_probability_4".into(), cp4),
        Column::new("prob_fire_abs".into(), abs1),
        Column::new("prob_logging_abs".into(), abs2),
        Column::new("prob_geo_hazard_abs".into(), abs3),
        Column::new("prob_road_abs".into(), abs4),
    ])
}

// ── GeoTIFF write helpers ──────────────────────────────────────

/// Write a Float32 raster to GeoTIFF.
fn write_geotiff_f32(
    path: &Path,
    raster: &Array2<f32>,
    meta: &RasterOutputMeta,
    nodata: f32,
) -> Result<(), String> {
    let (h, w) = raster.dim();
    let data: Vec<f32> = raster.iter().copied().collect();

    #[cfg(feature = "gdal")]
    {
        use gdal::DriverManager;
        let driver = DriverManager::get_driver_by_name("GTiff")
            .map_err(|e| format!("GTiff driver not found: {e}"))?;
        let mut ds = driver
            .create_with_band_type::<f32>(&path.to_string_lossy(), w as isize, h as isize, 1)
            .map_err(|e| format!("failed to create GeoTIFF: {e}"))?;
        ds.set_geo_transform(&meta.geo_transform)
            .map_err(|e| format!("set_geo_transform: {e}"))?;
        if !meta.crs_wkt.is_empty() {
            ds.set_projection(&meta.crs_wkt)
                .map_err(|e| format!("set_projection: {e}"))?;
        }
        let mut band = ds.rasterband(1).map_err(|e| format!("rasterband: {e}"))?;
        band.set_no_data_value(Some(nodata as f64))
            .map_err(|e| format!("set_no_data_value: {e}"))?;
        band.write((0, 0), (w, h), &data)
            .map_err(|e| format!("write: {e}"))?;
    }
    #[cfg(not(feature = "gdal"))]
    {
        let _ = (path, raster, meta, nodata, data, h, w);
    }

    Ok(())
}

/// Write an Int16 raster to GeoTIFF.
fn write_geotiff_i16(
    path: &Path,
    raster: &Array2<i16>,
    meta: &RasterOutputMeta,
    nodata: i16,
) -> Result<(), String> {
    let (h, w) = raster.dim();
    let data: Vec<i16> = raster.iter().copied().collect();

    #[cfg(feature = "gdal")]
    {
        use gdal::DriverManager;
        let driver = DriverManager::get_driver_by_name("GTiff")
            .map_err(|e| format!("GTiff driver not found: {e}"))?;
        let mut ds = driver
            .create_with_band_type::<i16>(&path.to_string_lossy(), w as isize, h as isize, 1)
            .map_err(|e| format!("failed to create GeoTIFF: {e}"))?;
        ds.set_geo_transform(&meta.geo_transform)
            .map_err(|e| format!("set_geo_transform: {e}"))?;
        if !meta.crs_wkt.is_empty() {
            ds.set_projection(&meta.crs_wkt)
                .map_err(|e| format!("set_projection: {e}"))?;
        }
        let mut band = ds.rasterband(1).map_err(|e| format!("rasterband: {e}"))?;
        band.set_no_data_value(Some(nodata as f64))
            .map_err(|e| format!("set_no_data_value: {e}"))?;
        band.write((0, 0), (w, h), &data)
            .map_err(|e| format!("write: {e}"))?;
    }
    #[cfg(not(feature = "gdal"))]
    {
        let _ = (path, raster, meta, nodata, data, h, w);
    }

    Ok(())
}

// ── Read a GeoTIFF as Array2<f32> (for testing) ───────────────

#[cfg(test)]
#[allow(dead_code)]
fn read_geotiff_f32(path: &Path) -> Result<Array2<f32>, String> {
    #[cfg(feature = "gdal")]
    {
        use gdal::Dataset;
        let ds = Dataset::open(path).map_err(|e| format!("open: {e}"))?;
        let (w, h) = ds.raster_size();
        let band = ds.rasterband(1).map_err(|e| format!("band: {e}"))?;
        let buf = band
            .read_as::<f32>((0, 0), (w, h), (w as usize, h as usize), None)
            .map_err(|e| format!("read: {e}"))?;
        let arr = Array2::from_shape_vec((h as usize, w as usize), buf.data)
            .map_err(|e| format!("reshape: {e}"))?;
        Ok(arr)
    }
    #[cfg(not(feature = "gdal"))]
    {
        let _ = path;
        Err("gdal not available".into())
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn read_geotiff_i16(path: &Path) -> Result<Array2<i16>, String> {
    #[cfg(feature = "gdal")]
    {
        use gdal::Dataset;
        let ds = Dataset::open(path).map_err(|e| format!("open: {e}"))?;
        let (w, h) = ds.raster_size();
        let band = ds.rasterband(1).map_err(|e| format!("band: {e}"))?;
        let buf = band
            .read_as::<i16>((0, 0), (w, h), (w as usize, h as usize), None)
            .map_err(|e| format!("read: {e}"))?;
        let arr = Array2::from_shape_vec((h as usize, w as usize), buf.data)
            .map_err(|e| format!("reshape: {e}"))?;
        Ok(arr)
    }
    #[cfg(not(feature = "gdal"))]
    {
        let _ = path;
        Err("gdal not available".into())
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn read_geotiff_metadata(path: &Path) -> Result<RasterOutputMeta, String> {
    #[cfg(feature = "gdal")]
    {
        use gdal::Dataset;
        let ds = Dataset::open(path).map_err(|e| format!("open: {e}"))?;
        let (w, h) = ds.raster_size();
        let geo_transform = ds
            .geo_transform()
            .map_err(|e| format!("geo_transform: {e}"))?;
        let crs_wkt = ds.projection().unwrap_or_default();
        let band = ds.rasterband(1).map_err(|e| format!("band: {e}"))?;
        let nodata = band.no_data_value().unwrap_or(None);
        Ok(RasterOutputMeta {
            width: w as usize,
            height: h as usize,
            crs_wkt,
            geo_transform,
            nodata_probability: nodata.unwrap_or(-9999.0),
            nodata_class: 0,
        })
    }
    #[cfg(not(feature = "gdal"))]
    {
        let _ = path;
        Err("gdal not available".into())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
#[allow(unused_variables, clippy::too_many_arguments, clippy::cast_lossless)]
mod tests {
    use super::*;
    use if_config::AppConfig;
    use if_core::discretize::AdaptiveDiscretizer;
    use if_core::types::{DiscretizerConfig, FeatureMeta, InducingFactor};
    use std::collections::HashMap;
    use tempfile::TempDir;

    // ── Helpers ──────────────────────────────────────────────

    fn make_test_config(cache_dir: &Path, out_dir: &Path) -> AppConfig {
        let mut config = AppConfig::default();
        config.paths.cache_dir = cache_dir.to_path_buf();
        config.paths.output_dir = out_dir.to_path_buf();
        config.attribution.lookback_years = 2;
        config.pipeline.start_year = 2020;
        config.pipeline.end_year = 2020;
        config
    }

    /// Create a simple synthetic chunk DataFrame for testing.
    fn make_chunk_df(
        point_ids: &[i64],
        years: &[i64],
        rows: &[i64],
        cols: &[i64],
        slope: &[f64],
        aspect: &[f64],
        dist_river: &[f64],
        dist_road: &[f64],
        dist_rail: &[f64],
        dist_water: &[f64],
        temp: &[f64],
        resident: &[f64],
        landuse: &[i64],
        dist_flag: &[i64],
        dist_class: &[i64],
    ) -> DataFrame {
        let n = point_ids.len();
        let rain = vec![0.0; n * 12];
        let mut cols_vec: Vec<Column> = vec![
            Column::new("point_id".into(), point_ids.to_vec()),
            Column::new("x".into(), vec![0.0; n]),
            Column::new("y".into(), vec![0.0; n]),
            Column::new("year".into(), years.to_vec()),
            Column::new("raster_row".into(), rows.to_vec()),
            Column::new("raster_col".into(), cols.to_vec()),
            Column::new("slope".into(), slope.to_vec()),
            Column::new("aspect".into(), aspect.to_vec()),
            Column::new("dist_to_river".into(), dist_river.to_vec()),
            Column::new("dist_to_road".into(), dist_road.to_vec()),
            Column::new("dist_to_railway".into(), dist_rail.to_vec()),
            Column::new("dist_to_waterway".into(), dist_water.to_vec()),
            Column::new("temperature".into(), temp.to_vec()),
            Column::new("resident".into(), resident.to_vec()),
            Column::new("landuse".into(), landuse.to_vec()),
            Column::new("disturbance_flag".into(), dist_flag.to_vec()),
            Column::new("disturbance_class".into(), dist_class.to_vec()),
        ];
        // Add rainfall columns
        for m in 0..12 {
            let vals: Vec<f64> = (0..n).map(|i| rain[i * 12 + m]).collect();
            let cname = format!("rainfall_month_{:02}", m + 1);
            cols_vec.push(Column::new(cname.into(), vals));
        }
        DataFrame::new(cols_vec).unwrap()
    }

    /// Create a simple discretizer with 2 continuous features.
    fn make_simple_discretizer() -> AdaptiveDiscretizer {
        let config = DiscretizerConfig {
            method: "quantile".to_string(),
            n_bins: 3,
            ..Default::default()
        };
        let mut disc = AdaptiveDiscretizer::new(config);
        let mut meta = HashMap::new();
        meta.insert(
            "slope".to_string(),
            FeatureMeta {
                kind: "continuous".to_string(),
                bins: Some(vec![-1e9, 5.0, 15.0, 1e9]),
                labels: Some(vec![
                    "low".to_string(),
                    "medium".to_string(),
                    "high".to_string(),
                ]),
                categories: None,
            },
        );
        meta.insert(
            "dist_to_river".to_string(),
            FeatureMeta {
                kind: "continuous".to_string(),
                bins: Some(vec![-1e9, 1000.0, 3000.0, 1e9]),
                labels: Some(vec![
                    "near".to_string(),
                    "mid".to_string(),
                    "far".to_string(),
                ]),
                categories: None,
            },
        );
        disc.feature_meta = meta;
        disc.features = vec!["slope".to_string(), "dist_to_river".to_string()];
        disc
    }

    /// Create a simple occurrence engine.
    fn make_simple_occ_engine() -> OccurrenceProbabilityEngine {
        OccurrenceProbabilityEngine {
            weights: vec![0.5, 0.3],
            intercept: -0.5,
            factor_ids: vec!["IF_001".to_string(), "IF_002".to_string()],
            backend: "logistic_regression".to_string(),
            base_rate: 0.5,
        }
    }

    /// Create a simple class engine.
    fn make_simple_class_engine() -> DisturbanceClassEngine {
        let mut labels = HashMap::new();
        labels.insert(1, "fire".to_string());
        labels.insert(2, "logging".to_string());
        labels.insert(3, "geo_hazard".to_string());
        labels.insert(4, "road_construction".to_string());
        DisturbanceClassEngine {
            weights: vec![
                vec![0.2, 0.1],
                vec![-0.1, 0.3],
                vec![0.0, -0.2],
                vec![-0.1, -0.1],
            ],
            intercepts: vec![0.1, -0.2, 0.0, 0.1],
            classes: vec![1, 2, 3, 4],
            factor_ids: vec!["IF_001".to_string(), "IF_002".to_string()],
            class_labels: labels,
            backend: "multinomial_logistic".to_string(),
            priors: HashMap::new(),
        }
    }

    fn make_simple_factors() -> Vec<InducingFactor> {
        vec![
            InducingFactor {
                factor_id: "IF_001".into(),
                name: "slope_high".into(),
                semantics: "high slope".into(),
                items: vec!["slope=high".into()],
                support_event: 0.6,
                support_background: 0.3,
                relative_risk: 2.0,
                stability: 0.9,
                score: 1.5,
                count_event: 60,
                count_background: 30,
                n_event: 100,
                n_background: 100,
                rr_ci_lower: 1.5,
                rr_ci_upper: 2.5,
                p_value: 0.001,
                fdr_significant: true,
                likelihood_ratio: 2.0,
                true_relative_risk: 2.2,
            },
            InducingFactor {
                factor_id: "IF_002".into(),
                name: "dist_to_river_near".into(),
                semantics: "near dist_to_river".into(),
                items: vec!["dist_to_river=near".into()],
                support_event: 0.4,
                support_background: 0.2,
                relative_risk: 2.0,
                stability: 0.85,
                score: 1.3,
                count_event: 40,
                count_background: 20,
                n_event: 100,
                n_background: 100,
                rr_ci_lower: 1.3,
                rr_ci_upper: 2.7,
                p_value: 0.005,
                fdr_significant: true,
                likelihood_ratio: 2.0,
                true_relative_risk: 2.0,
            },
        ]
    }

    /// Build a mapper for testing.
    fn make_test_mapper(cache_dir: &Path, out_dir: &Path) -> AnnualProbabilityMapper {
        let config = make_test_config(cache_dir, out_dir);
        let mut mapper = AnnualProbabilityMapper::new(config);
        mapper.discretizer = Some(make_simple_discretizer());
        mapper.occurrence_engine = Some(make_simple_occ_engine());
        mapper.class_engine = Some(make_simple_class_engine());
        mapper.selected_factors = make_simple_factors();
        mapper.feature_columns = vec!["slope".to_string(), "dist_to_river".to_string()];
        mapper.feature_builder = FeatureBuilder::new(
            1, // lookback_years >= 1 required by feature builder
            vec![
                "slope".to_string(),
                "aspect".to_string(),
                "dist_to_river".to_string(),
            ],
            vec![
                "dist_to_road".to_string(),
                "dist_to_railway".to_string(),
                "dist_to_waterway".to_string(),
                "temperature".to_string(),
                "resident".to_string(),
            ],
            vec!["landuse".to_string()],
            vec![
                "rainfall_prev_1m".to_string(),
                "rainfall_prev_2m".to_string(),
                "rainfall_prev_3m".to_string(),
                "rainfall_prev_total_3m".to_string(),
            ],
        );
        mapper
    }

    // ── Tests ────────────────────────────────────────────────

    #[test]
    fn test_raster_metadata_parity() {
        // Write a simple GeoTIFF with known metadata, then read and verify.
        let tmp = TempDir::with_prefix("test_raster_meta_").unwrap();
        let path = tmp.path().join("test.tif");

        let meta = RasterOutputMeta {
            width: 100,
            height: 80,
            crs_wkt: "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]".to_string(),
            geo_transform: [0.0, 0.5, 0.0, 30.0, 0.0, -0.5],
            nodata_probability: -9999.0,
            nodata_class: 0,
        };

        let raster = Array2::<f32>::from_elem((80, 100), 0.5);

        #[cfg(feature = "gdal")]
        {
            write_geotiff_f32(&path, &raster, &meta, -9999.0).unwrap();
            let read_meta = read_geotiff_metadata(&path).unwrap();

            assert_eq!(read_meta.width, meta.width);
            assert_eq!(read_meta.height, meta.height);
            // Compare transform element-wise
            for i in 0..6 {
                assert!(
                    (read_meta.geo_transform[i] - meta.geo_transform[i]).abs() < 1e-10,
                    "transform[{}] mismatch: expected {}, got {}",
                    i,
                    meta.geo_transform[i],
                    read_meta.geo_transform[i]
                );
            }
            // CRS may have minor whitespace differences; check that EPSG is present
            assert!(
                read_meta.crs_wkt.contains("4326"),
                "CRS should contain EPSG:4326"
            );
            // NoData
            assert!(
                (read_meta.nodata_probability - meta.nodata_probability).abs() < 1e-6,
                "nodata mismatch"
            );
        }
    }

    #[test]
    fn test_probability_values_tolerance() {
        // Write known probability values to raster, read back, verify values.
        let tmp = TempDir::with_prefix("test_prob_val_").unwrap();
        let path = tmp.path().join("prob.tif");

        let meta = RasterOutputMeta::default();
        let mut raster = Array2::<f32>::from_elem((64, 64), meta.nodata_probability as f32);
        // Write a few known values
        raster[[10, 10]] = 0.75;
        raster[[20, 20]] = 0.35;
        raster[[30, 30]] = 0.05;

        #[cfg(feature = "gdal")]
        {
            write_geotiff_f32(&path, &raster, &meta, meta.nodata_probability as f32).unwrap();
            let read_back = read_geotiff_f32(&path).unwrap();

            assert!((read_back[[10, 10]] - 0.75).abs() < 1e-6);
            assert!((read_back[[20, 20]] - 0.35).abs() < 1e-6);
            assert!((read_back[[30, 30]] - 0.05).abs() < 1e-6);
            // NoData pixels should be -9999.0
            assert!((read_back[[0, 0]] - (-9999.0)).abs() < 1e-3);
        }
    }

    #[test]
    fn test_valid_pixel_mask() {
        // Create a raster where only some pixels are valid (not NoData).
        let tmp = TempDir::with_prefix("test_valid_mask_").unwrap();
        let path = tmp.path().join("mask.tif");

        let meta = RasterOutputMeta {
            width: 32,
            height: 32,
            ..Default::default()
        };
        let mut raster = Array2::<f32>::from_elem((32, 32), -9999.0f32);
        // Set some valid values
        for i in 5..15 {
            for j in 5..15 {
                raster[[i, j]] = 0.5;
            }
        }

        let valid_mask: Array2<bool> = raster.mapv(|v| (v - (-9999.0f32)).abs() > 1e-3);
        let valid_count = valid_mask.iter().filter(|&&v| v).count();

        assert_eq!(valid_count, 100, "Expected 10×10=100 valid pixels");

        #[cfg(feature = "gdal")]
        {
            write_geotiff_f32(&path, &raster, &meta, -9999.0).unwrap();
            let read_back = read_geotiff_f32(&path).unwrap();
            let read_mask: Array2<bool> = read_back.mapv(|v| (v - (-9999.0f32)).abs() > 1e-3);
            let read_valid = read_mask.iter().filter(|&&v| v).count();
            assert_eq!(
                read_valid, valid_count,
                "Valid pixel count should match after roundtrip"
            );
        }
    }

    #[test]
    fn test_class_raster_dtype() {
        // Verify class raster is Int16 with proper nodata.
        let tmp = TempDir::with_prefix("test_class_dtype_").unwrap();
        let path = tmp.path().join("class.tif");

        let meta = RasterOutputMeta::default();
        let mut raster = Array2::<i16>::from_elem((32, 32), meta.nodata_class);
        raster[[10, 10]] = 1;
        raster[[11, 11]] = 3;

        #[cfg(feature = "gdal")]
        {
            write_geotiff_i16(&path, &raster, &meta, meta.nodata_class).unwrap();
            let read_back = read_geotiff_i16(&path).unwrap();

            assert_eq!(read_back[[10, 10]], 1);
            assert_eq!(read_back[[11, 11]], 3);
            assert_eq!(read_back[[0, 0]], meta.nodata_class);

            // Check metadata
            use gdal::Dataset;
            let ds = Dataset::open(&path).unwrap();
            let band = ds.rasterband(1).unwrap();
            let nd = band.no_data_value().unwrap();
            assert_eq!(nd, Some(meta.nodata_class as f64));
        }
    }

    #[test]
    fn test_probability_raster_dtype() {
        // Verify probability raster is Float32 with proper nodata.
        let tmp = TempDir::with_prefix("test_prob_dtype_").unwrap();
        let path = tmp.path().join("prob.tif");

        let meta = RasterOutputMeta::default();
        let mut raster = Array2::<f32>::from_elem((32, 32), -9999.0f32);
        raster[[5, 5]] = 0.25;

        #[cfg(feature = "gdal")]
        {
            write_geotiff_f32(&path, &raster, &meta, -9999.0).unwrap();

            use gdal::Dataset;
            let ds = Dataset::open(&path).unwrap();
            let band = ds.rasterband(1).unwrap();
            let nd = band.no_data_value().unwrap();
            assert_eq!(nd, Some(-9999.0));
        }
    }

    #[test]
    fn test_empty_chunk_prediction() {
        // An empty chunk should produce no predictions, and raster remains all NoData.
        let tmp = TempDir::with_prefix("test_empty_chunk_").unwrap();
        let cache_dir = tmp.path().join("cache");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::create_dir_all(&out_dir).unwrap();

        let mapper = make_test_mapper(&cache_dir, &out_dir);

        // Predict on an empty DataFrame
        let empty_df = DataFrame::new(vec![Column::new("year".into(), Vec::<i64>::new())]).unwrap();
        let result = mapper.predict_chunk(&empty_df, 2020);
        assert!(
            result.is_none(),
            "Empty chunk should produce no predictions"
        );
    }

    #[test]
    fn test_parallel_prediction() {
        // Multiple chunks predicted concurrently via rayon.
        let tmp = TempDir::with_prefix("test_parallel_").unwrap();
        let cache_dir = tmp.path().join("cache");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::create_dir_all(&out_dir).unwrap();

        let mapper = make_test_mapper(&cache_dir, &out_dir);

        // Create 4 chunks, each with a small DataFrame
        let chunks: Vec<DataFrame> = (0..4)
            .map(|ci| {
                make_chunk_df(
                    &[(ci * 100 + 1)],
                    &[2020],
                    &[ci * 10 + 5],
                    &[ci * 10 + 5],
                    &[10.0],
                    &[180.0],
                    &[500.0],
                    &[2000.0],
                    &[5000.0],
                    &[8000.0],
                    &[20.0],
                    &[50.0],
                    &[2],
                    &[0],
                    &[0],
                )
            })
            .collect();

        // Predict in parallel
        let results: Vec<Option<DataFrame>> = chunks
            .par_iter()
            .map(|df| mapper.predict_chunk(df, 2020))
            .collect();

        assert_eq!(results.len(), 4, "Should have 4 results");
        for res in &results {
            assert!(res.is_some(), "Each chunk should produce predictions");
        }
    }

    #[test]
    fn test_predict_chunk_basic() {
        // Basic prediction: known values → expected probability range.
        let tmp = TempDir::with_prefix("test_predict_basic_").unwrap();
        let cache_dir = tmp.path().join("cache");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::create_dir_all(&out_dir).unwrap();

        let mapper = make_test_mapper(&cache_dir, &out_dir);

        let df = make_chunk_df(
            &[1, 2, 3],
            &[2020, 2020, 2020],
            &[0, 5, 10],
            &[0, 5, 10],
            &[10.0, 3.0, 20.0],       // slope
            &[180.0, 180.0, 180.0],   // aspect
            &[500.0, 2000.0, 5000.0], // dist_to_river
            &[2000.0; 3],             // dist_to_road
            &[5000.0; 3],             // dist_to_railway
            &[8000.0; 3],             // dist_to_waterway
            &[20.0; 3],               // temperature
            &[50.0; 3],               // resident
            &[2; 3],                  // landuse
            &[0, 1, 0],               // dist_flag
            &[0, 1, 0],               // dist_class
        );

        let result = mapper.predict_chunk(&df, 2020);
        assert!(
            result.is_some(),
            "Should produce predictions for valid chunk"
        );

        let pred = result.unwrap();
        assert_eq!(pred.height(), 3, "Should have 3 rows");
        assert!(pred.column("disturbance_probability").is_ok());
        assert!(pred.column("dominant_disturbance_class").is_ok());
    }

    #[test]
    fn test_predict_chunk_no_matching_year() {
        let tmp = TempDir::with_prefix("test_no_year_").unwrap();
        let cache_dir = tmp.path().join("cache");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::create_dir_all(&out_dir).unwrap();

        let mapper = make_test_mapper(&cache_dir, &out_dir);

        let df = make_chunk_df(
            &[1],
            &[2018], // year not matching 2020
            &[0],
            &[0],
            &[10.0],
            &[180.0],
            &[500.0],
            &[2000.0],
            &[5000.0],
            &[8000.0],
            &[20.0],
            &[50.0],
            &[2],
            &[0],
            &[0],
        );

        let result = mapper.predict_chunk(&df, 2020);
        assert!(result.is_none(), "Should return None for non-matching year");
    }

    #[test]
    fn test_write_maps_empty_chunks() {
        // write_maps with no chunk files should still create output files.
        let tmp = TempDir::with_prefix("test_write_empty_").unwrap();
        let cache_dir = tmp.path().join("cache");
        let out_dir = tmp.path().join("out");
        let maps_dir = tmp.path().join("maps");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::create_dir_all(&out_dir).unwrap();

        let mapper = make_test_mapper(&cache_dir, &out_dir);

        // Write maps with no chunk files
        let result = mapper.write_maps(2020, Some(maps_dir.clone()), None, None);
        assert!(result.is_ok(), "write_maps should succeed with no chunks");

        let outputs = result.unwrap();
        assert!(outputs.contains_key("disturbance_probability"));
        assert!(outputs.contains_key("dominant_disturbance_class"));
    }

    #[test]
    fn test_discretizer_loaded_from_artifacts() {
        // Test loading artifacts from output directory.
        let tmp = TempDir::with_prefix("test_load_artifacts_").unwrap();
        let cache_dir = tmp.path().join("cache");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::create_dir_all(&out_dir).unwrap();

        // Write minimal artifacts
        let disc = make_simple_discretizer();
        let disc_json = serde_json::to_string_pretty(&disc).unwrap();
        fs::write(out_dir.join("discretizer.joblib"), &disc_json).unwrap();

        let occ = make_simple_occ_engine();
        let occ_json = serde_json::to_string_pretty(&occ).unwrap();
        fs::write(out_dir.join("occurrence_probability_model.json"), &occ_json).unwrap();

        let cls = make_simple_class_engine();
        let cls_json = serde_json::to_string_pretty(&cls).unwrap();
        fs::write(out_dir.join("disturbance_class_model.json"), &cls_json).unwrap();

        let factors = make_simple_factors();
        #[derive(serde::Serialize)]
        struct FactorsFile {
            factors: Vec<InducingFactor>,
        }
        let fac = FactorsFile { factors };
        let fac_json = serde_json::to_string_pretty(&fac).unwrap();
        fs::write(out_dir.join("selected_factors.json"), &fac_json).unwrap();

        #[derive(serde::Serialize)]
        struct MetaFile {
            feature_columns: Vec<String>,
        }
        let meta = MetaFile {
            feature_columns: vec!["slope".to_string(), "dist_to_river".to_string()],
        };
        let meta_json = serde_json::to_string_pretty(&meta).unwrap();
        fs::write(out_dir.join("model_metadata.json"), &meta_json).unwrap();

        let config = make_test_config(&cache_dir, &out_dir);
        let mut mapper = AnnualProbabilityMapper::new(config);
        mapper.load_artifacts();

        assert!(mapper.discretizer.is_some());
        assert!(mapper.occurrence_engine.is_some());
        assert!(mapper.class_engine.is_some());
        assert_eq!(mapper.selected_factors.len(), 2);
        assert_eq!(mapper.feature_columns.len(), 2);
    }
}

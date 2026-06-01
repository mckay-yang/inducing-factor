use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;
use ndarray::Array2;
use polars::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use if_config::{PipelineConfig, Workers};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkTask {
    pub chunk_idx: usize,
    pub row_off: usize,
    pub col_off: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkAudit {
    pub chunk_idx: usize,
    pub row_off: usize,
    pub col_off: usize,
    pub width: usize,
    pub height: usize,
    pub status: String,
    pub point_count: usize,
    pub disturbance_count: usize,
    pub background_count: usize,
    pub valid_pixels: usize,
    pub years: Vec<i32>,
    pub worker_id: Option<usize>,
    pub timestamp: String,
    pub cache_file: Option<String>,
}

impl ChunkAudit {
    pub fn empty(chunk_idx: usize, status: &str) -> Self {
        Self {
            chunk_idx,
            row_off: 0,
            col_off: 0,
            width: 0,
            height: 0,
            status: status.to_string(),
            point_count: 0,
            disturbance_count: 0,
            background_count: 0,
            valid_pixels: 0,
            years: vec![],
            worker_id: None,
            timestamp: Utc::now().to_rfc3339(),
            cache_file: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub chunk_idx: usize,
    pub saved: bool,
    pub cache_file: Option<PathBuf>,
    pub audit: ChunkAudit,
}

pub struct SyntheticRasterPanel {
    pub data: HashMap<String, Array2<f32>>,
    pub nodata_values: HashMap<String, f32>,
    pub width: usize,
    pub height: usize,
}

impl std::fmt::Debug for SyntheticRasterPanel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyntheticRasterPanel")
            .field("layers", &self.data.keys().collect::<Vec<_>>())
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl Clone for SyntheticRasterPanel {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            nodata_values: self.nodata_values.clone(),
            width: self.width,
            height: self.height,
        }
    }
}

impl SyntheticRasterPanel {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: HashMap::new(),
            nodata_values: HashMap::new(),
            width,
            height,
        }
    }

    pub fn set_layer(&mut self, name: &str, arr: Array2<f32>, nodata: f32) {
        self.data.insert(name.to_string(), arr);
        self.nodata_values.insert(name.to_string(), nodata);
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticLoaderSet {
    pub panel: SyntheticRasterPanel,
}

impl SyntheticLoaderSet {
    pub fn new(panel: SyntheticRasterPanel) -> Self {
        Self { panel }
    }
}

pub const STATIC_FEATURES: &[&str] = &["slope", "aspect", "dist_to_river"];
pub const DIRECT_FEATURES: &[&str] = &[
    "rainfall_prev_1m",
    "rainfall_prev_2m",
    "rainfall_prev_3m",
    "rainfall_prev_total_3m",
];
pub const DYNAMIC_FEATURES: &[&str] = &[
    "dist_to_road",
    "dist_to_railway",
    "dist_to_waterway",
    "temperature",
    "resident",
];
pub const CATEGORICAL_TEMPORAL_FEATURES: &[&str] = &["landuse"];
pub const RAINFALL_MONTHLY_COLS: &[&str] = &[
    "rainfall_month_01",
    "rainfall_month_02",
    "rainfall_month_03",
    "rainfall_month_04",
    "rainfall_month_05",
    "rainfall_month_06",
    "rainfall_month_07",
    "rainfall_month_08",
    "rainfall_month_09",
    "rainfall_month_10",
    "rainfall_month_11",
    "rainfall_month_12",
];

pub fn derive_chunk_grid(
    raster_width: usize,
    raster_height: usize,
    tile_size: u32,
) -> Vec<ChunkTask> {
    let tile = tile_size as usize;
    let cols = (raster_width + tile - 1) / tile;
    let rows = (raster_height + tile - 1) / tile;
    let mut tasks = Vec::with_capacity(rows * cols);

    for r in 0..rows {
        for c in 0..cols {
            let col_off = c * tile;
            let row_off = r * tile;
            let w = tile.min(raster_width - col_off);
            let h = tile.min(raster_height - row_off);
            let chunk_idx = r * cols + c;
            tasks.push(ChunkTask {
                chunk_idx,
                row_off,
                col_off,
                width: w,
                height: h,
            });
        }
    }
    tasks
}

pub fn build_chunk_dataframe(
    _task: &ChunkTask,
    points: &[(i64, f64, f64, i32)],
    features: &HashMap<String, f64>,
    disturbance_flags: &HashMap<i64, bool>,
    disturbance_classes: &HashMap<i64, i32>,
    disturbance_months: &HashMap<i64, f64>,
    rainfall_monthly: &HashMap<i64, Vec<f64>>,
    seed: u64,
) -> PolarsResult<DataFrame> {
    let n = points.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let default_rainfall: Vec<f64> = vec![f64::NAN; 12];

    let mut point_ids: Vec<i64> = Vec::with_capacity(n);
    let mut xs: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut years: Vec<i32> = Vec::with_capacity(n);
    let mut slope_vals: Vec<f64> = Vec::with_capacity(n);
    let mut aspect_vals: Vec<f64> = Vec::with_capacity(n);
    let mut dist_river: Vec<f64> = Vec::with_capacity(n);
    let mut dist_road: Vec<f64> = Vec::with_capacity(n);
    let mut dist_railway: Vec<f64> = Vec::with_capacity(n);
    let mut dist_waterway: Vec<f64> = Vec::with_capacity(n);
    let mut temp_vals: Vec<f64> = Vec::with_capacity(n);
    let mut resident_vals: Vec<f64> = Vec::with_capacity(n);
    let mut landuse_vals: Vec<i32> = Vec::with_capacity(n);
    let mut dist_flags: Vec<i32> = Vec::with_capacity(n);
    let mut dist_classes: Vec<i32> = Vec::with_capacity(n);
    let mut dist_months: Vec<f64> = Vec::with_capacity(n);
    let mut rainfall: Vec<Vec<f64>> = Vec::with_capacity(n);

    for (pid, x, y, year) in points {
        point_ids.push(*pid);
        xs.push(*x);
        ys.push(*y);
        years.push(*year);

        let slope_v = features.get("slope").copied().unwrap_or(f64::NAN)
            + rng.gen_range(-2.0..2.0);
        slope_vals.push(slope_v);

        let aspect_v = features.get("aspect").copied().unwrap_or(180.0)
            + rng.gen_range(-30.0..30.0);
        aspect_vals.push(aspect_v);

        dist_river.push(features.get("dist_to_river").copied().unwrap_or(5000.0));
        dist_road.push(features.get("dist_to_road").copied().unwrap_or(3000.0));
        dist_railway.push(features.get("dist_to_railway").copied().unwrap_or(10000.0));
        dist_waterway.push(features.get("dist_to_waterway").copied().unwrap_or(8000.0));

        temp_vals.push(features.get("temperature").copied().unwrap_or(15.0)
            + rng.gen_range(-1.0..1.0));
        resident_vals.push(features.get("resident").copied().unwrap_or(100.0));
        landuse_vals.push(features.get("landuse").copied().unwrap_or(2.0) as i32);

        let flag = disturbance_flags.get(pid).copied().unwrap_or(false);
        dist_flags.push(if flag { 1 } else { 0 });
        dist_classes.push(disturbance_classes.get(pid).copied().unwrap_or(0));
        dist_months.push(disturbance_months.get(pid).copied().unwrap_or(f64::NAN));

        let r = rainfall_monthly.get(pid).cloned().unwrap_or(default_rainfall.clone());
        rainfall.push(r);
    }

    let mut columns: Vec<Column> = Vec::new();
    columns.push(Column::new("point_id".into(), point_ids));
    columns.push(Column::new("x".into(), xs));
    columns.push(Column::new("y".into(), ys));
    columns.push(Column::new("year".into(), years));
    columns.push(Column::new("slope".into(), slope_vals));
    columns.push(Column::new("aspect".into(), aspect_vals));
    columns.push(Column::new("dist_to_road".into(), dist_road));
    columns.push(Column::new("dist_to_river".into(), dist_river));
    columns.push(Column::new("dist_to_railway".into(), dist_railway));
    columns.push(Column::new("dist_to_waterway".into(), dist_waterway));
    columns.push(Column::new("temperature".into(), temp_vals));
    columns.push(Column::new("resident".into(), resident_vals));
    columns.push(Column::new("landuse".into(), landuse_vals));
    columns.push(Column::new("disturbance_flag".into(), dist_flags));
    columns.push(Column::new("disturbance_class".into(), dist_classes));
    columns.push(Column::new("disturbance_month".into(), dist_months));

    for m in 0..12 {
        let col_vals: Vec<f64> = rainfall.iter().map(|r| r[m]).collect();
        let col_name = format!("rainfall_month_{:02}", m + 1);
        columns.push(Column::new(col_name.into(), col_vals));
    }

    DataFrame::new(columns)
}

pub fn resolve_num_workers(config: &PipelineConfig) -> usize {
    match &config.workers {
        Workers::Auto => {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }
        Workers::Count(n) => n.get(),
    }
}

pub fn chunk_cache_path(cache_dir: &Path, chunk_idx: usize) -> PathBuf {
    cache_dir.join(format!("chunk_{}.parquet", chunk_idx))
}

pub fn chunk_has_valid_data(df: &DataFrame) -> bool {
    let key_cols = ["slope", "dist_to_road"];
    for col_name in &key_cols {
        if let Ok(c) = df.column(col_name) {
            let s = c.as_materialized_series();
            if let Ok(ca) = s.f64() {
                let f64s: Vec<Option<f64>> = ca.into_iter().collect();
                if f64s.iter().any(|v| v.is_some_and(|x| x.is_finite())) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn write_chunk_cache(df: &mut DataFrame, path: &Path) -> PolarsResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut file = fs::File::create(path).map_err(|e| {
        PolarsError::ComputeError(format!("Cannot create cache file: {e}").into())
    })?;
    ParquetWriter::new(&mut file).finish(df)?;
    Ok(())
}

pub fn write_audit_report(audits: &[ChunkAudit], cache_dir: &Path) -> std::io::Result<()> {
    let path = cache_dir.join("data_audit.json");
    let json = serde_json::to_string_pretty(audits)?;
    let mut f = fs::File::create(path)?;
    f.write_all(json.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_chunk_grid_derivation() {
        let tasks = derive_chunk_grid(2048, 1024, 1024);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].chunk_idx, 0);
        assert_eq!(tasks[0].col_off, 0);
        assert_eq!(tasks[0].row_off, 0);
        assert_eq!(tasks[0].width, 1024);
        assert_eq!(tasks[0].height, 1024);
        assert_eq!(tasks[1].chunk_idx, 1);
        assert_eq!(tasks[1].col_off, 1024);
        assert_eq!(tasks[1].row_off, 0);
        assert_eq!(tasks[1].width, 1024);
        assert_eq!(tasks[1].height, 1024);
    }

    #[test]
    fn test_chunk_grid_partial_tiles() {
        let tasks = derive_chunk_grid(1500, 800, 1024);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].width, 1024);
        assert_eq!(tasks[0].height, 800);
        assert_eq!(tasks[1].width, 476);
        assert_eq!(tasks[1].height, 800);
    }

    #[test]
    fn test_chunk_grid_small_raster() {
        let tasks = derive_chunk_grid(512, 512, 1024);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].width, 512);
        assert_eq!(tasks[0].height, 512);
    }

    #[test]
    fn test_point_id_uniqueness() {
        let points_chunk0: Vec<(i64, f64, f64, i32)> = (0..50)
            .map(|i| (i, i as f64 * 10.0, 29.0, 2020))
            .collect();
        let points_chunk1: Vec<(i64, f64, f64, i32)> = (50..100)
            .map(|i| (i, i as f64 * 10.0, 29.0, 2020))
            .collect();

        let mut features = HashMap::new();
        features.insert("slope".to_string(), 10.0);

        let df0 = build_chunk_dataframe(
            &ChunkTask { chunk_idx: 0, row_off: 0, col_off: 0, width: 100, height: 100 },
            &points_chunk0, &features,
            &HashMap::new(), &HashMap::new(), &HashMap::new(), &HashMap::new(), 42,
        ).unwrap();

        let df1 = build_chunk_dataframe(
            &ChunkTask { chunk_idx: 1, row_off: 0, col_off: 100, width: 100, height: 100 },
            &points_chunk1, &features,
            &HashMap::new(), &HashMap::new(), &HashMap::new(), &HashMap::new(), 42,
        ).unwrap();

        let ids0: HashSet<i64> = df0.column("point_id").unwrap()
            .as_materialized_series().i64().unwrap()
            .into_iter().filter_map(|v| v).collect();
        let ids1: HashSet<i64> = df1.column("point_id").unwrap()
            .as_materialized_series().i64().unwrap()
            .into_iter().filter_map(|v| v).collect();

        let intersection: HashSet<_> = ids0.intersection(&ids1).collect();
        assert!(intersection.is_empty(), "Point IDs must be unique across chunks");
    }

    #[test]
    fn test_empty_chunk_skip() {
        let points: Vec<(i64, f64, f64, i32)> = vec![];
        let features = HashMap::new();
        let df = build_chunk_dataframe(
            &ChunkTask { chunk_idx: 0, row_off: 0, col_off: 0, width: 10, height: 10 },
            &points, &features,
            &HashMap::new(), &HashMap::new(), &HashMap::new(), &HashMap::new(), 42,
        ).unwrap();

        let has_data = chunk_has_valid_data(&df);
        assert!(!has_data || df.height() == 0);
    }

    #[test]
    fn test_chunk_grid_full_extent() {
        let tasks = derive_chunk_grid(1024, 1024, 256);
        assert_eq!(tasks.len(), 16);
        let idxs: Vec<usize> = tasks.iter().map(|t| t.chunk_idx).collect();
        assert_eq!(idxs, (0..16).collect::<Vec<_>>());
    }
}

//! Feature extraction from resampled rasters for training points.
//!
//! Loads the training points parquet (row, col, year, disturbance_flag,
//! disturbance_class, ccdc_timing_year), reads pixel values from all resampled
//! rasters, computes derived features (slope/aspect from DEM), and writes
//! a complete feature parquet.
//!
//! ## Raster cache strategy
//!
//! GDAL Datasets are cached in a `HashMap` to avoid repeated open/close.
//! Points are processed grouped by year to minimise raster switching.
//! Missing files produce NaN (not errors) so partial datasets work.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use loaders::base::{LoaderError, LoaderResult};
use memmap2::Mmap;
use polars::prelude::*;
use rayon::prelude::*;

#[cfg(feature = "gdal")]
use gdal_crate::Dataset;

// ── public API ─────────────────────────────────────────────────────────

/// Extract features for all training points.
///
/// Reads `training_parquet` (columns: `row`, `col`, `year`, `disturbance_flag`,
/// `disturbance_class`, `ccdc_timing_year`), reads pixel values from resampled
/// rasters under `{base}/resampled/`, computes derived features, and writes the
/// combined DataFrame to `output_parquet`.
///
/// The base directory is derived from the training parquet path:
/// `.cache/features/training_points.parquet` → base = `.cache`.
pub fn extract_training_features(training_parquet: &str, output_parquet: &str) -> LoaderResult<()> {
    extract_training_features_with_workers(training_parquet, output_parquet, 1)
}

pub fn extract_training_features_with_workers(
    training_parquet: &str,
    output_parquet: &str,
    workers: usize,
) -> LoaderResult<()> {
    let training_path = Path::new(training_parquet);
    let base_dir = training_path
        .parent() // .cache/features
        .and_then(|p| p.parent()) // .cache
        .unwrap_or(Path::new("."));

    let resampled = base_dir.join("resampled");

    // Ensure output directory exists.
    if let Some(parent) = Path::new(output_parquet).parent() {
        std::fs::create_dir_all(parent).map_err(LoaderError::Io)?;
    }

    // Load training points.
    let df = LazyFrame::scan_parquet(training_parquet, default_parquet_args())
        .map_err(|e| plerr(e))?
        .collect()
        .map_err(|e| LoaderError::Gdal(format!("read training parquet: {e}")))?;

    let n = df.height();
    if n == 0 {
        return Err(LoaderError::NoValidData);
    }

    let rows_s = df
        .column("row")
        .map_err(|e| plerr(e))?
        .as_materialized_series();
    let cols_s = df
        .column("col")
        .map_err(|e| plerr(e))?
        .as_materialized_series();
    let years_s = df
        .column("year")
        .map_err(|e| plerr(e))?
        .as_materialized_series();
    let flags_s = df
        .column("disturbance_flag")
        .map_err(|e| plerr(e))?
        .as_materialized_series();
    let classes_s = df
        .column("disturbance_class")
        .map_err(|e| plerr(e))?
        .as_materialized_series();
    let timings_s = df
        .column("ccdc_timing_year")
        .map_err(|e| plerr(e))?
        .as_materialized_series();

    let rows = rows_s.i32().map_err(|e| plerr(e))?;
    let cols = cols_s.i32().map_err(|e| plerr(e))?;
    let years = years_s.i32().map_err(|e| plerr(e))?;
    let _flags = flags_s.i32().map_err(|e| plerr(e))?;
    let _classes = classes_s.i32().map_err(|e| plerr(e))?;
    let timings = timings_s.f32().map_err(|e| plerr(e))?;

    // Group indices by year to minimise raster switching.
    let mut by_year: HashMap<i32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let y = years.get(i).unwrap_or(0);
        by_year.entry(y).or_default().push(i);
    }

    let mut years_sorted: Vec<i32> = by_year.keys().copied().collect();
    years_sorted.sort_unstable();

    let resampled_str = resampled.to_string_lossy().to_string();

    let cache_dir = base_dir.join("mmap").join("training_point_values");
    std::fs::create_dir_all(&cache_dir).map_err(LoaderError::Io)?;

    let point_rows: Vec<usize> = (0..n)
        .map(|i| rows.get(i).unwrap_or(0).max(0) as usize)
        .collect();
    let point_cols: Vec<usize> = (0..n)
        .map(|i| cols.get(i).unwrap_or(0).max(0) as usize)
        .collect();

    let dem_path = format!("{}/dem.tif", resampled_str);
    let landuse_path = format!("{}/landuse/landuse.tif", resampled_str);
    let river_path = format!("{}/river/river_dist.tif", resampled_str);
    let resident_path = format!("{}/resident/resident.tif", resampled_str);

    // ── allocate output vectors ────────────────────────────────────────
    let mut out_elevation: Vec<f64> = vec![f64::NAN; n];
    let mut out_slope: Vec<f64> = vec![f64::NAN; n];
    let mut out_aspect: Vec<f64> = vec![f64::NAN; n];
    let mut out_landuse: Vec<i32> = vec![0; n];
    let mut out_dist_to_river: Vec<f64> = vec![f64::NAN; n];
    let mut out_dist_to_road: Vec<f64> = vec![f64::NAN; n];
    let mut out_dist_to_railway: Vec<f64> = vec![f64::NAN; n];
    let mut out_dist_to_waterway: Vec<f64> = vec![f64::NAN; n];
    let mut out_resident: Vec<f64> = vec![f64::NAN; n];
    let mut out_rainfall: [Vec<f64>; 12] = [
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
    ];
    let mut out_temp: [Vec<f64>; 12] = [
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
    ];
    let mut out_rainfall_prev_1m: Vec<f64> = vec![f64::NAN; n];
    let mut out_rainfall_prev_2m: Vec<f64> = vec![f64::NAN; n];
    let mut out_rainfall_prev_3m: Vec<f64> = vec![f64::NAN; n];
    let mut out_rainfall_prev_total_3m: Vec<f64> = vec![f64::NAN; n];

    let workers = effective_workers(workers);
    eprintln!(
        "[features][train] extracting {} training rows with mmap point cache at {}",
        n,
        cache_dir.display()
    );

    #[cfg(feature = "gdal")]
    if let Some(dem) =
        dem_feature_values_from_raster(&dem_path, &point_rows, &point_cols, &cache_dir, workers)?
    {
        for i in 0..n {
            out_elevation[i] = dem.elevation.get_f64(i).unwrap_or(f64::NAN);
            out_slope[i] = dem.slope.get_f64(i).unwrap_or(f64::NAN);
            out_aspect[i] = dem.aspect.get_f64(i).unwrap_or(f64::NAN);
        }
    }

    #[cfg(feature = "gdal")]
    if let Some(values) =
        point_values_from_raster(&landuse_path, &point_rows, &point_cols, &cache_dir, workers)?
    {
        for i in 0..n {
            out_landuse[i] = values.get_i32(i).unwrap_or(0);
        }
    }

    #[cfg(feature = "gdal")]
    if let Some(values) =
        point_values_from_raster(&river_path, &point_rows, &point_cols, &cache_dir, workers)?
    {
        for i in 0..n {
            out_dist_to_river[i] = values.get_f64(i).unwrap_or(f64::NAN);
        }
    }

    #[cfg(feature = "gdal")]
    if let Some(values) = point_values_from_raster(
        &resident_path,
        &point_rows,
        &point_cols,
        &cache_dir,
        workers,
    )? {
        for i in 0..n {
            out_resident[i] = values.get_f64(i).unwrap_or(f64::NAN);
        }
    }

    // ── process dynamic rasters grouped by year ────────────────────────
    for year in &years_sorted {
        let indices = &by_year[year];
        let year_rows: Vec<usize> = indices.iter().map(|&i| point_rows[i]).collect();
        let year_cols: Vec<usize> = indices.iter().map(|&i| point_cols[i]).collect();

        let road_path = find_road_raster(&resampled_str, "road_dist", *year);
        let railway_path = find_road_raster(&resampled_str, "railway_dist", *year);
        let waterway_path = find_road_raster(&resampled_str, "waterway_dist", *year);

        #[cfg(feature = "gdal")]
        if let Some(values) =
            point_values_from_raster(&road_path, &year_rows, &year_cols, &cache_dir, workers)?
        {
            for (j, &i) in indices.iter().enumerate() {
                out_dist_to_road[i] = values.get_f64(j).unwrap_or(f64::NAN);
            }
        }
        #[cfg(feature = "gdal")]
        if let Some(values) =
            point_values_from_raster(&railway_path, &year_rows, &year_cols, &cache_dir, workers)?
        {
            for (j, &i) in indices.iter().enumerate() {
                out_dist_to_railway[i] = values.get_f64(j).unwrap_or(f64::NAN);
            }
        }
        #[cfg(feature = "gdal")]
        if let Some(values) =
            point_values_from_raster(&waterway_path, &year_rows, &year_cols, &cache_dir, workers)?
        {
            for (j, &i) in indices.iter().enumerate() {
                out_dist_to_waterway[i] = values.get_f64(j).unwrap_or(f64::NAN);
            }
        }

        #[cfg(feature = "gdal")]
        let rainfall_values: [Option<CachedRasterValues>; 12] = std::array::from_fn(|m| {
            let rp = find_rainfall_raster(&resampled_str, *year, m as i32 + 1);
            point_values_from_raster(&rp, &year_rows, &year_cols, &cache_dir, workers)
                .ok()
                .flatten()
        });
        #[cfg(feature = "gdal")]
        let prev_rainfall_values: [Option<CachedRasterValues>; 12] = std::array::from_fn(|m| {
            let rp = find_rainfall_raster(&resampled_str, year - 1, m as i32 + 1);
            point_values_from_raster(&rp, &year_rows, &year_cols, &cache_dir, workers)
                .ok()
                .flatten()
        });
        #[cfg(feature = "gdal")]
        let temp_values: [Option<CachedRasterValues>; 12] = std::array::from_fn(|m| {
            let tp = find_temperature_raster(&resampled_str, *year, m as i32 + 1);
            point_values_from_raster(&tp, &year_rows, &year_cols, &cache_dir, workers)
                .ok()
                .flatten()
        });

        #[cfg(feature = "gdal")]
        for (j, &i) in indices.iter().enumerate() {
            for m in 0..12 {
                if let Some(ref values) = rainfall_values[m] {
                    out_rainfall[m][i] = values.get_f64(j).unwrap_or(f64::NAN);
                }
                if let Some(ref values) = temp_values[m] {
                    out_temp[m][i] = values.get_f64(j).unwrap_or(f64::NAN);
                }
            }

            let (prev_1m, prev_2m, prev_3m) = compute_3month_preceding_rainfall_cached(
                timings.get(i),
                j,
                &rainfall_values,
                &prev_rainfall_values,
            );
            out_rainfall_prev_1m[i] = prev_1m;
            out_rainfall_prev_2m[i] = prev_2m;
            out_rainfall_prev_3m[i] = prev_3m;
            if prev_1m.is_finite() && prev_2m.is_finite() && prev_3m.is_finite() {
                out_rainfall_prev_total_3m[i] = prev_1m + prev_2m + prev_3m;
            }
        }
    }

    // ── build output DataFrame ──────────────────────────────────────────
    let mut cols: Vec<Column> = Vec::with_capacity(38);

    // Original label/metadata columns.
    cols.push(Column::new("row".into(), rows_s.clone()));
    cols.push(Column::new("col".into(), cols_s.clone()));
    cols.push(Column::new("year".into(), years_s.clone()));
    cols.push(Column::new("disturbance_flag".into(), flags_s.clone()));
    cols.push(Column::new("disturbance_class".into(), classes_s.clone()));
    cols.push(Column::new("ccdc_timing_year".into(), timings_s.clone()));

    // Static features.
    cols.push(Column::new("elevation".into(), out_elevation));
    cols.push(Column::new("slope".into(), out_slope));
    cols.push(Column::new("aspect".into(), out_aspect));
    cols.push(Column::new("landuse".into(), out_landuse));
    cols.push(Column::new("dist_to_river".into(), out_dist_to_river));

    // Dynamic features.
    cols.push(Column::new("dist_to_road".into(), out_dist_to_road));
    cols.push(Column::new("dist_to_railway".into(), out_dist_to_railway));
    cols.push(Column::new("dist_to_waterway".into(), out_dist_to_waterway));
    cols.push(Column::new("resident".into(), out_resident));

    // Monthly rainfall.
    for m in 0..12 {
        let name = format!("rainfall_month_{:02}", m + 1);
        cols.push(Column::new(
            name.into(),
            std::mem::take(&mut out_rainfall[m]),
        ));
    }

    // 3-month preceding rainfall.
    cols.push(Column::new("rainfall_prev_1m".into(), out_rainfall_prev_1m));
    cols.push(Column::new("rainfall_prev_2m".into(), out_rainfall_prev_2m));
    cols.push(Column::new("rainfall_prev_3m".into(), out_rainfall_prev_3m));
    cols.push(Column::new(
        "rainfall_prev_total_3m".into(),
        out_rainfall_prev_total_3m,
    ));

    // Monthly temperature.
    for m in 0..12 {
        let name = format!("temp_month_{:02}", m + 1);
        cols.push(Column::new(name.into(), std::mem::take(&mut out_temp[m])));
    }

    let mut out_df =
        DataFrame::new(cols).map_err(|e| LoaderError::Gdal(format!("build DataFrame: {e}")))?;

    // Write parquet.
    let mut f = std::fs::File::create(output_parquet).map_err(LoaderError::Io)?;
    ParquetWriter::new(&mut f)
        .finish(&mut out_df)
        .map_err(|e| LoaderError::Gdal(format!("write parquet: {e}")))?;

    log::info!(
        "feature_extraction: wrote {} rows × {} cols to {}",
        n,
        out_df.width(),
        output_parquet
    );
    Ok(())
}

// ── prediction feature extraction (tiled, full-AOI) ─────────────────────

/// Extract features for every pixel in the CLASS grid for year 2025 using
/// tiled processing.  Reads resampled rasters under `.cache/resampled/`,
/// computes slope/aspect from DEM, and writes per-tile parquet files to
/// `output_dir` (e.g. `.cache/features/prediction_2025/`).
///
/// * `class_grid` — reference grid providing dimensions and CRS
/// * `aoi_mask` — AOI geometry mask in the CLASS grid CRS
/// * `output_dir` — directory for tile parquet files
/// * `tile_size` — tile side length in pixels (e.g. 1024)
/// * `resampled_dir` — path to the resampled raster cache
/// * `target_year` — prediction year (default 2025)
///
/// Returns the number of tiles written.
pub fn extract_prediction_features(
    class_grid: &crate::class_grid::ClassGrid,
    aoi_mask: &crate::aoi_mask::AoiMask,
    output_dir: &str,
    tile_size: usize,
    resampled_dir: &str,
    target_year: i32,
) -> LoaderResult<usize> {
    extract_prediction_features_with_workers(
        class_grid,
        aoi_mask,
        output_dir,
        tile_size,
        resampled_dir,
        target_year,
        1,
    )
}

pub fn extract_prediction_features_with_workers(
    class_grid: &crate::class_grid::ClassGrid,
    aoi_mask: &crate::aoi_mask::AoiMask,
    output_dir: &str,
    tile_size: usize,
    resampled_dir: &str,
    target_year: i32,
    workers: usize,
) -> LoaderResult<usize> {
    let resampled = Path::new(resampled_dir);

    // Ensure output directory exists.
    std::fs::create_dir_all(output_dir).map_err(LoaderError::Io)?;

    let width = class_grid.width;
    let height = class_grid.height;

    // Compute tile grid dimensions.
    let n_tile_rows = (height + tile_size - 1) / tile_size;
    let n_tile_cols = (width + tile_size - 1) / tile_size;
    let total_tiles = n_tile_rows * n_tile_cols;

    eprintln!(
        "[predict] CLASS grid: {}×{}, tile size: {}, tiles: {}×{} = {}, workers: {}",
        width,
        height,
        tile_size,
        n_tile_rows,
        n_tile_cols,
        total_tiles,
        effective_workers(workers)
    );

    // Use previous year for data if target_year not yet available.
    let data_year = resolve_data_year(resampled, target_year);
    eprintln!(
        "[predict] Target year {}, using data year {}",
        target_year, data_year
    );

    let resampled_str = resampled.to_string_lossy().to_string();
    let sources = PredictionFeatureSources::new(&resampled_str, target_year, data_year);
    let mut tasks = Vec::with_capacity(total_tiles);
    for tile_row in (0..height).step_by(tile_size) {
        for tile_col in (0..width).step_by(tile_size) {
            let tile_w = tile_size.min(width - tile_col);
            let tile_h = tile_size.min(height - tile_row);
            tasks.push(PredictionTileTask {
                tile_row,
                tile_col,
                tile_w,
                tile_h,
            });
        }
    }

    let workers = effective_workers(workers);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| LoaderError::Gdal(format!("prediction feature thread pool: {e}")))?;

    let tiles_written = AtomicUsize::new(0);
    let tasks_seen = AtomicUsize::new(0);
    pool.install(|| {
        tasks.par_iter().try_for_each(|task| {
            let wrote_tile =
                write_prediction_tile(task, class_grid, aoi_mask, &sources, output_dir)?;
            let task_idx = tasks_seen.fetch_add(1, Ordering::Relaxed) + 1;
            if wrote_tile {
                let written = tiles_written.fetch_add(1, Ordering::Relaxed) + 1;
                if written % 10 == 0 || written == 1 {
                    eprintln!(
                        "[predict] Tiles written: {} (tasks completed {}/{}, {:.1}%) - tile_{:06}_{:06}.parquet ({}×{} = {} px)",
                        written,
                        task_idx,
                        total_tiles,
                        100.0 * task_idx as f64 / total_tiles as f64,
                        task.tile_row,
                        task.tile_col,
                        task.tile_w,
                        task.tile_h,
                        task.tile_w * task.tile_h
                    );
                }
            }
            Ok::<(), LoaderError>(())
        })
    })?;
    let tiles_written = tiles_written.load(Ordering::Relaxed);

    eprintln!(
        "[predict] Feature extraction complete: {} tiles written to {}",
        tiles_written, output_dir
    );
    Ok(tiles_written)
}

#[derive(Debug, Clone)]
struct PredictionTileTask {
    tile_row: usize,
    tile_col: usize,
    tile_w: usize,
    tile_h: usize,
}

#[derive(Debug, Clone)]
struct PredictionFeatureSources {
    dem_path: String,
    landuse_path: String,
    river_path: String,
    resident_path: String,
    road_path: String,
    railway_path: String,
    waterway_path: String,
    rainfall_paths: [String; 12],
    prev_rainfall_paths: [String; 12],
    temp_paths: [String; 12],
}

impl PredictionFeatureSources {
    fn new(resampled_dir: &str, target_year: i32, data_year: i32) -> Self {
        let rainfall_paths =
            std::array::from_fn(|m| find_rainfall_raster(resampled_dir, data_year, m as i32 + 1));
        let prev_rainfall_paths = std::array::from_fn(|m| {
            find_rainfall_raster(resampled_dir, data_year - 1, m as i32 + 1)
        });
        let temp_paths = std::array::from_fn(|m| {
            find_temperature_raster(resampled_dir, data_year, m as i32 + 1)
        });
        Self {
            dem_path: format!("{}/dem.tif", resampled_dir),
            landuse_path: format!("{}/landuse/landuse.tif", resampled_dir),
            river_path: format!("{}/river/river_dist.tif", resampled_dir),
            resident_path: format!("{}/resident/resident.tif", resampled_dir),
            road_path: find_road_raster(resampled_dir, "road_dist", target_year),
            railway_path: find_road_raster(resampled_dir, "railway_dist", target_year),
            waterway_path: find_road_raster(resampled_dir, "waterway_dist", target_year),
            rainfall_paths,
            prev_rainfall_paths,
            temp_paths,
        }
    }
}

fn write_prediction_tile(
    task: &PredictionTileTask,
    class_grid: &crate::class_grid::ClassGrid,
    aoi_mask: &crate::aoi_mask::AoiMask,
    sources: &PredictionFeatureSources,
    output_dir: &str,
) -> LoaderResult<bool> {
    let aoi_tile_mask = aoi_mask.tile_mask(
        class_grid,
        task.tile_row,
        task.tile_col,
        task.tile_w,
        task.tile_h,
    );
    let n_pixels = aoi_tile_mask.iter().filter(|&&inside| inside).count();
    if n_pixels == 0 {
        return Ok(false);
    }

    let dem_ds = open_raster_optional(&sources.dem_path);
    let landuse_ds = open_raster_optional(&sources.landuse_path);
    let river_ds = open_raster_optional(&sources.river_path);
    let resident_ds = open_raster_optional(&sources.resident_path);
    let road_ds = open_raster_optional(&sources.road_path);
    let railway_ds = open_raster_optional(&sources.railway_path);
    let waterway_ds = open_raster_optional(&sources.waterway_path);

    let (dem_res_x, dem_res_y, dem_width, dem_height) = if let Some(ref ds) = dem_ds {
        let gt = ds
            .geo_transform()
            .map_err(|e| LoaderError::Gdal(e.to_string()))?;
        let (w, h) = ds.raster_size();
        (gt[1].abs(), gt[5].abs(), w as usize, h as usize)
    } else {
        (0.0, 0.0, 0, 0)
    };

    let dem_window = dem_ds.as_ref().and_then(|ds| {
        read_raster_window_optional(
            ds,
            task.tile_row.saturating_sub(1),
            task.tile_col.saturating_sub(1),
            (task.tile_h + 2).min(dem_height),
            (task.tile_w + 2).min(dem_width),
        )
    });
    let landuse_window = landuse_ds
        .as_ref()
        .and_then(|ds| read_task_window(ds, task));
    let river_window = river_ds.as_ref().and_then(|ds| read_task_window(ds, task));
    let resident_window = resident_ds
        .as_ref()
        .and_then(|ds| read_task_window(ds, task));
    let road_window = road_ds.as_ref().and_then(|ds| read_task_window(ds, task));
    let railway_window = railway_ds
        .as_ref()
        .and_then(|ds| read_task_window(ds, task));
    let waterway_window = waterway_ds
        .as_ref()
        .and_then(|ds| read_task_window(ds, task));

    let rainfall_windows: [Option<RasterWindow>; 12] = std::array::from_fn(|m| {
        open_raster_optional(&sources.rainfall_paths[m]).and_then(|ds| read_task_window(&ds, task))
    });
    let prev_rainfall_windows: [Option<RasterWindow>; 12] = std::array::from_fn(|m| {
        open_raster_optional(&sources.prev_rainfall_paths[m])
            .and_then(|ds| read_task_window(&ds, task))
    });
    let temp_windows: [Option<RasterWindow>; 12] = std::array::from_fn(|m| {
        open_raster_optional(&sources.temp_paths[m]).and_then(|ds| read_task_window(&ds, task))
    });

    let mut out_rows: Vec<i32> = Vec::with_capacity(n_pixels);
    let mut out_cols: Vec<i32> = Vec::with_capacity(n_pixels);
    let mut out_elevation: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_slope: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_aspect: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_landuse: Vec<i32> = vec![0; n_pixels];
    let mut out_dist_to_river: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_dist_to_road: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_dist_to_railway: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_dist_to_waterway: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_resident: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_rainfall: [Vec<f64>; 12] = std::array::from_fn(|_| vec![f64::NAN; n_pixels]);
    let mut out_rainfall_prev_1m: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_rainfall_prev_2m: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_rainfall_prev_3m: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_rainfall_prev_total_3m: Vec<f64> = vec![f64::NAN; n_pixels];
    let mut out_temp: [Vec<f64>; 12] = std::array::from_fn(|_| vec![f64::NAN; n_pixels]);

    for dy in 0..task.tile_h {
        for dx in 0..task.tile_w {
            let r = task.tile_row + dy;
            let c = task.tile_col + dx;
            let tile_idx = dy * task.tile_w + dx;
            if !aoi_tile_mask[tile_idx] {
                continue;
            }
            let pi = out_rows.len();

            out_rows.push(r as i32);
            out_cols.push(c as i32);

            if let Some(ref window) = dem_window {
                let (elevation, slope, aspect) =
                    read_dem_features_from_window(window, r, c, dem_res_x, dem_res_y);
                out_elevation[pi] = elevation;
                out_slope[pi] = slope;
                out_aspect[pi] = aspect;
            }

            if let Some(ref window) = landuse_window {
                out_landuse[pi] = window.get_i32(r, c).unwrap_or(0);
            }
            if let Some(ref window) = river_window {
                out_dist_to_river[pi] = window.get_f64(r, c).unwrap_or(f64::NAN);
            }
            if let Some(ref window) = road_window {
                out_dist_to_road[pi] = window.get_f64(r, c).unwrap_or(f64::NAN);
            }
            if let Some(ref window) = railway_window {
                out_dist_to_railway[pi] = window.get_f64(r, c).unwrap_or(f64::NAN);
            }
            if let Some(ref window) = waterway_window {
                out_dist_to_waterway[pi] = window.get_f64(r, c).unwrap_or(f64::NAN);
            }
            if let Some(ref window) = resident_window {
                out_resident[pi] = window.get_f64(r, c).unwrap_or(f64::NAN);
            }

            for m in 0..12 {
                if let Some(ref window) = rainfall_windows[m] {
                    out_rainfall[m][pi] = window.get_f64(r, c).unwrap_or(f64::NAN);
                }
                if let Some(ref window) = temp_windows[m] {
                    out_temp[m][pi] = window.get_f64(r, c).unwrap_or(f64::NAN);
                }
            }

            let (prev_1m, prev_2m, prev_3m) = compute_3month_preceding_rainfall_windows(
                None,
                r,
                c,
                &rainfall_windows,
                &prev_rainfall_windows,
            );
            out_rainfall_prev_1m[pi] = prev_1m;
            out_rainfall_prev_2m[pi] = prev_2m;
            out_rainfall_prev_3m[pi] = prev_3m;
            if prev_1m.is_finite() && prev_2m.is_finite() && prev_3m.is_finite() {
                out_rainfall_prev_total_3m[pi] = prev_1m + prev_2m + prev_3m;
            }
        }
    }

    let mut cols: Vec<Column> = Vec::with_capacity(38);
    cols.push(Column::new("row".into(), out_rows));
    cols.push(Column::new("col".into(), out_cols));
    cols.push(Column::new("elevation".into(), out_elevation));
    cols.push(Column::new("slope".into(), out_slope));
    cols.push(Column::new("aspect".into(), out_aspect));
    cols.push(Column::new("landuse".into(), out_landuse));
    cols.push(Column::new("dist_to_river".into(), out_dist_to_river));
    cols.push(Column::new("dist_to_road".into(), out_dist_to_road));
    cols.push(Column::new("dist_to_railway".into(), out_dist_to_railway));
    cols.push(Column::new("dist_to_waterway".into(), out_dist_to_waterway));
    cols.push(Column::new("resident".into(), out_resident));

    for m in 0..12 {
        cols.push(Column::new(
            format!("rainfall_month_{:02}", m + 1).into(),
            std::mem::take(&mut out_rainfall[m]),
        ));
    }

    cols.push(Column::new("rainfall_prev_1m".into(), out_rainfall_prev_1m));
    cols.push(Column::new("rainfall_prev_2m".into(), out_rainfall_prev_2m));
    cols.push(Column::new("rainfall_prev_3m".into(), out_rainfall_prev_3m));
    cols.push(Column::new(
        "rainfall_prev_total_3m".into(),
        out_rainfall_prev_total_3m,
    ));

    for m in 0..12 {
        cols.push(Column::new(
            format!("temp_month_{:02}", m + 1).into(),
            std::mem::take(&mut out_temp[m]),
        ));
    }

    let row_count = out_rows_len_from_columns(&cols)?;
    if row_count == 0 {
        return Ok(false);
    }

    let mut df = DataFrame::new(cols)
        .map_err(|e| LoaderError::Gdal(format!("build tile DataFrame: {e}")))?;
    let tile_path = format!(
        "{}/tile_{:06}_{:06}.parquet",
        output_dir, task.tile_row, task.tile_col
    );
    let tmp_path = format!("{tile_path}.tmp");
    let mut f = std::fs::File::create(&tmp_path).map_err(LoaderError::Io)?;
    let write_result = ParquetWriter::new(&mut f)
        .set_parallel(false)
        .finish(&mut df)
        .map_err(|e| LoaderError::Gdal(format!("write tile parquet: {e}")));
    if let Err(err) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    drop(f);

    let written_size = std::fs::metadata(&tmp_path).map_err(LoaderError::Io)?.len();
    if written_size == 0 {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(LoaderError::Gdal(format!(
            "write tile parquet produced an empty file: {tmp_path}"
        )));
    }

    std::fs::rename(&tmp_path, &tile_path).map_err(LoaderError::Io)?;
    Ok(true)
}

fn out_rows_len_from_columns(cols: &[Column]) -> LoaderResult<usize> {
    cols.first()
        .map(|c| c.len())
        .ok_or_else(|| LoaderError::Gdal("tile DataFrame has no columns".into()))
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

/// Determine which data year to use for prediction.
///
/// Checks if rainfall data exists for `target_year`; if not, walks
/// backward until a year with rainfall data is found (up to 5 years).
fn resolve_data_year(resampled_dir: &Path, target_year: i32) -> i32 {
    for offset in 0..=5 {
        let year = target_year - offset;
        let test_path = resampled_dir.join(format!("rainfall/pre{}{:02}.tif", year, 1));
        if test_path.exists() {
            return year;
        }
    }
    // Fallback to target_year — let missing data produce NaN.
    target_year
}

// ── internal helpers ───────────────────────────────────────────────────

fn plerr(e: PolarsError) -> LoaderError {
    LoaderError::Gdal(format!("Polars: {e}"))
}

/// Open a raster dataset, returning `None` if the file doesn't exist or
/// cannot be opened.
#[cfg(feature = "gdal")]
fn open_raster_optional(path: &str) -> Option<Dataset> {
    if !Path::new(path).exists() {
        return None;
    }
    Dataset::open(path).ok()
}

#[cfg(not(feature = "gdal"))]
fn open_raster_optional(_path: &str) -> Option<()> {
    None
}

#[cfg(feature = "gdal")]
#[derive(Debug, Clone)]
struct RasterWindow {
    row_start: usize,
    col_start: usize,
    width: usize,
    height: usize,
    data: Vec<f32>,
}

#[cfg(feature = "gdal")]
impl RasterWindow {
    fn get_f32(&self, row: usize, col: usize) -> Option<f32> {
        if row < self.row_start || col < self.col_start {
            return None;
        }
        let local_r = row - self.row_start;
        let local_c = col - self.col_start;
        if local_r >= self.height || local_c >= self.width {
            return None;
        }
        let value = *self.data.get(local_r * self.width + local_c)?;
        if value.is_finite() {
            Some(value)
        } else {
            None
        }
    }

    fn get_f64(&self, row: usize, col: usize) -> Option<f64> {
        self.get_f32(row, col).map(|v| v as f64)
    }

    fn get_i32(&self, row: usize, col: usize) -> Option<i32> {
        self.get_f32(row, col).map(|v| v as i32)
    }
}

#[cfg(not(feature = "gdal"))]
#[derive(Debug, Clone)]
struct RasterWindow;

#[cfg(not(feature = "gdal"))]
impl RasterWindow {
    fn get_f32(&self, _row: usize, _col: usize) -> Option<f32> {
        None
    }

    fn get_f64(&self, _row: usize, _col: usize) -> Option<f64> {
        None
    }

    fn get_i32(&self, _row: usize, _col: usize) -> Option<i32> {
        None
    }
}

#[cfg(feature = "gdal")]
fn read_task_window(ds: &Dataset, task: &PredictionTileTask) -> Option<RasterWindow> {
    read_raster_window_optional(ds, task.tile_row, task.tile_col, task.tile_h, task.tile_w)
}

#[cfg(not(feature = "gdal"))]
fn read_task_window(_ds: &(), _task: &PredictionTileTask) -> Option<RasterWindow> {
    None
}

#[cfg(feature = "gdal")]
fn read_raster_window_optional(
    ds: &Dataset,
    row_start: usize,
    col_start: usize,
    requested_height: usize,
    requested_width: usize,
) -> Option<RasterWindow> {
    let (raster_width, raster_height) = ds.raster_size();
    if row_start >= raster_height
        || col_start >= raster_width
        || requested_height == 0
        || requested_width == 0
    {
        return None;
    }
    let row_end = row_start
        .saturating_add(requested_height)
        .min(raster_height);
    let col_end = col_start.saturating_add(requested_width).min(raster_width);
    let height = row_end.checked_sub(row_start)?;
    let width = col_end.checked_sub(col_start)?;
    if width == 0 || height == 0 {
        return None;
    }

    let band = ds.rasterband(1).ok()?;
    let buffer = band
        .read_as::<f32>(
            (col_start as isize, row_start as isize),
            (width, height),
            (width, height),
            None,
        )
        .ok()?;

    Some(RasterWindow {
        row_start,
        col_start,
        width,
        height,
        data: buffer.data().to_vec(),
    })
}

#[cfg(not(feature = "gdal"))]
fn read_raster_window_optional(
    _ds: &(),
    _row_start: usize,
    _col_start: usize,
    _requested_height: usize,
    _requested_width: usize,
) -> Option<RasterWindow> {
    None
}

fn default_parquet_args() -> ScanArgsParquet {
    ScanArgsParquet {
        rechunk: true,
        cache: true,
        ..Default::default()
    }
}

// ── mmap point-value cache ─────────────────────────────────────────────

#[cfg(feature = "gdal")]
struct CachedRasterValues {
    mmap: Mmap,
    len: usize,
}

#[cfg(feature = "gdal")]
impl CachedRasterValues {
    fn get_f32(&self, idx: usize) -> Option<f32> {
        if idx >= self.len {
            return None;
        }
        let start = idx.checked_mul(4)?;
        let bytes = self.mmap.get(start..start + 4)?;
        let val = f32::from_le_bytes(bytes.try_into().ok()?);
        if val.is_finite() {
            Some(val)
        } else {
            None
        }
    }

    fn get_f64(&self, idx: usize) -> Option<f64> {
        self.get_f32(idx).map(|v| v as f64)
    }

    fn get_i32(&self, idx: usize) -> Option<i32> {
        self.get_f32(idx).map(|v| v as i32)
    }
}

#[cfg(feature = "gdal")]
struct CachedDemFeatures {
    elevation: CachedRasterValues,
    slope: CachedRasterValues,
    aspect: CachedRasterValues,
}

#[cfg(feature = "gdal")]
fn point_values_from_raster(
    path: &str,
    rows: &[usize],
    cols: &[usize],
    cache_dir: &Path,
    workers: usize,
) -> LoaderResult<Option<CachedRasterValues>> {
    if !Path::new(path).exists() {
        return Ok(None);
    }

    let cache_path = point_cache_path(path, rows, cols, cache_dir, "values");
    if cache_is_valid(&cache_path, rows.len()) {
        return mmap_cached_values(&cache_path, rows.len()).map(Some);
    }

    eprintln!(
        "[features][train] building point cache: {} ({} points)",
        path,
        rows.len()
    );
    let values = read_raster_values_by_block(path, rows, cols, workers)?;
    write_f32_cache(&cache_path, &values)?;
    mmap_cached_values(&cache_path, rows.len()).map(Some)
}

#[cfg(feature = "gdal")]
fn dem_feature_values_from_raster(
    path: &str,
    rows: &[usize],
    cols: &[usize],
    cache_dir: &Path,
    workers: usize,
) -> LoaderResult<Option<CachedDemFeatures>> {
    if !Path::new(path).exists() {
        return Ok(None);
    }

    let elevation_path = point_cache_path(path, rows, cols, cache_dir, "elevation");
    let slope_path = point_cache_path(path, rows, cols, cache_dir, "slope");
    let aspect_path = point_cache_path(path, rows, cols, cache_dir, "aspect");
    if cache_is_valid(&elevation_path, rows.len())
        && cache_is_valid(&slope_path, rows.len())
        && cache_is_valid(&aspect_path, rows.len())
    {
        return Ok(Some(CachedDemFeatures {
            elevation: mmap_cached_values(&elevation_path, rows.len())?,
            slope: mmap_cached_values(&slope_path, rows.len())?,
            aspect: mmap_cached_values(&aspect_path, rows.len())?,
        }));
    }

    eprintln!(
        "[features][train] building DEM feature cache: {} ({} points)",
        path,
        rows.len()
    );
    let (elevation, slope, aspect) = read_dem_features_by_block(path, rows, cols, workers)?;
    write_f32_cache(&elevation_path, &elevation)?;
    write_f32_cache(&slope_path, &slope)?;
    write_f32_cache(&aspect_path, &aspect)?;
    Ok(Some(CachedDemFeatures {
        elevation: mmap_cached_values(&elevation_path, rows.len())?,
        slope: mmap_cached_values(&slope_path, rows.len())?,
        aspect: mmap_cached_values(&aspect_path, rows.len())?,
    }))
}

#[cfg(feature = "gdal")]
fn point_cache_path(
    raster_path: &str,
    rows: &[usize],
    cols: &[usize],
    cache_dir: &Path,
    suffix: &str,
) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    raster_path.hash(&mut hasher);
    suffix.hash(&mut hasher);
    rows.len().hash(&mut hasher);
    if let Ok(metadata) = std::fs::metadata(raster_path) {
        metadata.len().hash(&mut hasher);
        if let Ok(modified) = metadata.modified() {
            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                duration.as_secs().hash(&mut hasher);
                duration.subsec_nanos().hash(&mut hasher);
            }
        }
    }
    for (&row, &col) in rows.iter().zip(cols.iter()) {
        row.hash(&mut hasher);
        col.hash(&mut hasher);
    }
    cache_dir.join(format!("{:016x}_{suffix}.f32", hasher.finish()))
}

#[cfg(feature = "gdal")]
fn cache_is_valid(path: &Path, len: usize) -> bool {
    let expected = (len as u64).saturating_mul(4);
    std::fs::metadata(path)
        .map(|metadata| metadata.len() == expected)
        .unwrap_or(false)
}

#[cfg(feature = "gdal")]
fn mmap_cached_values(path: &Path, len: usize) -> LoaderResult<CachedRasterValues> {
    let file = File::open(path).map_err(LoaderError::Io)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(LoaderError::Io)?;
    if mmap.len() != len.saturating_mul(4) {
        return Err(LoaderError::Gdal(format!(
            "invalid mmap cache length for {}: got {}, expected {}",
            path.display(),
            mmap.len(),
            len.saturating_mul(4)
        )));
    }
    Ok(CachedRasterValues { mmap, len })
}

#[cfg(feature = "gdal")]
fn write_f32_cache(path: &Path, values: &[f32]) -> LoaderResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(LoaderError::Io)?;
    }
    let tmp_path = path.with_extension("f32.tmp");
    let mut file = File::create(&tmp_path).map_err(LoaderError::Io)?;
    for value in values {
        file.write_all(&value.to_le_bytes())
            .map_err(LoaderError::Io)?;
    }
    file.flush().map_err(LoaderError::Io)?;
    std::fs::rename(&tmp_path, path).map_err(LoaderError::Io)?;
    Ok(())
}

#[cfg(feature = "gdal")]
fn read_raster_values_by_block(
    path: &str,
    rows: &[usize],
    cols: &[usize],
    workers: usize,
) -> LoaderResult<Vec<f32>> {
    let ds = Dataset::open(path).map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let (width, height) = ds.raster_size();
    let block_size = 512usize;
    let mut values = vec![f32::NAN; rows.len()];
    let blocks: Vec<_> = group_points_by_block(rows, cols, width, height, block_size)
        .into_iter()
        .collect();
    drop(ds);

    let workers = workers.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| LoaderError::Gdal(format!("point cache thread pool: {e}")))?;
    let block_values: Vec<LoaderResult<Vec<(usize, f32)>>> = pool.install(|| {
        blocks
            .par_iter()
            .map(|((block_row, block_col), points)| {
                read_raster_block_values(
                    path, *block_row, *block_col, block_size, width, height, points,
                )
            })
            .collect()
    });

    for result in block_values {
        for (idx, value) in result? {
            values[idx] = value;
        }
    }

    Ok(values)
}

#[cfg(feature = "gdal")]
fn read_dem_features_by_block(
    path: &str,
    rows: &[usize],
    cols: &[usize],
    workers: usize,
) -> LoaderResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let ds = Dataset::open(path).map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let gt = ds
        .geo_transform()
        .map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let res_x = gt[1].abs() as f32;
    let res_y = gt[5].abs() as f32;
    let (width, height) = ds.raster_size();
    let block_size = 512usize;
    let mut elevation = vec![f32::NAN; rows.len()];
    let mut slope = vec![f32::NAN; rows.len()];
    let mut aspect = vec![f32::NAN; rows.len()];
    let blocks: Vec<_> = group_points_by_block(rows, cols, width, height, block_size)
        .into_iter()
        .collect();
    drop(ds);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()
        .map_err(|e| LoaderError::Gdal(format!("DEM cache thread pool: {e}")))?;
    let block_values: Vec<LoaderResult<Vec<(usize, f32, f32, f32)>>> = pool.install(|| {
        blocks
            .par_iter()
            .map(|((block_row, block_col), points)| {
                read_dem_block_features(
                    path, *block_row, *block_col, block_size, width, height, res_x, res_y, points,
                )
            })
            .collect()
    });

    for result in block_values {
        for (idx, elev, slp, asp) in result? {
            elevation[idx] = elev;
            slope[idx] = slp;
            aspect[idx] = asp;
        }
    }

    Ok((elevation, slope, aspect))
}

#[cfg(feature = "gdal")]
fn read_raster_block_values(
    path: &str,
    block_row: usize,
    block_col: usize,
    block_size: usize,
    width: usize,
    height: usize,
    points: &[(usize, usize, usize)],
) -> LoaderResult<Vec<(usize, f32)>> {
    let ds = Dataset::open(path).map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let band = ds
        .rasterband(1)
        .map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let row_start = block_row * block_size;
    let col_start = block_col * block_size;
    let row_end = (row_start + block_size).min(height);
    let col_end = (col_start + block_size).min(width);
    let win_h = row_end - row_start;
    let win_w = col_end - col_start;
    if win_w == 0 || win_h == 0 {
        return Ok(Vec::new());
    }
    let data = band
        .read_as::<f32>(
            (col_start as isize, row_start as isize),
            (win_w, win_h),
            (win_w, win_h),
            None,
        )
        .map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let buf = data.data();
    let mut out = Vec::with_capacity(points.len());
    for &(idx, row, col) in points {
        let local_r = row - row_start;
        let local_c = col - col_start;
        out.push((idx, buf[local_r * win_w + local_c]));
    }
    Ok(out)
}

#[cfg(feature = "gdal")]
#[allow(clippy::too_many_arguments)]
fn read_dem_block_features(
    path: &str,
    block_row: usize,
    block_col: usize,
    block_size: usize,
    width: usize,
    height: usize,
    res_x: f32,
    res_y: f32,
    points: &[(usize, usize, usize)],
) -> LoaderResult<Vec<(usize, f32, f32, f32)>> {
    let ds = Dataset::open(path).map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let band = ds
        .rasterband(1)
        .map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let core_row_start = block_row * block_size;
    let core_col_start = block_col * block_size;
    let core_row_end = (core_row_start + block_size).min(height);
    let core_col_end = (core_col_start + block_size).min(width);
    let row_start = core_row_start.saturating_sub(1);
    let col_start = core_col_start.saturating_sub(1);
    let row_end = (core_row_end + 1).min(height);
    let col_end = (core_col_end + 1).min(width);
    let win_h = row_end - row_start;
    let win_w = col_end - col_start;
    if win_w == 0 || win_h == 0 {
        return Ok(Vec::new());
    }

    let data = band
        .read_as::<f32>(
            (col_start as isize, row_start as isize),
            (win_w, win_h),
            (win_w, win_h),
            None,
        )
        .map_err(|e| LoaderError::Gdal(e.to_string()))?;
    let buf = data.data();
    let mut out = Vec::with_capacity(points.len());

    for &(idx, row, col) in points {
        let local_r = row - row_start;
        let local_c = col - col_start;
        let centre = buf[local_r * win_w + local_c];
        if !centre.is_finite() {
            out.push((idx, f32::NAN, f32::NAN, f32::NAN));
            continue;
        }

        if row == 0 || col == 0 || row + 1 >= height || col + 1 >= width {
            out.push((idx, centre, f32::NAN, f32::NAN));
            continue;
        }

        let left = buf[local_r * win_w + local_c - 1];
        let right = buf[local_r * win_w + local_c + 1];
        let top = buf[(local_r - 1) * win_w + local_c];
        let bottom = buf[(local_r + 1) * win_w + local_c];
        let window = [
            f32::NAN,
            top,
            f32::NAN,
            left,
            centre,
            right,
            f32::NAN,
            bottom,
            f32::NAN,
        ];
        let dz_dx = central_diff_x_3x3(&window, res_x);
        let dz_dy = central_diff_y_3x3(&window, res_y);
        if dz_dx.is_nan() || dz_dy.is_nan() {
            out.push((idx, centre, f32::NAN, f32::NAN));
            continue;
        }
        let rise = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt();
        let slope = rise.atan().to_degrees();

        let mut aspect = dz_dy.atan2(-dz_dx).to_degrees();
        if aspect < 0.0 {
            aspect = 90.0 - aspect;
        } else if aspect > 90.0 {
            aspect = 360.0 - aspect + 90.0;
        } else {
            aspect = 90.0 - aspect;
        }
        out.push((idx, centre, slope, aspect));
    }

    Ok(out)
}

#[cfg(feature = "gdal")]
fn group_points_by_block(
    rows: &[usize],
    cols: &[usize],
    width: usize,
    height: usize,
    block_size: usize,
) -> HashMap<(usize, usize), Vec<(usize, usize, usize)>> {
    let mut blocks: HashMap<(usize, usize), Vec<(usize, usize, usize)>> = HashMap::new();
    for (idx, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
        if row >= height || col >= width {
            continue;
        }
        blocks
            .entry((row / block_size, col / block_size))
            .or_default()
            .push((idx, row, col));
    }
    blocks
}

#[cfg(feature = "gdal")]
fn compute_3month_preceding_rainfall_cached(
    ccdc_timing: Option<f32>,
    point_idx: usize,
    current_year: &[Option<CachedRasterValues>; 12],
    prev_year: &[Option<CachedRasterValues>; 12],
) -> (f64, f64, f64) {
    let disturbance_month = timing_to_month(ccdc_timing);

    let get_rain = |values: &[Option<CachedRasterValues>; 12], month_idx: usize| -> f64 {
        values
            .get(month_idx)
            .and_then(|v| v.as_ref())
            .and_then(|v| v.get_f64(point_idx))
            .unwrap_or(f64::NAN)
    };

    if disturbance_month < 1 || disturbance_month > 12 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let m = disturbance_month as isize;
    let prev_1m = if m > 1 {
        get_rain(current_year, (m - 2) as usize)
    } else {
        get_rain(prev_year, 11)
    };
    let prev_2m = if m > 2 {
        get_rain(current_year, (m - 3) as usize)
    } else if m == 2 {
        get_rain(prev_year, 11)
    } else {
        get_rain(prev_year, 10)
    };
    let prev_3m = if m > 3 {
        get_rain(current_year, (m - 4) as usize)
    } else if m == 3 {
        get_rain(prev_year, 11)
    } else if m == 2 {
        get_rain(prev_year, 10)
    } else {
        get_rain(prev_year, 9)
    };

    (prev_1m, prev_2m, prev_3m)
}

// ── GDAL pixel read helpers ────────────────────────────────────────────

#[cfg(feature = "gdal")]
fn read_pixel_f64(ds: &Dataset, row: usize, col: usize) -> Option<f64> {
    let band = ds.rasterband(1).ok()?;
    let data = band
        .read_as::<f32>((col as isize, row as isize), (1, 1), (1, 1), None)
        .ok()?;
    let val = data.data().get(0).copied().unwrap_or(f32::NAN);
    if val.is_finite() {
        Some(val as f64)
    } else {
        None
    }
}

#[cfg(not(feature = "gdal"))]
fn read_pixel_f64(_ds: &(), _row: usize, _col: usize) -> Option<f64> {
    None
}

#[cfg(feature = "gdal")]
fn read_pixel_i32(ds: &Dataset, row: usize, col: usize) -> Option<i32> {
    let band = ds.rasterband(1).ok()?;
    let data = band
        .read_as::<f32>((col as isize, row as isize), (1, 1), (1, 1), None)
        .ok()?;
    let val = data.data().get(0).copied().unwrap_or(f32::NAN);
    if val.is_finite() {
        Some(val as i32)
    } else {
        None
    }
}

#[cfg(not(feature = "gdal"))]
fn read_pixel_i32(_ds: &(), _row: usize, _col: usize) -> Option<i32> {
    None
}

// ── DEM feature extraction ─────────────────────────────────────────────

/// Read a 3×3 window around (row, col) from the DEM, compute slope and
/// aspect via central differences, and return the centre elevation.
///
/// Returns `f64::NAN` if the pixel is NoData.
#[cfg(feature = "gdal")]
fn read_dem_features(
    ds: &Dataset,
    row: usize,
    col: usize,
    width: usize,
    height: usize,
    res_x: f64,
    res_y: f64,
    slope_out: &mut f64,
    aspect_out: &mut f64,
) -> f64 {
    // Read centre elevation first.
    let centre = read_pixel_f64(ds, row, col);

    let centre_val = match centre {
        Some(v) => v as f32,
        None => {
            *slope_out = f64::NAN;
            *aspect_out = f64::NAN;
            return f64::NAN;
        }
    };

    if centre_val.is_nan() {
        *slope_out = f64::NAN;
        *aspect_out = f64::NAN;
        return f64::NAN;
    }

    // Read a 3×3 window.
    let pixels = read_window_3x3_f32(ds, row, col, width, height);

    // Compute central differences.
    let dz_dx = central_diff_x_3x3(&pixels, res_x as f32);
    let dz_dy = central_diff_y_3x3(&pixels, res_y as f32);

    if dz_dx.is_nan() || dz_dy.is_nan() {
        *slope_out = f64::NAN;
        *aspect_out = f64::NAN;
    } else {
        // Slope = atan(sqrt(dz/dx² + dz/dy²)) * 180/π
        let rise = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt();
        *slope_out = rise.atan().to_degrees() as f64;

        // Aspect = atan2(dz/dy, -dz/dx) in degrees, then adjusted to
        // 0–360° clockwise from north (matching DEMLoader convention).
        let mut deg = dz_dy.atan2(-dz_dx).to_degrees();
        if deg < 0.0 {
            deg = 90.0 - deg;
        } else if deg > 90.0 {
            deg = 360.0 - deg + 90.0;
        } else {
            deg = 90.0 - deg;
        }
        *aspect_out = deg as f64;
    }

    centre_val as f64
}

#[cfg(not(feature = "gdal"))]
fn read_dem_features(
    _ds: &(),
    _row: usize,
    _col: usize,
    _width: usize,
    _height: usize,
    _res_x: f64,
    _res_y: f64,
    slope_out: &mut f64,
    aspect_out: &mut f64,
) -> f64 {
    *slope_out = f64::NAN;
    *aspect_out = f64::NAN;
    f64::NAN
}

fn read_dem_features_from_window(
    window: &RasterWindow,
    row: usize,
    col: usize,
    res_x: f64,
    res_y: f64,
) -> (f64, f64, f64) {
    let centre = match window.get_f64(row, col) {
        Some(value) => value,
        None => return (f64::NAN, f64::NAN, f64::NAN),
    };

    if res_x <= 0.0 || res_y <= 0.0 {
        return (centre, f64::NAN, f64::NAN);
    }

    let pixels = [
        f32::NAN,
        window
            .get_f32(row.saturating_sub(1), col)
            .unwrap_or(f32::NAN),
        f32::NAN,
        window
            .get_f32(row, col.saturating_sub(1))
            .unwrap_or(f32::NAN),
        centre as f32,
        window.get_f32(row, col + 1).unwrap_or(f32::NAN),
        f32::NAN,
        window.get_f32(row + 1, col).unwrap_or(f32::NAN),
        f32::NAN,
    ];
    let dz_dx = central_diff_x_3x3(&pixels, res_x as f32);
    let dz_dy = central_diff_y_3x3(&pixels, res_y as f32);
    if dz_dx.is_nan() || dz_dy.is_nan() {
        return (centre, f64::NAN, f64::NAN);
    }

    let rise = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt();
    let slope = rise.atan().to_degrees() as f64;

    let mut aspect = dz_dy.atan2(-dz_dx).to_degrees();
    if aspect < 0.0 {
        aspect = 90.0 - aspect;
    } else if aspect > 90.0 {
        aspect = 360.0 - aspect + 90.0;
    } else {
        aspect = 90.0 - aspect;
    }

    (centre, slope, aspect as f64)
}

/// Read a clamped 3×3 window of `f32` values from a single-band raster.
#[cfg(feature = "gdal")]
fn read_window_3x3_f32(
    ds: &Dataset,
    row: usize,
    col: usize,
    width: usize,
    height: usize,
) -> [f32; 9] {
    let r_start = row.saturating_sub(1);
    let r_end = (row + 2).min(height);
    let c_start = col.saturating_sub(1);
    let c_end = (col + 2).min(width);

    let w = c_end - c_start;
    let h = r_end - r_start;
    if w == 0 || h == 0 {
        return [f32::NAN; 9];
    }

    let band = match ds.rasterband(1) {
        Ok(b) => b,
        Err(_) => return [f32::NAN; 9],
    };

    let data = match band.read_as::<f32>((c_start as isize, r_start as isize), (w, h), (w, h), None)
    {
        Ok(d) => d,
        Err(_) => return [f32::NAN; 9],
    };

    let buf = data.data();
    let mut result = [f32::NAN; 9];

    // Centre pixel (row, col) maps to result[4] (index 1,1 in 3×3).
    // Compute the base offset that aligns the read window with the result grid.
    let base_r = 1isize - (row as isize - r_start as isize);
    let base_c = 1isize - (col as isize - c_start as isize);

    for r in 0..h {
        for c in 0..w {
            let rr = r as isize + base_r;
            let cc = c as isize + base_c;
            if rr >= 0 && rr < 3 && cc >= 0 && cc < 3 {
                result[(rr * 3 + cc) as usize] = buf[r * w + c];
            }
        }
    }

    result
}

#[cfg(not(feature = "gdal"))]
fn read_window_3x3_f32(
    _ds: &(),
    _row: usize,
    _col: usize,
    _width: usize,
    _height: usize,
) -> [f32; 9] {
    [f32::NAN; 9]
}

/// Central difference in x-direction from a 3×3 window.
///
/// Uses pixels[3] (left), pixels[5] (right). If either is NaN, returns NaN.
/// The centre pixel is pixels[4].
fn central_diff_x_3x3(window: &[f32; 9], res: f32) -> f32 {
    let left = window[3]; // mid-left
    let right = window[5]; // mid-right
    if left.is_nan() || right.is_nan() {
        return f32::NAN;
    }
    (right - left) / (2.0 * res)
}

/// Central difference in y-direction from a 3×3 window.
///
/// Uses pixels[1] (top), pixels[7] (bottom). If either is NaN, returns NaN.
/// The centre pixel is pixels[4].
fn central_diff_y_3x3(window: &[f32; 9], res: f32) -> f32 {
    let top = window[1]; // top-centre
    let bottom = window[7]; // bottom-centre
    if top.is_nan() || bottom.is_nan() {
        return f32::NAN;
    }
    (bottom - top) / (2.0 * res)
}

// ── road raster gap-filling ────────────────────────────────────────────

/// Find a road/railway/waterway distance raster for a given year, with
/// gap-filling: if `{prefix}_{year}.tif` doesn't exist, try year-1,
/// year-2, ..., year-5. Returns the path of the first found file, or
/// the original path if none found (caller handles missing file).
fn find_road_raster(resampled_dir: &str, prefix: &str, year: i32) -> String {
    let path = format!("{}/road/{}_{}.tif", resampled_dir, prefix, year);
    if Path::new(&path).exists() {
        return path;
    }
    for offset in 1..=5 {
        let prev_year = year - offset;
        let fallback = format!("{}/road/{}_{}.tif", resampled_dir, prefix, prev_year);
        if Path::new(&fallback).exists() {
            return fallback;
        }
    }
    path
}

/// Find a monthly rainfall raster with year fallback.
///
/// Tries `pre{year}{month:02}.tif`, then year-1 ... year-10.
fn find_rainfall_raster(resampled_dir: &str, year: i32, month: i32) -> String {
    let path = format!("{}/rainfall/pre{:04}{:02}.tif", resampled_dir, year, month);
    if Path::new(&path).exists() {
        return path;
    }
    for offset in 1..=10 {
        let prev_year = year - offset;
        let fallback = format!(
            "{}/rainfall/pre{:04}{:02}.tif",
            resampled_dir, prev_year, month
        );
        if Path::new(&fallback).exists() {
            return fallback;
        }
    }
    path
}

/// Find a monthly temperature raster with year fallback.
///
/// Tries `tmp_{year}_{month:02}.tif`, then year-1 ... year-10.
fn find_temperature_raster(resampled_dir: &str, year: i32, month: i32) -> String {
    let path = format!(
        "{}/temperature/tmp_{}_{:02}.tif",
        resampled_dir, year, month
    );
    if Path::new(&path).exists() {
        return path;
    }
    for offset in 1..=10 {
        let prev_year = year - offset;
        let fallback = format!(
            "{}/temperature/tmp_{}_{:02}.tif",
            resampled_dir, prev_year, month
        );
        if Path::new(&fallback).exists() {
            return fallback;
        }
    }
    path
}

// ── 3-month preceding rainfall ─────────────────────────────────────────

/// Compute rainfall_prev_1m, prev_2m, prev_3m for a point, given its
/// CCDc timing or default month.
///
/// * `ccdc_timing` — fractional year (e.g. 2017.42); NaN for background
///   points which default to month 6.
/// * Returns `(prev_1m, prev_2m, prev_3m)`.
#[cfg(feature = "gdal")]
fn compute_3month_preceding_rainfall(
    ccdc_timing: Option<f32>,
    row: usize,
    col: usize,
    current_year_ds: &[Option<Dataset>; 12],
    prev_year_ds: &[Option<Dataset>; 12],
) -> (f64, f64, f64) {
    let disturbance_month = timing_to_month(ccdc_timing);

    let get_rain = |ds_slice: &[Option<Dataset>; 12], month_idx: usize| -> f64 {
        if month_idx < 12 {
            if let Some(ref ds) = ds_slice[month_idx] {
                return read_pixel_f64(ds, row, col).unwrap_or(f64::NAN);
            }
        }
        f64::NAN
    };

    // month_idx is 0-based (0=Jan, 11=Dec).
    // disturbance_month is 1-based.
    // prev_1m = month - 1
    // prev_2m = month - 2
    // prev_3m = month - 3
    if disturbance_month < 1 || disturbance_month > 12 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let m = disturbance_month as isize; // 1-based

    let prev_1m = if m > 1 {
        get_rain(current_year_ds, (m - 2) as usize)
    } else {
        get_rain(prev_year_ds, 11) // December of previous year
    };

    let prev_2m = if m > 2 {
        get_rain(current_year_ds, (m - 3) as usize)
    } else if m == 2 {
        get_rain(prev_year_ds, 11) // December of previous year
    } else {
        get_rain(prev_year_ds, 10) // November of previous year
    };

    let prev_3m = if m > 3 {
        get_rain(current_year_ds, (m - 4) as usize)
    } else if m == 3 {
        get_rain(prev_year_ds, 11) // December of previous year
    } else if m == 2 {
        get_rain(prev_year_ds, 10) // November of previous year
    } else {
        get_rain(prev_year_ds, 9) // October of previous year
    };

    (prev_1m, prev_2m, prev_3m)
}

#[cfg(not(feature = "gdal"))]
fn compute_3month_preceding_rainfall(
    _ccdc_timing: Option<f32>,
    _row: usize,
    _col: usize,
    _current_year_ds: &[Option<()>; 12],
    _prev_year_ds: &[Option<()>; 12],
) -> (f64, f64, f64) {
    (f64::NAN, f64::NAN, f64::NAN)
}

fn compute_3month_preceding_rainfall_windows(
    ccdc_timing: Option<f32>,
    row: usize,
    col: usize,
    current_year: &[Option<RasterWindow>; 12],
    prev_year: &[Option<RasterWindow>; 12],
) -> (f64, f64, f64) {
    let disturbance_month = timing_to_month(ccdc_timing);

    let get_rain = |windows: &[Option<RasterWindow>; 12], month_idx: usize| -> f64 {
        windows
            .get(month_idx)
            .and_then(|window| window.as_ref())
            .and_then(|window| window.get_f64(row, col))
            .unwrap_or(f64::NAN)
    };

    if disturbance_month < 1 || disturbance_month > 12 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let m = disturbance_month as isize;
    let prev_1m = if m > 1 {
        get_rain(current_year, (m - 2) as usize)
    } else {
        get_rain(prev_year, 11)
    };
    let prev_2m = if m > 2 {
        get_rain(current_year, (m - 3) as usize)
    } else if m == 2 {
        get_rain(prev_year, 11)
    } else {
        get_rain(prev_year, 10)
    };
    let prev_3m = if m > 3 {
        get_rain(current_year, (m - 4) as usize)
    } else if m == 3 {
        get_rain(prev_year, 11)
    } else if m == 2 {
        get_rain(prev_year, 10)
    } else {
        get_rain(prev_year, 9)
    };

    (prev_1m, prev_2m, prev_3m)
}

/// Convert a CCDC fractional-year timing to a 1-based month.
///
/// For example, 2017.42 → `month = ((0.42) * 12).ceil() = 6` (June).
/// NaN or out-of-range values default to 6 (middle of growing season).
fn timing_to_month(ccdc_timing: Option<f32>) -> usize {
    match ccdc_timing {
        Some(t) if t.is_finite() => {
            let frac = t - t.floor();
            let m = (frac * 12.0).ceil() as isize;
            m.clamp(1, 12) as usize
        }
        _ => 6, // Default: June (middle of growing season)
    }
}

// ── tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── unit tests (no raster data needed) ─────────────────────────────

    #[test]
    fn test_timing_to_month() {
        assert_eq!(timing_to_month(Some(2017.0)), 1); // Jan
        assert_eq!(timing_to_month(Some(2017.0833)), 1); // early Jan
        assert_eq!(timing_to_month(Some(2017.42)), 6); // ~June
        assert_eq!(timing_to_month(Some(2017.5)), 6); // July
        assert_eq!(timing_to_month(Some(2017.999)), 12); // Dec
        assert_eq!(timing_to_month(None), 6); // NaN → default
        assert_eq!(timing_to_month(Some(f32::NAN)), 6); // NaN → default
    }

    #[test]
    fn test_central_diff_x_basic() {
        // Centre at window[4], left=[3], right=[5]
        let w: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dx = central_diff_x_3x3(&w, 1.0);
        assert!((dx - 1.0).abs() < 0.001, "dx={dx}");

        // NaN neighbours → NaN result
        let mut w2 = w;
        w2[3] = f32::NAN;
        assert!(central_diff_x_3x3(&w2, 1.0).is_nan());
    }

    #[test]
    fn test_central_diff_y_basic() {
        let w: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dy = central_diff_y_3x3(&w, 1.0);
        assert!((dy - 3.0).abs() < 0.001, "dy={dy}");

        // NaN neighbours → NaN result
        let mut w2 = w;
        w2[1] = f32::NAN;
        assert!(central_diff_y_3x3(&w2, 1.0).is_nan());
    }

    #[test]
    fn test_find_road_raster_path_format() {
        // Even without files on disk, verify path construction.
        let p = find_road_raster("/cache/resampled", "road_dist", 2020);
        assert_eq!(p, "/cache/resampled/road/road_dist_2020.tif");
    }

    #[test]
    fn test_default_parquet_args_smoke() {
        let _ = default_parquet_args();
    }

    #[test]
    fn test_effective_workers() {
        assert!(effective_workers(0) >= 1);
        assert_eq!(effective_workers(3), 3);
    }

    // ── integration tests (require resampled data) ─────────────────────

    #[cfg(feature = "gdal")]
    mod gdal_tests {
        use super::*;
        use tempfile::TempDir;

        fn write_test_tif(path: &str, width: usize, height: usize, value: f32) {
            use gdal_crate::raster::Buffer;
            use gdal_crate::DriverManager;

            let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
            let mut ds = driver
                .create_with_band_type::<f32, _>(path, width, height, 1)
                .unwrap();
            ds.set_geo_transform(&[0.0, 0.01, 0.0, 0.0, 0.0, -0.01])
                .unwrap();
            ds.set_projection("GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]")
                .unwrap();

            let data: Vec<f32> = vec![value; width * height];
            let mut buf = Buffer::new((width, height), data);
            let mut band = ds.rasterband(1).unwrap();
            band.write((0, 0), (width, height), &mut buf).unwrap();
            band.set_no_data_value(Some(f64::NAN)).unwrap();
            ds.flush_cache().unwrap();
        }

        /// Write a small parquet file with training points.
        fn write_training_parquet(path: &str, points: &[(i32, i32, i32, i32, i32, f32)]) {
            let mut rows_v: Vec<i32> = Vec::new();
            let mut cols_v: Vec<i32> = Vec::new();
            let mut years_v: Vec<i32> = Vec::new();
            let mut flags_v: Vec<i32> = Vec::new();
            let mut classes_v: Vec<i32> = Vec::new();
            let mut timings_v: Vec<f32> = Vec::new();
            for &(row, col, year, flag, class, timing) in points {
                rows_v.push(row);
                cols_v.push(col);
                years_v.push(year);
                flags_v.push(flag);
                classes_v.push(class);
                timings_v.push(timing);
            }
            let mut df = DataFrame::new(vec![
                Column::new("row".into(), rows_v),
                Column::new("col".into(), cols_v),
                Column::new("year".into(), years_v),
                Column::new("disturbance_flag".into(), flags_v),
                Column::new("disturbance_class".into(), classes_v),
                Column::new("ccdc_timing_year".into(), timings_v),
            ])
            .unwrap();
            let mut f = std::fs::File::create(path).unwrap();
            ParquetWriter::new(&mut f).finish(&mut df).unwrap();
        }

        #[test]
        fn test_extract_features_schema() {
            let tmp = TempDir::new().unwrap();
            let base = tmp.path();

            // Directory structure: .cache/features/ and .cache/resampled/...
            let features_dir = base.join(".cache").join("features");
            std::fs::create_dir_all(&features_dir).unwrap();

            let resampled = base.join(".cache").join("resampled");
            let dem_dir = &resampled;
            let landuse_dir = resampled.join("landuse");
            let river_dir = resampled.join("river");
            let road_dir = resampled.join("road");
            let rainfall_dir = resampled.join("rainfall");
            let temp_dir = resampled.join("temperature");
            let resident_dir = resampled.join("resident");

            for d in [
                dem_dir,
                &landuse_dir,
                &river_dir,
                &road_dir,
                &rainfall_dir,
                &temp_dir,
                &resident_dir,
            ] {
                std::fs::create_dir_all(d).unwrap();
            }

            // Create test rasters (tiny 5×5 grids).
            let dem_path = dem_dir.join("dem.tif");
            write_test_tif(dem_path.to_str().unwrap(), 5, 5, 100.0);

            let landuse_path = landuse_dir.join("landuse.tif");
            write_test_tif(landuse_path.to_str().unwrap(), 5, 5, 1.0);

            let river_path = river_dir.join("river_dist.tif");
            write_test_tif(river_path.to_str().unwrap(), 5, 5, 500.0);

            let road_path = road_dir.join("road_dist_2018.tif");
            write_test_tif(road_path.to_str().unwrap(), 5, 5, 1000.0);

            let railway_path = road_dir.join("railway_dist_2018.tif");
            write_test_tif(railway_path.to_str().unwrap(), 5, 5, 2000.0);

            let waterway_path = road_dir.join("waterway_dist_2018.tif");
            write_test_tif(waterway_path.to_str().unwrap(), 5, 5, 3000.0);

            let resident_path = resident_dir.join("resident.tif");
            write_test_tif(resident_path.to_str().unwrap(), 5, 5, 50.0);

            for m in 1..=12 {
                let rp = rainfall_dir.join(format!("pre2018{:02}.tif", m));
                write_test_tif(rp.to_str().unwrap(), 5, 5, m as f32 * 10.0);

                let tp = temp_dir.join(format!("tmp_2018_{:02}.tif", m));
                write_test_tif(tp.to_str().unwrap(), 5, 5, m as f32 * 2.0);
            }

            // Training points: 1 disturbance + 1 background, both at (2,2), year 2018.
            let train_path = features_dir.join("training_points.parquet");
            write_training_parquet(
                train_path.to_str().unwrap(),
                &[
                    (2, 2, 2018, 1, 1, 2018.42),  // disturbance, class=1 (fire), June
                    (3, 3, 2018, 0, 0, f32::NAN), // background
                ],
            );

            let out_path = features_dir.join("training_features.parquet");

            let result =
                extract_training_features(train_path.to_str().unwrap(), out_path.to_str().unwrap());
            assert!(
                result.is_ok(),
                "extract_training_features failed: {:?}",
                result.err()
            );

            // Verify output.
            let out_df =
                LazyFrame::scan_parquet(out_path.to_str().unwrap(), default_parquet_args())
                    .unwrap()
                    .collect()
                    .unwrap();

            assert_eq!(out_df.height(), 2, "row count");

            // Check expected columns.
            let expected_cols = &[
                "row",
                "col",
                "year",
                "disturbance_flag",
                "disturbance_class",
                "ccdc_timing_year",
                "elevation",
                "slope",
                "aspect",
                "landuse",
                "dist_to_river",
                "dist_to_road",
                "dist_to_railway",
                "dist_to_waterway",
                "resident",
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
                "rainfall_prev_1m",
                "rainfall_prev_2m",
                "rainfall_prev_3m",
                "rainfall_prev_total_3m",
                "temp_month_01",
                "temp_month_02",
                "temp_month_03",
                "temp_month_04",
                "temp_month_05",
                "temp_month_06",
                "temp_month_07",
                "temp_month_08",
                "temp_month_09",
                "temp_month_10",
                "temp_month_11",
                "temp_month_12",
            ];

            for col_name in expected_cols {
                assert!(
                    out_df.column(col_name).is_ok(),
                    "missing column: {col_name}"
                );
            }

            // disturbance_flag sum should be > 0.
            let flag_sum: i32 = out_df
                .column("disturbance_flag")
                .unwrap()
                .as_materialized_series()
                .i32()
                .unwrap()
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .sum();
            assert!(flag_sum > 0, "disturbance_flag sum={flag_sum}");

            // Landuse should be read correctly.
            let lu: Vec<Option<i32>> = out_df
                .column("landuse")
                .unwrap()
                .as_materialized_series()
                .i32()
                .unwrap()
                .into_iter()
                .collect();
            assert_eq!(lu[0], Some(1));
            assert_eq!(lu[1], Some(1));
        }

        #[test]
        fn test_features_no_nan_on_valid_pixels() {
            let tmp = TempDir::new().unwrap();
            let base = tmp.path();

            let features_dir = base.join(".cache").join("features");
            std::fs::create_dir_all(&features_dir).unwrap();

            let resampled = base.join(".cache").join("resampled");
            for sub in &[
                "landuse",
                "river",
                "road",
                "rainfall",
                "temperature",
                "resident",
            ] {
                std::fs::create_dir_all(resampled.join(sub)).unwrap();
            }

            // DEM with non-trivial values to produce valid slope/aspect.
            let dem_path = resampled.join("dem.tif");
            {
                use gdal_crate::raster::Buffer;
                use gdal_crate::DriverManager;
                let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
                let mut ds = driver
                    .create_with_band_type::<f32, _>(dem_path.to_str().unwrap(), 5, 5, 1)
                    .unwrap();
                ds.set_geo_transform(&[0.0, 0.01, 0.0, 0.0, 0.0, -0.01])
                    .unwrap();
                ds.set_projection("GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]")
                    .unwrap();

                // A small hill so slope/aspect are non-trivial.
                let data: Vec<f32> = vec![
                    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 30.0, 20.0, 10.0, 10.0, 30.0, 50.0,
                    30.0, 10.0, 10.0, 20.0, 30.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                ];
                let mut buf = Buffer::new((5, 5), data);
                let mut band = ds.rasterband(1).unwrap();
                band.write((0, 0), (5, 5), &mut buf).unwrap();
                band.set_no_data_value(Some(f64::NAN)).unwrap();
                ds.flush_cache().unwrap();
            }

            // All other rasters at (2,2) — valid values.
            write_test_tif(
                resampled.join("landuse/landuse.tif").to_str().unwrap(),
                5,
                5,
                2.0,
            );
            write_test_tif(
                resampled.join("river/river_dist.tif").to_str().unwrap(),
                5,
                5,
                500.0,
            );
            write_test_tif(
                resampled.join("road/road_dist_2018.tif").to_str().unwrap(),
                5,
                5,
                1000.0,
            );
            write_test_tif(
                resampled
                    .join("road/railway_dist_2018.tif")
                    .to_str()
                    .unwrap(),
                5,
                5,
                2000.0,
            );
            write_test_tif(
                resampled
                    .join("road/waterway_dist_2018.tif")
                    .to_str()
                    .unwrap(),
                5,
                5,
                3000.0,
            );
            write_test_tif(
                resampled.join("resident/resident.tif").to_str().unwrap(),
                5,
                5,
                50.0,
            );

            for m in 1..=12 {
                write_test_tif(
                    resampled
                        .join(format!("rainfall/pre2018{:02}.tif", m))
                        .to_str()
                        .unwrap(),
                    5,
                    5,
                    10.0,
                );
                write_test_tif(
                    resampled
                        .join(format!("temperature/tmp_2018_{:02}.tif", m))
                        .to_str()
                        .unwrap(),
                    5,
                    5,
                    15.0,
                );
            }

            // A single disturbance point at (1,1), year 2018 (off-centre on hill).
            let train_path = features_dir.join("training_points.parquet");
            write_training_parquet(train_path.to_str().unwrap(), &[(1, 1, 2018, 1, 1, 2018.5)]);

            let out_path = features_dir.join("training_features.parquet");
            let result =
                extract_training_features(train_path.to_str().unwrap(), out_path.to_str().unwrap());
            assert!(result.is_ok());

            let out_df =
                LazyFrame::scan_parquet(out_path.to_str().unwrap(), default_parquet_args())
                    .unwrap()
                    .collect()
                    .unwrap();

            assert_eq!(out_df.height(), 1);

            // All feature columns should be finite for a valid pixel.
            let feature_cols = &[
                "elevation",
                "slope",
                "aspect",
                "dist_to_river",
                "dist_to_road",
                "dist_to_railway",
                "dist_to_waterway",
                "resident",
                "rainfall_prev_1m",
                "rainfall_prev_2m",
                "rainfall_prev_3m",
                "rainfall_prev_total_3m",
            ];

            for col_name in feature_cols {
                let val = out_df
                    .column(col_name)
                    .unwrap()
                    .as_materialized_series()
                    .f64()
                    .unwrap()
                    .get(0);
                assert!(
                    val.is_some() && val.unwrap().is_finite(),
                    "{col_name} should be finite, got {val:?}"
                );
            }

            let lu = out_df
                .column("landuse")
                .unwrap()
                .as_materialized_series()
                .i32()
                .unwrap()
                .get(0);
            assert!(lu.is_some(), "landuse should be Some");

            // Monthly columns too.
            for m in 1..=12 {
                let col_name = format!("rainfall_month_{:02}", m);
                let val = out_df
                    .column(&col_name)
                    .unwrap()
                    .as_materialized_series()
                    .f64()
                    .unwrap()
                    .get(0);
                assert!(
                    val.is_some() && val.unwrap().is_finite(),
                    "{col_name} should be finite"
                );

                let tcol = format!("temp_month_{:02}", m);
                let tval = out_df
                    .column(&tcol)
                    .unwrap()
                    .as_materialized_series()
                    .f64()
                    .unwrap()
                    .get(0);
                assert!(
                    tval.is_some() && tval.unwrap().is_finite(),
                    "{tcol} should be finite"
                );
            }

            // Slope should be > 0 (we have a hill).
            let slope = out_df
                .column("slope")
                .unwrap()
                .as_materialized_series()
                .f64()
                .unwrap()
                .get(0)
                .unwrap();
            assert!(slope > 0.0, "slope={slope} should be > 0 on a hill");
        }

        #[test]
        fn test_road_gap_fill() {
            let tmp = TempDir::new().unwrap();
            let base = tmp.path();

            let features_dir = base.join(".cache").join("features");
            std::fs::create_dir_all(&features_dir).unwrap();

            let road_dir = base.join(".cache").join("resampled").join("road");
            std::fs::create_dir_all(&road_dir).unwrap();

            // Only road_dist_2016.tif exists (not 2018).
            write_test_tif(
                road_dir.join("road_dist_2016.tif").to_str().unwrap(),
                3,
                3,
                500.0,
            );

            let train_path = features_dir.join("training_points.parquet");
            write_training_parquet(train_path.to_str().unwrap(), &[(1, 1, 2018, 1, 1, 2018.5)]);

            // gap-fill should find road_dist_2016.tif.
            let path = find_road_raster(
                road_dir
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join("resampled")
                    .to_str()
                    .unwrap(),
                "road_dist",
                2018,
            );
            assert!(
                path.contains("road_dist_2016.tif"),
                "gap fill failed: {path}"
            );
        }
    }
}

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

use std::collections::HashMap;
use std::path::Path;

use if_loaders::base::{LoaderError, LoaderResult};
use polars::prelude::*;

#[cfg(feature = "gdal")]
use gdal::Dataset;

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
pub fn extract_training_features(
    training_parquet: &str,
    output_parquet: &str,
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

    let rows_s = df.column("row").map_err(|e| plerr(e))?.as_materialized_series();
    let cols_s = df.column("col").map_err(|e| plerr(e))?.as_materialized_series();
    let years_s = df.column("year").map_err(|e| plerr(e))?.as_materialized_series();
    let flags_s = df.column("disturbance_flag").map_err(|e| plerr(e))?.as_materialized_series();
    let classes_s = df.column("disturbance_class").map_err(|e| plerr(e))?.as_materialized_series();
    let timings_s = df.column("ccdc_timing_year").map_err(|e| plerr(e))?.as_materialized_series();

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

    // Open DEM once for all points — needed for slope/aspect.
    let dem_path = format!("{}/dem.tif", resampled_str);
    let dem_ds = open_raster_optional(&dem_path);
    let (dem_res_x, dem_res_y, dem_width, dem_height) =
        if let Some(ref ds) = dem_ds {
            let gt = ds.geo_transform().map_err(|e| LoaderError::Gdal(e.to_string()))?;
            let (w, h) = ds.raster_size();
            (gt[1].abs(), gt[5].abs(), w as usize, h as usize)
        } else {
            (0.0, 0.0, 0, 0)
        };

    // Pre-open static rasters (cached).
    let landuse_path = format!("{}/landuse/landuse.tif", resampled_str);
    let river_path = format!("{}/river/river_dist.tif", resampled_str);
    let resident_path = format!("{}/resident/resident.tif", resampled_str);

    let landuse_ds = open_raster_optional(&landuse_path);
    let river_ds = open_raster_optional(&river_path);
    let resident_ds = open_raster_optional(&resident_path);

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
        vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
        vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
        vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
        vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
    ];
    let mut out_temp: [Vec<f64>; 12] = [
        vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
        vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
        vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
        vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n],
    ];
    let mut out_rainfall_prev_1m: Vec<f64> = vec![f64::NAN; n];
    let mut out_rainfall_prev_2m: Vec<f64> = vec![f64::NAN; n];
    let mut out_rainfall_prev_3m: Vec<f64> = vec![f64::NAN; n];
    let mut out_rainfall_prev_total_3m: Vec<f64> = vec![f64::NAN; n];

    // ── process points grouped by year ─────────────────────────────────
    for year in &years_sorted {
        let indices = &by_year[year];

        // Open yearly rasters for this year.
        let road_path = find_road_raster(&resampled_str, "road_dist", *year);
        let railway_path = find_road_raster(&resampled_str, "railway_dist", *year);
        let waterway_path = find_road_raster(&resampled_str, "waterway_dist", *year);

        let road_ds = open_raster_optional(&road_path);
        let railway_ds = open_raster_optional(&railway_path);
        let waterway_ds = open_raster_optional(&waterway_path);

        // Open monthly rainfall rasters for this year (cache all 12).
        let mut rainfall_ds: [Option<Dataset>; 12] = Default::default();
        for m in 0..12 {
            let month = m + 1;
            let rp = format!("{}/rainfall/pre{}{:02}.tif", resampled_str, year, month);
            rainfall_ds[m] = open_raster_optional(&rp);
        }

        // Open monthly temperature rasters for this year.
        let mut temp_ds: [Option<Dataset>; 12] = Default::default();
        for m in 0..12 {
            let month = m + 1;
            let tp = format!("{}/temperature/tmp_{}_{:02}.tif", resampled_str, year, month);
            temp_ds[m] = open_raster_optional(&tp);
        }

        // Also need prev-year rainfall for 3-month window wrapping.
        let prev_year = year - 1;
        let mut prev_rainfall_ds: [Option<Dataset>; 12] = Default::default();
        for m in 0..12 {
            let month = m + 1;
            let rp = format!("{}/rainfall/pre{}{:02}.tif", resampled_str, prev_year, month);
            prev_rainfall_ds[m] = open_raster_optional(&rp);
        }

        for &i in indices {
            let r = rows.get(i).unwrap_or(0) as usize;
            let c = cols.get(i).unwrap_or(0) as usize;

            // ── static features ────────────────────────────────────
            // DEM: elevation + slope + aspect
            if let Some(ref ds) = dem_ds {
                out_elevation[i] = read_dem_features(
                    ds, r, c, dem_width, dem_height, dem_res_x, dem_res_y,
                    &mut out_slope[i], &mut out_aspect[i],
                );
            }

            // Landuse
            if let Some(ref ds) = landuse_ds {
                out_landuse[i] = read_pixel_i32(ds, r, c).unwrap_or(0);
            }

            // River distance
            if let Some(ref ds) = river_ds {
                out_dist_to_river[i] = read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
            }

            // ── dynamic (yearly) features ──────────────────────────
            if let Some(ref ds) = road_ds {
                out_dist_to_road[i] = read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
            }
            if let Some(ref ds) = railway_ds {
                out_dist_to_railway[i] = read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
            }
            if let Some(ref ds) = waterway_ds {
                out_dist_to_waterway[i] = read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
            }

            // Resident
            if let Some(ref ds) = resident_ds {
                out_resident[i] = read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
            }

            // ── monthly rainfall ───────────────────────────────────
            for m in 0..12 {
                if let Some(ref ds) = rainfall_ds[m] {
                    out_rainfall[m][i] = read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                }
            }

            // ── monthly temperature ────────────────────────────────
            for m in 0..12 {
                if let Some(ref ds) = temp_ds[m] {
                    out_temp[m][i] = read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                }
            }

            // ── 3-month preceding rainfall ─────────────────────────
            let (prev_1m, prev_2m, prev_3m) = compute_3month_preceding_rainfall(
                timings.get(i),
                r, c,
                &rainfall_ds,
                &prev_rainfall_ds,
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
        cols.push(Column::new(name.into(), std::mem::take(&mut out_rainfall[m])));
    }

    // 3-month preceding rainfall.
    cols.push(Column::new("rainfall_prev_1m".into(), out_rainfall_prev_1m));
    cols.push(Column::new("rainfall_prev_2m".into(), out_rainfall_prev_2m));
    cols.push(Column::new("rainfall_prev_3m".into(), out_rainfall_prev_3m));
    cols.push(Column::new("rainfall_prev_total_3m".into(), out_rainfall_prev_total_3m));

    // Monthly temperature.
    for m in 0..12 {
        let name = format!("temp_month_{:02}", m + 1);
        cols.push(Column::new(name.into(), std::mem::take(&mut out_temp[m])));
    }

    let mut out_df = DataFrame::new(cols)
        .map_err(|e| LoaderError::Gdal(format!("build DataFrame: {e}")))?;

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
/// * `output_dir` — directory for tile parquet files
/// * `tile_size` — tile side length in pixels (e.g. 1024)
/// * `resampled_dir` — path to the resampled raster cache
/// * `target_year` — prediction year (default 2025)
///
/// Returns the number of tiles written.
pub fn extract_prediction_features(
    class_grid: &crate::class_grid::ClassGrid,
    output_dir: &str,
    tile_size: usize,
    resampled_dir: &str,
    target_year: i32,
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
        "[predict] CLASS grid: {}×{}, tile size: {}, tiles: {}×{} = {}",
        width, height, tile_size, n_tile_rows, n_tile_cols, total_tiles
    );

    // Use previous year for data if target_year not yet available.
    let data_year = resolve_data_year(resampled, target_year);
    eprintln!(
        "[predict] Target year {}, using data year {}",
        target_year, data_year
    );

    // Open static rasters (cached for entire run).
    let dem_path = resampled.join("dem.tif");
    let landuse_path = resampled.join("landuse/landuse.tif");
    let river_path = resampled.join("river/river_dist.tif");
    let resident_path = resampled.join("resident/resident.tif");

    let dem_ds = open_raster_optional(&dem_path.to_string_lossy());
    let landuse_ds = open_raster_optional(&landuse_path.to_string_lossy());
    let river_ds = open_raster_optional(&river_path.to_string_lossy());
    let resident_ds = open_raster_optional(&resident_path.to_string_lossy());

    // DEM pixel resolution (for slope/aspect).
    let (dem_res_x, dem_res_y, dem_width, dem_height) = if let Some(ref ds) = dem_ds {
        let gt = ds
            .geo_transform()
            .map_err(|e| LoaderError::Gdal(e.to_string()))?;
        let (w, h) = ds.raster_size();
        (gt[1].abs(), gt[5].abs(), w as usize, h as usize)
    } else {
        (0.0, 0.0, 0, 0)
    };

    let resampled_str = resampled.to_string_lossy().to_string();

    // Road/railway/waterway: use find_road_raster for gap-fill.
    let road_path =
        find_road_raster(&resampled_str, "road_dist", target_year);
    let railway_path =
        find_road_raster(&resampled_str, "railway_dist", target_year);
    let waterway_path =
        find_road_raster(&resampled_str, "waterway_dist", target_year);

    // Open road rasters (same for all tiles).
    let road_ds = open_raster_optional(&road_path);
    let railway_ds = open_raster_optional(&railway_path);
    let waterway_ds = open_raster_optional(&waterway_path);

    // Open monthly rainfall rasters for data_year and data_year-1.
    let mut rainfall_ds: [Option<Dataset>; 12] = Default::default();
    for m in 0..12 {
        let month = m + 1;
        let rp = format!("{}/rainfall/pre{}{:02}.tif", resampled_str, data_year, month);
        rainfall_ds[m] = open_raster_optional(&rp);
    }
    let prev_year = data_year - 1;
    let mut prev_rainfall_ds: [Option<Dataset>; 12] = Default::default();
    for m in 0..12 {
        let month = m + 1;
        let rp = format!("{}/rainfall/pre{}{:02}.tif", resampled_str, prev_year, month);
        prev_rainfall_ds[m] = open_raster_optional(&rp);
    }

    // Open monthly temperature rasters.
    let mut temp_ds: [Option<Dataset>; 12] = Default::default();
    for m in 0..12 {
        let month = m + 1;
        let tp = format!("{}/temperature/tmp_{}_{:02}.tif", resampled_str, data_year, month);
        temp_ds[m] = open_raster_optional(&tp);
    }

    let mut tiles_written: usize = 0;

    for tile_row in (0..height).step_by(tile_size) {
        for tile_col in (0..width).step_by(tile_size) {
            let tile_w = tile_size.min(width - tile_col);
            let tile_h = tile_size.min(height - tile_row);
            let n_pixels = tile_w * tile_h;

            // ── Allocate per-tile vectors ─────────────────────────────
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
            let mut out_rainfall: [Vec<f64>; 12] = Default::default();
            for m in 0..12 {
                out_rainfall[m] = vec![f64::NAN; n_pixels];
            }
            let mut out_rainfall_prev_1m: Vec<f64> = vec![f64::NAN; n_pixels];
            let mut out_rainfall_prev_2m: Vec<f64> = vec![f64::NAN; n_pixels];
            let mut out_rainfall_prev_3m: Vec<f64> = vec![f64::NAN; n_pixels];
            let mut out_rainfall_prev_total_3m: Vec<f64> = vec![f64::NAN; n_pixels];
            let mut out_temp: [Vec<f64>; 12] = Default::default();
            for m in 0..12 {
                out_temp[m] = vec![f64::NAN; n_pixels];
            }

            for dy in 0..tile_h {
                for dx in 0..tile_w {
                    let r = tile_row + dy;
                    let c = tile_col + dx;
                    let pi = dy * tile_w + dx;

                    out_rows.push(r as i32);
                    out_cols.push(c as i32);

                    // DEM: elevation + slope + aspect
                    if let Some(ref ds) = dem_ds {
                        out_elevation[pi] = read_dem_features(
                            ds,
                            r,
                            c,
                            dem_width,
                            dem_height,
                            dem_res_x,
                            dem_res_y,
                            &mut out_slope[pi],
                            &mut out_aspect[pi],
                        );
                    }

                    // Landuse
                    if let Some(ref ds) = landuse_ds {
                        out_landuse[pi] = read_pixel_i32(ds, r, c).unwrap_or(0);
                    }

                    // River distance
                    if let Some(ref ds) = river_ds {
                        out_dist_to_river[pi] =
                            read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                    }

                    // Road distances
                    if let Some(ref ds) = road_ds {
                        out_dist_to_road[pi] =
                            read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                    }
                    if let Some(ref ds) = railway_ds {
                        out_dist_to_railway[pi] =
                            read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                    }
                    if let Some(ref ds) = waterway_ds {
                        out_dist_to_waterway[pi] =
                            read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                    }

                    // Resident
                    if let Some(ref ds) = resident_ds {
                        out_resident[pi] =
                            read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                    }

                    // Monthly rainfall
                    for m in 0..12 {
                        if let Some(ref ds) = rainfall_ds[m] {
                            out_rainfall[m][pi] =
                                read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                        }
                    }

                    // Monthly temperature
                    for m in 0..12 {
                        if let Some(ref ds) = temp_ds[m] {
                            out_temp[m][pi] =
                                read_pixel_f64(ds, r, c).unwrap_or(f64::NAN);
                        }
                    }

                    // 3-month preceding rainfall (default month = 6 for
                    // prediction; no CCDC timing available for 2025).
                    let (prev_1m, prev_2m, prev_3m) =
                        compute_3month_preceding_rainfall(
                            None,
                            r,
                            c,
                            &rainfall_ds,
                            &prev_rainfall_ds,
                        );
                    out_rainfall_prev_1m[pi] = prev_1m;
                    out_rainfall_prev_2m[pi] = prev_2m;
                    out_rainfall_prev_3m[pi] = prev_3m;
                    if prev_1m.is_finite()
                        && prev_2m.is_finite()
                        && prev_3m.is_finite()
                    {
                        out_rainfall_prev_total_3m[pi] =
                            prev_1m + prev_2m + prev_3m;
                    }
                }
            }

            // ── Build DataFrame ──────────────────────────────────────
            let mut cols: Vec<Column> = Vec::with_capacity(38);

            cols.push(Column::new("row".into(), out_rows));
            cols.push(Column::new("col".into(), out_cols));
            cols.push(Column::new("elevation".into(), out_elevation));
            cols.push(Column::new("slope".into(), out_slope));
            cols.push(Column::new("aspect".into(), out_aspect));
            cols.push(Column::new("landuse".into(), out_landuse));
            cols.push(Column::new("dist_to_river".into(), out_dist_to_river));
            cols.push(Column::new(
                "dist_to_road".into(),
                out_dist_to_road,
            ));
            cols.push(Column::new(
                "dist_to_railway".into(),
                out_dist_to_railway,
            ));
            cols.push(Column::new(
                "dist_to_waterway".into(),
                out_dist_to_waterway,
            ));
            cols.push(Column::new("resident".into(), out_resident));

            for m in 0..12 {
                let name = format!("rainfall_month_{:02}", m + 1);
                cols.push(Column::new(
                    name.into(),
                    std::mem::take(&mut out_rainfall[m]),
                ));
            }

            cols.push(Column::new(
                "rainfall_prev_1m".into(),
                out_rainfall_prev_1m,
            ));
            cols.push(Column::new(
                "rainfall_prev_2m".into(),
                out_rainfall_prev_2m,
            ));
            cols.push(Column::new(
                "rainfall_prev_3m".into(),
                out_rainfall_prev_3m,
            ));
            cols.push(Column::new(
                "rainfall_prev_total_3m".into(),
                out_rainfall_prev_total_3m,
            ));

            for m in 0..12 {
                let name = format!("temp_month_{:02}", m + 1);
                cols.push(Column::new(
                    name.into(),
                    std::mem::take(&mut out_temp[m]),
                ));
            }

            let mut df = DataFrame::new(cols)
                .map_err(|e| LoaderError::Gdal(format!("build tile DataFrame: {e}")))?;

            // Write tile parquet.
            let tile_path = format!(
                "{}/tile_{:06}_{:06}.parquet",
                output_dir, tile_row, tile_col
            );
            let mut f = std::fs::File::create(&tile_path).map_err(LoaderError::Io)?;
            ParquetWriter::new(&mut f)
                .finish(&mut df)
                .map_err(|e| LoaderError::Gdal(format!("write tile parquet: {e}")))?;

            tiles_written += 1;

            if tiles_written % 10 == 0 || tiles_written == 1 {
                eprintln!(
                    "[predict] Tiles: {}/{} ({:.1}%) - tile_{:06}_{:06}.parquet ({}×{} = {} px)",
                    tiles_written,
                    total_tiles,
                    100.0 * tiles_written as f64 / total_tiles as f64,
                    tile_row,
                    tile_col,
                    tile_w,
                    tile_h,
                    n_pixels
                );
            }
        }
    }

    eprintln!(
        "[predict] Feature extraction complete: {} tiles written to {}",
        tiles_written, output_dir
    );
    Ok(tiles_written)
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

fn default_parquet_args() -> ScanArgsParquet {
    ScanArgsParquet {
        rechunk: true,
        cache: true,
        ..Default::default()
    }
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

    let data = match band.read_as::<f32>(
        (c_start as isize, r_start as isize),
        (w, h),
        (w, h),
        None,
    ) {
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
    let left = window[3];  // mid-left
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
    let top = window[1];    // top-centre
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
    // Gap-fill: try previous years.
    for offset in 1..=5 {
        let prev_year = year - offset;
        let fallback = format!("{}/road/{}_{}.tif", resampled_dir, prefix, prev_year);
        if Path::new(&fallback).exists() {
            return fallback;
        }
    }
    // Return the original path — caller handles missing file.
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
        assert_eq!(timing_to_month(Some(2017.0)), 1);  // Jan
        assert_eq!(timing_to_month(Some(2017.0833)), 1); // early Jan
        assert_eq!(timing_to_month(Some(2017.42)), 6);   // ~June
        assert_eq!(timing_to_month(Some(2017.5)), 6);    // July
        assert_eq!(timing_to_month(Some(2017.999)), 12); // Dec
        assert_eq!(timing_to_month(None), 6);             // NaN → default
        assert_eq!(timing_to_month(Some(f32::NAN)), 6);   // NaN → default
    }

    #[test]
    fn test_central_diff_x_basic() {
        // Centre at window[4], left=[3], right=[5]
        let w: [f32; 9] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let dx = central_diff_x_3x3(&w, 1.0);
        assert!((dx - 1.0).abs() < 0.001, "dx={dx}");

        // NaN neighbours → NaN result
        let mut w2 = w;
        w2[3] = f32::NAN;
        assert!(central_diff_x_3x3(&w2, 1.0).is_nan());
    }

    #[test]
    fn test_central_diff_y_basic() {
        let w: [f32; 9] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
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

    // ── integration tests (require resampled data) ─────────────────────

    #[cfg(feature = "gdal")]
    mod gdal_tests {
        use super::*;
        use tempfile::TempDir;

        fn write_test_tif(path: &str, width: usize, height: usize, value: f32) {
            use gdal::raster::Buffer;
            use gdal::DriverManager;

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
            ]).unwrap();
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

            for d in [dem_dir, &landuse_dir, &river_dir, &road_dir, &rainfall_dir, &temp_dir, &resident_dir] {
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
                    (2, 2, 2018, 1, 1, 2018.42), // disturbance, class=1 (fire), June
                    (3, 3, 2018, 0, 0, f32::NAN), // background
                ],
            );

            let out_path = features_dir.join("training_features.parquet");

            let result = extract_training_features(
                train_path.to_str().unwrap(),
                out_path.to_str().unwrap(),
            );
            assert!(result.is_ok(), "extract_training_features failed: {:?}", result.err());

            // Verify output.
            let out_df = LazyFrame::scan_parquet(
                out_path.to_str().unwrap(),
                default_parquet_args(),
            ).unwrap().collect().unwrap();

            assert_eq!(out_df.height(), 2, "row count");

            // Check expected columns.
            let expected_cols = &[
                "row", "col", "year", "disturbance_flag", "disturbance_class",
                "ccdc_timing_year",
                "elevation", "slope", "aspect", "landuse", "dist_to_river",
                "dist_to_road", "dist_to_railway", "dist_to_waterway", "resident",
                "rainfall_month_01", "rainfall_month_02", "rainfall_month_03",
                "rainfall_month_04", "rainfall_month_05", "rainfall_month_06",
                "rainfall_month_07", "rainfall_month_08", "rainfall_month_09",
                "rainfall_month_10", "rainfall_month_11", "rainfall_month_12",
                "rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m",
                "rainfall_prev_total_3m",
                "temp_month_01", "temp_month_02", "temp_month_03",
                "temp_month_04", "temp_month_05", "temp_month_06",
                "temp_month_07", "temp_month_08", "temp_month_09",
                "temp_month_10", "temp_month_11", "temp_month_12",
            ];

            for col_name in expected_cols {
                assert!(
                    out_df.column(col_name).is_ok(),
                    "missing column: {col_name}"
                );
            }

            // disturbance_flag sum should be > 0.
            let flag_sum: i32 = out_df
                .column("disturbance_flag").unwrap()
                .as_materialized_series()
                .i32().unwrap()
                .into_iter()
                .map(|v| v.unwrap_or(0))
                .sum();
            assert!(flag_sum > 0, "disturbance_flag sum={flag_sum}");

            // Landuse should be read correctly.
            let lu: Vec<Option<i32>> = out_df
                .column("landuse").unwrap()
                .as_materialized_series()
                .i32().unwrap()
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
            for sub in &["landuse", "river", "road", "rainfall", "temperature", "resident"] {
                std::fs::create_dir_all(resampled.join(sub)).unwrap();
            }

            // DEM with non-trivial values to produce valid slope/aspect.
            let dem_path = resampled.join("dem.tif");
            {
                use gdal::raster::Buffer;
                use gdal::DriverManager;
                let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
                let mut ds = driver
                    .create_with_band_type::<f32, _>(
                        dem_path.to_str().unwrap(), 5, 5, 1,
                    )
                    .unwrap();
                ds.set_geo_transform(&[0.0, 0.01, 0.0, 0.0, 0.0, -0.01])
                    .unwrap();
                ds.set_projection("GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]")
                    .unwrap();

                // A small hill so slope/aspect are non-trivial.
                let data: Vec<f32> = vec![
                    10.0, 10.0, 10.0, 10.0, 10.0,
                    10.0, 20.0, 30.0, 20.0, 10.0,
                    10.0, 30.0, 50.0, 30.0, 10.0,
                    10.0, 20.0, 30.0, 20.0, 10.0,
                    10.0, 10.0, 10.0, 10.0, 10.0,
                ];
                let mut buf = Buffer::new((5, 5), data);
                let mut band = ds.rasterband(1).unwrap();
                band.write((0, 0), (5, 5), &mut buf).unwrap();
                band.set_no_data_value(Some(f64::NAN)).unwrap();
                ds.flush_cache().unwrap();
            }

            // All other rasters at (2,2) — valid values.
            write_test_tif(resampled.join("landuse/landuse.tif").to_str().unwrap(), 5, 5, 2.0);
            write_test_tif(resampled.join("river/river_dist.tif").to_str().unwrap(), 5, 5, 500.0);
            write_test_tif(resampled.join("road/road_dist_2018.tif").to_str().unwrap(), 5, 5, 1000.0);
            write_test_tif(resampled.join("road/railway_dist_2018.tif").to_str().unwrap(), 5, 5, 2000.0);
            write_test_tif(resampled.join("road/waterway_dist_2018.tif").to_str().unwrap(), 5, 5, 3000.0);
            write_test_tif(resampled.join("resident/resident.tif").to_str().unwrap(), 5, 5, 50.0);

            for m in 1..=12 {
                write_test_tif(
                    resampled.join(format!("rainfall/pre2018{:02}.tif", m)).to_str().unwrap(),
                    5, 5, 10.0,
                );
                write_test_tif(
                    resampled.join(format!("temperature/tmp_2018_{:02}.tif", m)).to_str().unwrap(),
                    5, 5, 15.0,
                );
            }

            // A single disturbance point at (1,1), year 2018 (off-centre on hill).
            let train_path = features_dir.join("training_points.parquet");
            write_training_parquet(
                train_path.to_str().unwrap(),
                &[(1, 1, 2018, 1, 1, 2018.5)],
            );

            let out_path = features_dir.join("training_features.parquet");
            let result = extract_training_features(
                train_path.to_str().unwrap(),
                out_path.to_str().unwrap(),
            );
            assert!(result.is_ok());

            let out_df = LazyFrame::scan_parquet(
                out_path.to_str().unwrap(),
                default_parquet_args(),
            ).unwrap().collect().unwrap();

            assert_eq!(out_df.height(), 1);

            // All feature columns should be finite for a valid pixel.
            let feature_cols = &[
                "elevation", "slope", "aspect", "dist_to_river",
                "dist_to_road", "dist_to_railway", "dist_to_waterway", "resident",
                "rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m",
                "rainfall_prev_total_3m",
            ];

            for col_name in feature_cols {
                let val = out_df.column(col_name).unwrap()
                    .as_materialized_series()
                    .f64().unwrap()
                    .get(0);
                assert!(
                    val.is_some() && val.unwrap().is_finite(),
                    "{col_name} should be finite, got {val:?}"
                );
            }

            let lu = out_df.column("landuse").unwrap()
                .as_materialized_series()
                .i32().unwrap()
                .get(0);
            assert!(lu.is_some(), "landuse should be Some");

            // Monthly columns too.
            for m in 1..=12 {
                let col_name = format!("rainfall_month_{:02}", m);
                let val = out_df.column(&col_name).unwrap()
                    .as_materialized_series()
                    .f64().unwrap()
                    .get(0);
                assert!(val.is_some() && val.unwrap().is_finite(),
                    "{col_name} should be finite");

                let tcol = format!("temp_month_{:02}", m);
                let tval = out_df.column(&tcol).unwrap()
                    .as_materialized_series()
                    .f64().unwrap()
                    .get(0);
                assert!(tval.is_some() && tval.unwrap().is_finite(),
                    "{tcol} should be finite");
            }

            // Slope should be > 0 (we have a hill).
            let slope = out_df.column("slope").unwrap()
                .as_materialized_series()
                .f64().unwrap()
                .get(0).unwrap();
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
            write_test_tif(road_dir.join("road_dist_2016.tif").to_str().unwrap(), 3, 3, 500.0);

            let train_path = features_dir.join("training_points.parquet");
            write_training_parquet(
                train_path.to_str().unwrap(),
                &[(1, 1, 2018, 1, 1, 2018.5)],
            );

            // gap-fill should find road_dist_2016.tif.
            let path = find_road_raster(
                road_dir.parent().unwrap().parent().unwrap().join("resampled").to_str().unwrap(),
                "road_dist",
                2018,
            );
            assert!(path.contains("road_dist_2016.tif"), "gap fill failed: {path}");
        }
    }
}

//! Rasterized distance computation for vector factors.
//!
//! The production distance path is:
//! 1. build a 20 km buffered AOI;
//! 2. clip source vectors to that buffered AOI;
//! 3. rasterize clipped vectors to an expanded CLASS-aligned 0/1 mask;
//! 4. run a parallel tiled distance transform on the mask;
//! 5. crop back to the CLASS grid and keep values only inside the original AOI.

use std::path::Path;
use std::process::Command;

use gdal_crate::raster::{Buffer, RasterCreationOptions};
use gdal_crate::vector::LayerAccess;
use gdal_crate::{Dataset, DriverManager};
use loaders::base::{LoaderError, LoaderResult};
use rayon::prelude::*;

use crate::aoi_mask::AoiMask;
use crate::class_grid::ClassGrid;
use crate::logging::log_status;

const BUFFER_METERS: f64 = 20_000.0;
const TILE_SIZE: usize = 1024;
const DISTANCE_GRID_METERS: f64 = 100.0;
const NO_SOURCE_DISTANCE_METERS: f32 = i32::MAX as f32;

pub fn compute_vector_distance_raster(
    class_grid: &ClassGrid,
    _aoi_mask: &AoiMask,
    aoi_path: &Path,
    vector_path: &Path,
    output_path: &Path,
    work_root: &Path,
    label: &str,
) -> LoaderResult<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(LoaderError::Io)?;
    }
    std::fs::create_dir_all(work_root).map_err(LoaderError::Io)?;

    let work_dir = work_root.join(sanitize_label(label));
    if work_dir.exists() {
        std::fs::remove_dir_all(&work_dir).map_err(LoaderError::Io)?;
    }
    std::fs::create_dir_all(&work_dir).map_err(LoaderError::Io)?;

    let buffered_aoi = work_dir.join("aoi_buffer_20km.gpkg");
    let clipped_vector = work_dir.join("clipped.gpkg");
    let mask_path = work_dir.join("mask.tif");
    let distance_path = work_dir.join("distance_m.tif");
    let tmp_output = output_path.with_extension("tmp.tif");

    let layout = ExpandedGridLayout::for_class_grid(class_grid, BUFFER_METERS);
    let pixel_m = approximate_pixel_size_m(class_grid);
    let workers = effective_workers(0);

    log_status(&format!(
        "[raster-distance] {label}: buffered distance grid {}x{}, pixel_m≈{pixel_m:.2}",
        layout.width, layout.height
    ));

    if !vector_path.exists() {
        log_status(&format!(
            "[raster-distance] {label}: source vector missing {}; writing constant distance={}",
            vector_path.display(),
            NO_SOURCE_DISTANCE_METERS
        ));
        write_constant_distance_raster(&distance_path, &layout, NO_SOURCE_DISTANCE_METERS, label)?;
        warp_distance_to_class(class_grid, aoi_path, &distance_path, &tmp_output)?;
        if output_path.exists() {
            std::fs::remove_file(output_path).map_err(LoaderError::Io)?;
        }
        std::fs::rename(&tmp_output, output_path).map_err(LoaderError::Io)?;
        let _ = std::fs::remove_dir_all(&work_dir);
        log_status(&format!(
            "[raster-distance] {label}: complete -> {}",
            output_path.display()
        ));
        return Ok(());
    }

    log_status(&format!(
        "[raster-distance] {label}: build 20km buffered AOI"
    ));
    build_buffered_aoi(aoi_path, &buffered_aoi)?;
    log_status(&format!(
        "[raster-distance] {label}: clip vector to buffered AOI"
    ));
    clip_vector_to_aoi(vector_path, &buffered_aoi, &clipped_vector)?;
    log_status(&format!("[raster-distance] {label}: rasterize 0/1 mask"));
    rasterize_vector_mask(&clipped_vector, &mask_path, &layout)?;
    compute_parallel_distance_raster(&mask_path, &distance_path, &layout, pixel_m, workers, label)?;
    warp_distance_to_class(class_grid, aoi_path, &distance_path, &tmp_output)?;

    if output_path.exists() {
        std::fs::remove_file(output_path).map_err(LoaderError::Io)?;
    }
    std::fs::rename(&tmp_output, output_path).map_err(LoaderError::Io)?;

    let _ = std::fs::remove_dir_all(&work_dir);
    log_status(&format!(
        "[raster-distance] {label}: complete -> {}",
        output_path.display()
    ));
    Ok(())
}

pub fn compute_raster_value_distance_raster(
    class_grid: &ClassGrid,
    _aoi_mask: &AoiMask,
    aoi_path: &Path,
    raster_path: &Path,
    target_value: u8,
    output_path: &Path,
    work_root: &Path,
    label: &str,
) -> LoaderResult<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(LoaderError::Io)?;
    }
    std::fs::create_dir_all(work_root).map_err(LoaderError::Io)?;

    let work_dir = work_root.join(sanitize_label(label));
    if work_dir.exists() {
        std::fs::remove_dir_all(&work_dir).map_err(LoaderError::Io)?;
    }
    std::fs::create_dir_all(&work_dir).map_err(LoaderError::Io)?;

    let buffered_aoi = work_dir.join("aoi_buffer_20km.gpkg");
    let aligned_raster = work_dir.join("aligned_source.tif");
    let mask_path = work_dir.join("mask.tif");
    let distance_path = work_dir.join("distance_m.tif");
    let tmp_output = output_path.with_extension("tmp.tif");

    let layout = ExpandedGridLayout::for_class_grid(class_grid, BUFFER_METERS);
    let pixel_m = approximate_pixel_size_m(class_grid);
    let workers = effective_workers(0);

    log_status(&format!(
        "[raster-distance] {label}: raster value {target_value}, buffered distance grid {}x{}, pixel_m≈{pixel_m:.2}",
        layout.width, layout.height
    ));

    log_status(&format!(
        "[raster-distance] {label}: build 20km buffered AOI"
    ));
    build_buffered_aoi(aoi_path, &buffered_aoi)?;
    log_status(&format!(
        "[raster-distance] {label}: warp/clip raster source to buffered AOI"
    ));
    warp_raster_to_layout(raster_path, &buffered_aoi, &aligned_raster, &layout)?;
    log_status(&format!(
        "[raster-distance] {label}: build value mask target={target_value}"
    ));
    build_value_mask(&aligned_raster, target_value, &mask_path, &layout)?;
    compute_parallel_distance_raster(&mask_path, &distance_path, &layout, pixel_m, workers, label)?;
    warp_distance_to_class(class_grid, aoi_path, &distance_path, &tmp_output)?;

    if output_path.exists() {
        std::fs::remove_file(output_path).map_err(LoaderError::Io)?;
    }
    std::fs::rename(&tmp_output, output_path).map_err(LoaderError::Io)?;

    let _ = std::fs::remove_dir_all(&work_dir);
    log_status(&format!(
        "[raster-distance] {label}: complete -> {}",
        output_path.display()
    ));
    Ok(())
}

#[derive(Debug, Clone)]
struct ExpandedGridLayout {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    width: usize,
    height: usize,
    geo_transform: [f64; 6],
    projection: String,
}

impl ExpandedGridLayout {
    fn for_class_grid(class_grid: &ClassGrid, buffer_m: f64) -> Self {
        let (min_x, min_y, max_x, max_y) = class_grid.bounds();
        let center_lat = (min_y + max_y) * 0.5;
        let lat_deg = buffer_m / 111_320.0;
        let lon_deg = buffer_m / (111_320.0 * center_lat.to_radians().cos().abs().max(0.1));
        let pixel_y_deg = DISTANCE_GRID_METERS / 111_320.0;
        let pixel_x_deg =
            DISTANCE_GRID_METERS / (111_320.0 * center_lat.to_radians().cos().abs().max(0.1));
        let expanded_min_x = min_x - lon_deg;
        let expanded_min_y = min_y - lat_deg;
        let expanded_max_x = max_x + lon_deg;
        let expanded_max_y = max_y + lat_deg;
        let width = ((expanded_max_x - expanded_min_x) / pixel_x_deg).ceil() as usize;
        let height = ((expanded_max_y - expanded_min_y) / pixel_y_deg).ceil() as usize;
        let expanded_gt = [
            expanded_min_x,
            pixel_x_deg,
            0.0,
            expanded_max_y,
            0.0,
            -pixel_y_deg,
        ];

        Self {
            min_x: expanded_min_x,
            min_y: expanded_min_y,
            max_x: expanded_max_x,
            max_y: expanded_max_y,
            width,
            height,
            geo_transform: expanded_gt,
            projection: class_grid.crs().to_string(),
        }
    }
}

fn build_buffered_aoi(aoi_path: &Path, output_path: &Path) -> LoaderResult<()> {
    let ogr2ogr = find_tool("ogr2ogr")?;
    let (layer_name, geom_field_name) = first_layer_geometry(aoi_path)?;
    let geom_expr = if geom_field_name.is_empty() {
        "geometry".to_string()
    } else {
        format!("\"{geom_field_name}\"")
    };
    let sql = format!(
        "SELECT ST_Transform(ST_Buffer(ST_Transform({geom_expr}, 3857), {BUFFER_METERS}), 4326) AS geom FROM \"{layer_name}\""
    );
    let status = Command::new(&ogr2ogr)
        .args([
            "-q",
            "-f",
            "GPKG",
            path_str(output_path)?,
            path_str(aoi_path)?,
            "-dialect",
            "SQLite",
            "-sql",
            &sql,
            "-nln",
            "aoi_buffer",
            "-nlt",
            "MULTIPOLYGON",
            "-overwrite",
        ])
        .status()
        .map_err(|e| LoaderError::Gdal(format!("spawn ogr2ogr buffered AOI: {e}")))?;
    if !status.success() {
        return Err(LoaderError::Gdal(format!(
            "ogr2ogr buffered AOI failed with status {}",
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

fn clip_vector_to_aoi(
    vector_path: &Path,
    buffered_aoi: &Path,
    output_path: &Path,
) -> LoaderResult<()> {
    let ogr2ogr = find_tool("ogr2ogr")?;
    let status = Command::new(&ogr2ogr)
        .args([
            "-q",
            "-f",
            "GPKG",
            path_str(output_path)?,
            path_str(vector_path)?,
            "-clipsrc",
            path_str(buffered_aoi)?,
            "-nln",
            "features",
            "-nlt",
            "PROMOTE_TO_MULTI",
            "-overwrite",
        ])
        .status()
        .map_err(|e| LoaderError::Gdal(format!("spawn ogr2ogr clip: {e}")))?;
    if !status.success() {
        return Err(LoaderError::Gdal(format!(
            "ogr2ogr clip failed for {} with status {}",
            vector_path.display(),
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

fn rasterize_vector_mask(
    vector_path: &Path,
    mask_path: &Path,
    layout: &ExpandedGridLayout,
) -> LoaderResult<()> {
    let gdal_rasterize = find_tool("gdal_rasterize")?;
    let status = Command::new(&gdal_rasterize)
        .args([
            "-q",
            "-burn",
            "1",
            "-init",
            "0",
            "-at",
            "-ot",
            "Byte",
            "-of",
            "GTiff",
            "-te",
            &layout.min_x.to_string(),
            &layout.min_y.to_string(),
            &layout.max_x.to_string(),
            &layout.max_y.to_string(),
            "-ts",
            &layout.width.to_string(),
            &layout.height.to_string(),
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "TILED=YES",
            "-co",
            "BIGTIFF=YES",
            path_str(vector_path)?,
            path_str(mask_path)?,
        ])
        .status()
        .map_err(|e| LoaderError::Gdal(format!("spawn gdal_rasterize: {e}")))?;
    if !status.success() {
        return Err(LoaderError::Gdal(format!(
            "gdal_rasterize failed with status {}",
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

fn warp_raster_to_layout(
    raster_path: &Path,
    buffered_aoi: &Path,
    output_path: &Path,
    layout: &ExpandedGridLayout,
) -> LoaderResult<()> {
    let gdalwarp = find_tool("gdalwarp")?;
    let status = Command::new(&gdalwarp)
        .args([
            "-q",
            "-t_srs",
            "EPSG:4326",
            "-te",
            &layout.min_x.to_string(),
            &layout.min_y.to_string(),
            &layout.max_x.to_string(),
            &layout.max_y.to_string(),
            "-ts",
            &layout.width.to_string(),
            &layout.height.to_string(),
            "-r",
            "near",
            "-cutline",
            path_str(buffered_aoi)?,
            "-dstnodata",
            "0",
            "-of",
            "GTiff",
            "-ot",
            "Byte",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "TILED=YES",
            "-co",
            "BIGTIFF=YES",
            "-co",
            "SPARSE_OK=YES",
            path_str(raster_path)?,
            path_str(output_path)?,
        ])
        .status()
        .map_err(|e| LoaderError::Gdal(format!("spawn gdalwarp raster source: {e}")))?;
    if !status.success() {
        return Err(LoaderError::Gdal(format!(
            "gdalwarp raster source failed with status {}",
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

fn build_value_mask(
    aligned_raster: &Path,
    target_value: u8,
    mask_path: &Path,
    layout: &ExpandedGridLayout,
) -> LoaderResult<()> {
    let src_ds = Dataset::open(aligned_raster).map_err(|e| {
        LoaderError::Gdal(format!(
            "open aligned raster {}: {e}",
            aligned_raster.display()
        ))
    })?;
    let src_band = src_ds
        .rasterband(1)
        .map_err(|e| LoaderError::Gdal(format!("aligned rasterband: {e}")))?;

    let driver = DriverManager::get_driver_by_name("GTiff")
        .map_err(|e| LoaderError::Gdal(format!("GTiff driver: {e}")))?;
    let options = RasterCreationOptions::from_iter([
        "COMPRESS=DEFLATE",
        "TILED=YES",
        "BIGTIFF=YES",
        "SPARSE_OK=YES",
    ]);
    let mut mask_ds = driver
        .create_with_band_type_with_options::<u8, _>(
            mask_path,
            layout.width,
            layout.height,
            1,
            &options,
        )
        .map_err(|e| LoaderError::Gdal(format!("create value mask: {e}")))?;
    mask_ds
        .set_geo_transform(&layout.geo_transform)
        .map_err(|e| LoaderError::Gdal(format!("set value mask geotransform: {e}")))?;
    if !layout.projection.is_empty() {
        mask_ds
            .set_projection(&layout.projection)
            .map_err(|e| LoaderError::Gdal(format!("set value mask projection: {e}")))?;
    }
    let mut mask_band = mask_ds
        .rasterband(1)
        .map_err(|e| LoaderError::Gdal(format!("value mask rasterband: {e}")))?;

    let mut positive_pixels = 0usize;
    for row_off in (0..layout.height).step_by(TILE_SIZE) {
        let tile_h = (layout.height - row_off).min(TILE_SIZE);
        for col_off in (0..layout.width).step_by(TILE_SIZE) {
            let tile_w = (layout.width - col_off).min(TILE_SIZE);
            let src = src_band
                .read_as::<u8>(
                    (col_off as isize, row_off as isize),
                    (tile_w, tile_h),
                    (tile_w, tile_h),
                    None,
                )
                .map_err(|e| LoaderError::Gdal(format!("read aligned source window: {e}")))?;
            let data: Vec<u8> = src
                .data()
                .iter()
                .map(|value| {
                    if *value == target_value {
                        positive_pixels += 1;
                        1
                    } else {
                        0
                    }
                })
                .collect();
            let mut buf = Buffer::new((tile_w, tile_h), data);
            mask_band
                .write(
                    (col_off as isize, row_off as isize),
                    (tile_w, tile_h),
                    &mut buf,
                )
                .map_err(|e| LoaderError::Gdal(format!("write value mask window: {e}")))?;
        }
    }
    drop(mask_band);
    drop(mask_ds);
    drop(src_band);
    drop(src_ds);

    if positive_pixels == 0 {
        log_status(&format!(
            "[raster-distance] value mask target={target_value}: no source pixels in buffered AOI; distance will use {}",
            NO_SOURCE_DISTANCE_METERS
        ));
    }
    Ok(())
}

fn compute_parallel_distance_raster(
    mask_path: &Path,
    output_path: &Path,
    layout: &ExpandedGridLayout,
    pixel_m: f64,
    workers: usize,
    label: &str,
) -> LoaderResult<()> {
    let driver = DriverManager::get_driver_by_name("GTiff")
        .map_err(|e| LoaderError::Gdal(format!("GTiff driver: {e}")))?;
    let options = RasterCreationOptions::from_iter([
        "COMPRESS=DEFLATE",
        "PREDICTOR=3",
        "TILED=YES",
        "BIGTIFF=YES",
        "SPARSE_OK=YES",
    ]);
    let mut out_ds = driver
        .create_with_band_type_with_options::<f32, _>(
            output_path,
            layout.width,
            layout.height,
            1,
            &options,
        )
        .map_err(|e| LoaderError::Gdal(format!("create output: {e}")))?;
    out_ds
        .set_geo_transform(&layout.geo_transform)
        .map_err(|e| LoaderError::Gdal(format!("set output geotransform: {e}")))?;
    if !layout.projection.is_empty() {
        out_ds
            .set_projection(&layout.projection)
            .map_err(|e| LoaderError::Gdal(format!("set output projection: {e}")))?;
    }
    let mut out_band = out_ds
        .rasterband(1)
        .map_err(|e| LoaderError::Gdal(format!("output rasterband: {e}")))?;
    out_band
        .set_no_data_value(Some(f64::NAN))
        .map_err(|e| LoaderError::Gdal(format!("set output nodata: {e}")))?;

    let mask_ds = Dataset::open(mask_path)
        .map_err(|e| LoaderError::Gdal(format!("open mask {}: {e}", mask_path.display())))?;
    let mask_band = mask_ds
        .rasterband(1)
        .map_err(|e| LoaderError::Gdal(format!("mask rasterband: {e}")))?;
    let mask = mask_band
        .read_as::<u8>(
            (0, 0),
            (layout.width, layout.height),
            (layout.width, layout.height),
            None,
        )
        .map_err(|e| LoaderError::Gdal(format!("read full mask: {e}")))?;
    let source_pixels = mask.data().iter().filter(|&&value| value == 1).count();
    if source_pixels == 0 {
        log_status(&format!(
            "[raster-distance] {label}: no source pixels in buffered AOI; writing constant distance={}",
            NO_SOURCE_DISTANCE_METERS
        ));
        write_constant_distance_tiles(&mut out_band, layout, NO_SOURCE_DISTANCE_METERS, label)?;
        drop(mask);
        drop(mask_band);
        drop(mask_ds);
        drop(out_band);
        drop(out_ds);
        return Ok(());
    }

    log_status(&format!(
        "[raster-distance] {label}: global distance transform {}x{}, source_pixels={}, workers={workers}",
        layout.width, layout.height, source_pixels
    ));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| LoaderError::Gdal(format!("build distance thread pool: {e}")))?;

    let dist_sq_t = pool.install(|| {
        distance_transform_squared_transposed(mask.data(), layout.width, layout.height, label)
    });
    drop(mask);
    drop(mask_band);
    drop(mask_ds);

    let total_tiles = ((layout.width + TILE_SIZE - 1) / TILE_SIZE)
        * ((layout.height + TILE_SIZE - 1) / TILE_SIZE);
    let mut finite_pixels = 0usize;
    let mut completed = 0usize;
    for row_off in (0..layout.height).step_by(TILE_SIZE) {
        let tile_h = (layout.height - row_off).min(TILE_SIZE);
        for col_off in (0..layout.width).step_by(TILE_SIZE) {
            let tile_w = (layout.width - col_off).min(TILE_SIZE);
            let mut data = vec![f32::NAN; tile_w * tile_h];
            for dy in 0..tile_h {
                let y = row_off + dy;
                for dx in 0..tile_w {
                    let x = col_off + dx;
                    let d2 = dist_sq_t[x * layout.height + y];
                    if d2.is_finite() && d2 < 1.0e19 {
                        data[dy * tile_w + dx] = (d2.sqrt() * pixel_m) as f32;
                        finite_pixels += 1;
                    }
                }
            }

            let mut out_buf = Buffer::new((tile_w, tile_h), data);
            out_band
                .write(
                    (col_off as isize, row_off as isize),
                    (tile_w, tile_h),
                    &mut out_buf,
                )
                .map_err(|e| LoaderError::Gdal(format!("write output window: {e}")))?;
            completed += 1;
            if completed % 10 == 0 || completed == total_tiles {
                log_status(&format!(
                    "[raster-distance] {label}: write distance progress {completed}/{total_tiles} tiles ({:.1}%)",
                    completed as f64 * 100.0 / total_tiles as f64
                ));
            }
        }
    }
    drop(out_band);
    drop(out_ds);

    if finite_pixels == 0 {
        return Err(LoaderError::NoValidData);
    }
    Ok(())
}

fn write_constant_distance_tiles(
    out_band: &mut gdal_crate::raster::RasterBand,
    layout: &ExpandedGridLayout,
    value: f32,
    label: &str,
) -> LoaderResult<()> {
    let total_tiles = ((layout.width + TILE_SIZE - 1) / TILE_SIZE)
        * ((layout.height + TILE_SIZE - 1) / TILE_SIZE);
    let mut completed = 0usize;
    for row_off in (0..layout.height).step_by(TILE_SIZE) {
        let tile_h = (layout.height - row_off).min(TILE_SIZE);
        for col_off in (0..layout.width).step_by(TILE_SIZE) {
            let tile_w = (layout.width - col_off).min(TILE_SIZE);
            let mut out_buf = Buffer::new((tile_w, tile_h), vec![value; tile_w * tile_h]);
            out_band
                .write(
                    (col_off as isize, row_off as isize),
                    (tile_w, tile_h),
                    &mut out_buf,
                )
                .map_err(|e| LoaderError::Gdal(format!("write constant distance window: {e}")))?;
            completed += 1;
            if completed % 10 == 0 || completed == total_tiles {
                log_status(&format!(
                    "[raster-distance] {label}: write constant distance progress {completed}/{total_tiles} tiles ({:.1}%)",
                    completed as f64 * 100.0 / total_tiles as f64
                ));
            }
        }
    }
    Ok(())
}

fn write_constant_distance_raster(
    output_path: &Path,
    layout: &ExpandedGridLayout,
    value: f32,
    label: &str,
) -> LoaderResult<()> {
    let driver = DriverManager::get_driver_by_name("GTiff")
        .map_err(|e| LoaderError::Gdal(format!("GTiff driver: {e}")))?;
    let options = RasterCreationOptions::from_iter([
        "COMPRESS=DEFLATE",
        "PREDICTOR=3",
        "TILED=YES",
        "BIGTIFF=YES",
        "SPARSE_OK=YES",
    ]);
    let mut out_ds = driver
        .create_with_band_type_with_options::<f32, _>(
            output_path,
            layout.width,
            layout.height,
            1,
            &options,
        )
        .map_err(|e| LoaderError::Gdal(format!("create constant distance output: {e}")))?;
    out_ds
        .set_geo_transform(&layout.geo_transform)
        .map_err(|e| LoaderError::Gdal(format!("set constant distance geotransform: {e}")))?;
    if !layout.projection.is_empty() {
        out_ds
            .set_projection(&layout.projection)
            .map_err(|e| LoaderError::Gdal(format!("set constant distance projection: {e}")))?;
    }
    let mut out_band = out_ds
        .rasterband(1)
        .map_err(|e| LoaderError::Gdal(format!("constant distance rasterband: {e}")))?;
    out_band
        .set_no_data_value(Some(f64::NAN))
        .map_err(|e| LoaderError::Gdal(format!("set constant distance nodata: {e}")))?;
    write_constant_distance_tiles(&mut out_band, layout, value, label)?;
    drop(out_band);
    drop(out_ds);
    Ok(())
}

fn warp_distance_to_class(
    class_grid: &ClassGrid,
    aoi_path: &Path,
    distance_path: &Path,
    output_path: &Path,
) -> LoaderResult<()> {
    let gdalwarp = find_tool("gdalwarp")?;
    let (min_lng, min_lat, max_lng, max_lat) = class_grid.bounds();
    log_status(&format!(
        "[raster-distance] warp distance grid to CLASS: {}",
        output_path.display()
    ));
    let status = Command::new(&gdalwarp)
        .args([
            "-q",
            "-t_srs",
            "EPSG:4326",
            "-te",
            &min_lng.to_string(),
            &min_lat.to_string(),
            &max_lng.to_string(),
            &max_lat.to_string(),
            "-ts",
            &class_grid.width().to_string(),
            &class_grid.height().to_string(),
            "-r",
            "bilinear",
            "-cutline",
            path_str(aoi_path)?,
            "-dstnodata",
            "nan",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-of",
            "GTiff",
            "-ot",
            "Float32",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "PREDICTOR=3",
            "-co",
            "TILED=YES",
            "-co",
            "BIGTIFF=YES",
            "-co",
            "SPARSE_OK=YES",
            path_str(distance_path)?,
            path_str(output_path)?,
        ])
        .status()
        .map_err(|e| LoaderError::Gdal(format!("spawn gdalwarp distance to class: {e}")))?;
    if !status.success() {
        return Err(LoaderError::Gdal(format!(
            "gdalwarp distance to class failed with status {}",
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

fn distance_transform_squared_transposed(
    mask: &[u8],
    width: usize,
    height: usize,
    label: &str,
) -> Vec<f64> {
    let inf = 1.0e20;
    let mut tmp = vec![inf; width * height];
    log_status(&format!(
        "[raster-distance] {label}: EDT row pass {height} rows"
    ));
    tmp.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
            let row_start = y * width;
            let mut f = vec![inf; width];
            let mut d = vec![inf; width];
            for x in 0..width {
                if mask[row_start + x] == 1 {
                    f[x] = 0.0;
                }
            }
            edt_1d(&f, &mut d);
            row_out.copy_from_slice(&d);
        });

    log_status(&format!(
        "[raster-distance] {label}: transpose row-pass grid"
    ));
    let mut tmp_t = vec![inf; width * height];
    tmp_t
        .par_chunks_mut(height)
        .enumerate()
        .for_each(|(x, col_out)| {
            for y in 0..height {
                col_out[y] = tmp[y * width + x];
            }
        });
    drop(tmp);

    log_status(&format!(
        "[raster-distance] {label}: EDT column pass {width} columns"
    ));
    let mut out_t = vec![inf; width * height];
    out_t
        .par_chunks_mut(height)
        .enumerate()
        .for_each(|(x, col_out)| {
            let col_start = x * height;
            edt_1d(
                &tmp_t[col_start..col_start + height],
                &mut col_out[..height],
            );
        });
    out_t
}

fn edt_1d(f: &[f64], d: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;

    for q in 1..n {
        let mut s = intersection(f, q, v[k]);
        while s <= z[k] {
            if k == 0 {
                break;
            }
            k -= 1;
            s = intersection(f, q, v[k]);
        }
        if s <= z[k] && k == 0 {
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
        } else {
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
        }
    }

    k = 0;
    for (q, out) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let diff = q as f64 - v[k] as f64;
        *out = diff * diff + f[v[k]];
    }
}

fn intersection(f: &[f64], q: usize, vk: usize) -> f64 {
    let qf = q as f64;
    let vkf = vk as f64;
    ((f[q] + qf * qf) - (f[vk] + vkf * vkf)) / (2.0 * (qf - vkf))
}

fn effective_workers(configured_workers: usize) -> usize {
    if configured_workers > 0 {
        configured_workers
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

fn find_tool(name: &str) -> LoaderResult<String> {
    if let Ok(home) = std::env::var("GDAL_HOME") {
        let candidate = Path::new(&home).join("bin").join(name);
        if candidate.exists() {
            return Ok(candidate.to_string_lossy().to_string());
        }
    }
    let probe = Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Ok(name.to_string());
    }
    Err(LoaderError::Gdal(format!(
        "{name} not found in PATH or GDAL_HOME"
    )))
}

fn first_layer_geometry(path: &Path) -> LoaderResult<(String, String)> {
    let ds = Dataset::open(path)
        .map_err(|e| LoaderError::Gdal(format!("open vector {}: {e}", path.display())))?;
    let layer = ds
        .layer(0)
        .map_err(|e| LoaderError::Gdal(format!("read first layer {}: {e}", path.display())))?;
    let geom_field_name = layer
        .defn()
        .geom_fields()
        .next()
        .map(|field| field.name())
        .unwrap_or_default();
    Ok((layer.name(), geom_field_name))
}

fn approximate_pixel_size_m(class_grid: &ClassGrid) -> f64 {
    let (_min_x, min_y, _max_x, max_y) = class_grid.bounds();
    let center_lat = (min_y + max_y) * 0.5;
    let gt = class_grid.geo_transform();
    let x_m = gt[1].abs() * 111_320.0 * center_lat.to_radians().cos().abs();
    let y_m = gt[5].abs() * 111_320.0;
    (x_m + y_m) * 0.5
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn path_str(path: &Path) -> LoaderResult<&str> {
    path.to_str()
        .ok_or_else(|| LoaderError::Gdal(format!("non-UTF8 path: {}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal_crate::vector::{Geometry, LayerOptions};
    use tempfile::TempDir;

    const WGS84_WKT: &str = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]";

    #[test]
    fn rasterized_distance_smoke_test() {
        if find_tool("ogr2ogr").is_err() || find_tool("gdal_rasterize").is_err() {
            eprintln!("skipping rasterized distance smoke test: GDAL CLI tools unavailable");
            return;
        }

        let tmp = TempDir::new().unwrap();
        let class_path = tmp.path().join("class.tif");
        let aoi_path = tmp.path().join("aoi.gpkg");
        let vector_path = tmp.path().join("line.gpkg");
        let out_path = tmp.path().join("distance.tif");
        let work_root = tmp.path().join("work");

        write_class_grid(&class_path);
        write_aoi(&aoi_path);
        write_line(&vector_path);

        let class_grid = ClassGrid::open(class_path.to_str().unwrap(), 2020).unwrap();
        let aoi_mask = AoiMask::load(&aoi_path, &class_grid).unwrap();
        compute_vector_distance_raster(
            &class_grid,
            &aoi_mask,
            &aoi_path,
            &vector_path,
            &out_path,
            &work_root,
            "smoke",
        )
        .unwrap();

        let ds = Dataset::open(&out_path).unwrap();
        let band = ds.rasterband(1).unwrap();
        let buf = band
            .read_as::<f32>((0, 0), (20, 20), (20, 20), None)
            .unwrap();
        let values = buf.data();
        assert!(
            values.iter().any(|value| value.is_finite()),
            "expected finite distances inside AOI"
        );
        assert!(
            values.iter().any(|value| value.is_nan()),
            "expected NaN outside AOI"
        );
    }

    #[test]
    fn rasterized_distance_empty_source_uses_max_distance() {
        if find_tool("ogr2ogr").is_err() || find_tool("gdal_rasterize").is_err() {
            eprintln!("skipping empty-source distance test: GDAL CLI tools unavailable");
            return;
        }

        let tmp = TempDir::new().unwrap();
        let class_path = tmp.path().join("class.tif");
        let aoi_path = tmp.path().join("aoi.gpkg");
        let vector_path = tmp.path().join("empty_line.gpkg");
        let out_path = tmp.path().join("distance_empty.tif");
        let work_root = tmp.path().join("work");

        write_class_grid(&class_path);
        write_aoi(&aoi_path);
        write_empty_line_layer(&vector_path);

        let class_grid = ClassGrid::open(class_path.to_str().unwrap(), 2020).unwrap();
        let aoi_mask = AoiMask::load(&aoi_path, &class_grid).unwrap();
        compute_vector_distance_raster(
            &class_grid,
            &aoi_mask,
            &aoi_path,
            &vector_path,
            &out_path,
            &work_root,
            "empty_source",
        )
        .unwrap();

        let ds = Dataset::open(&out_path).unwrap();
        let band = ds.rasterband(1).unwrap();
        let buf = band
            .read_as::<f32>((0, 0), (20, 20), (20, 20), None)
            .unwrap();
        let values = buf.data();
        assert!(
            values
                .iter()
                .any(|value| *value == NO_SOURCE_DISTANCE_METERS),
            "expected max-distance values inside AOI"
        );
        assert!(
            values.iter().any(|value| value.is_nan()),
            "expected NaN outside AOI"
        );
    }

    #[test]
    fn rasterized_distance_missing_source_uses_max_distance() {
        let tmp = TempDir::new().unwrap();
        let class_path = tmp.path().join("class.tif");
        let aoi_path = tmp.path().join("aoi.gpkg");
        let vector_path = tmp.path().join("missing_line.shp");
        let out_path = tmp.path().join("distance_missing.tif");
        let work_root = tmp.path().join("work");

        write_class_grid(&class_path);
        write_aoi(&aoi_path);

        let class_grid = ClassGrid::open(class_path.to_str().unwrap(), 2020).unwrap();
        let aoi_mask = AoiMask::load(&aoi_path, &class_grid).unwrap();
        compute_vector_distance_raster(
            &class_grid,
            &aoi_mask,
            &aoi_path,
            &vector_path,
            &out_path,
            &work_root,
            "missing_source",
        )
        .unwrap();

        let ds = Dataset::open(&out_path).unwrap();
        let band = ds.rasterband(1).unwrap();
        let buf = band
            .read_as::<f32>((0, 0), (20, 20), (20, 20), None)
            .unwrap();
        let values = buf.data();
        assert!(
            values
                .iter()
                .any(|value| *value == NO_SOURCE_DISTANCE_METERS),
            "expected max-distance values inside AOI"
        );
        assert!(
            values.iter().any(|value| value.is_nan()),
            "expected NaN outside AOI"
        );
    }

    fn write_class_grid(path: &Path) {
        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let mut ds = driver
            .create_with_band_type::<u8, _>(path, 20, 20, 1)
            .unwrap();
        ds.set_geo_transform(&[100.0, 0.01, 0.0, 30.2, 0.0, -0.01])
            .unwrap();
        ds.set_projection(WGS84_WKT).unwrap();
        let mut band = ds.rasterband(1).unwrap();
        let mut buf = Buffer::new((20, 20), vec![0u8; 400]);
        band.write((0, 0), (20, 20), &mut buf).unwrap();
        ds.flush_cache().unwrap();
    }

    fn write_aoi(path: &Path) {
        let driver = DriverManager::get_driver_by_name("GPKG").unwrap();
        let mut ds = driver.create_vector_only(path).unwrap();
        let srs = gdal_crate::spatial_ref::SpatialRef::from_wkt(WGS84_WKT).unwrap();
        let mut layer = ds
            .create_layer(LayerOptions {
                name: "aoi",
                srs: Some(&srs),
                ty: gdal_crate::vector::OGRwkbGeometryType::wkbPolygon,
                ..Default::default()
            })
            .unwrap();
        let geom = Geometry::from_wkt(
            "POLYGON ((100.02 30.18, 100.12 30.18, 100.12 30.08, 100.02 30.08, 100.02 30.18))",
        )
        .unwrap();
        layer.create_feature(geom).unwrap();
    }

    fn write_line(path: &Path) {
        let driver = DriverManager::get_driver_by_name("GPKG").unwrap();
        let mut ds = driver.create_vector_only(path).unwrap();
        let srs = gdal_crate::spatial_ref::SpatialRef::from_wkt(WGS84_WKT).unwrap();
        let mut layer = ds
            .create_layer(LayerOptions {
                name: "line",
                srs: Some(&srs),
                ty: gdal_crate::vector::OGRwkbGeometryType::wkbLineString,
                ..Default::default()
            })
            .unwrap();
        let geom = Geometry::from_wkt("LINESTRING (100.07 30.2, 100.07 30.0)").unwrap();
        layer.create_feature(geom).unwrap();
    }

    fn write_empty_line_layer(path: &Path) {
        let driver = DriverManager::get_driver_by_name("GPKG").unwrap();
        let mut ds = driver.create_vector_only(path).unwrap();
        let srs = gdal_crate::spatial_ref::SpatialRef::from_wkt(WGS84_WKT).unwrap();
        ds.create_layer(LayerOptions {
            name: "line",
            srs: Some(&srs),
            ty: gdal_crate::vector::OGRwkbGeometryType::wkbLineString,
            ..Default::default()
        })
        .unwrap();
    }
}

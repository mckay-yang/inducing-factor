//! Raster resampling — reproject data sources to CLASS reference grid.
//!
//! Uses `gdalwarp` (and `gdal_translate` for band extraction) to resample
//! DEM, CCDC, rainfall, and temperature rasters to the CLASS grid dimensions
//! and extent.  Bilinear interpolation is used for continuous data (DEM,
//! rainfall, temperature) and nearest-neighbour for classification data (CCDC).

use std::path::Path;
use std::process::Command;

use if_loaders::base::{LoaderError, LoaderResult};

use crate::class_grid::ClassGrid;

fn find_gdalwarp() -> Option<String> {
    if let Ok(home) = std::env::var("GDAL_HOME") {
        let candidate = Path::new(&home).join("bin").join("gdalwarp");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    let probe = Command::new("gdalwarp")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Some("gdalwarp".to_string());
    }
    None
}

fn find_gdal_translate() -> Option<String> {
    if let Ok(home) = std::env::var("GDAL_HOME") {
        let candidate = Path::new(&home).join("bin").join("gdal_translate");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    let probe = Command::new("gdal_translate")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Some("gdal_translate".to_string());
    }
    None
}

fn find_gdalbuildvrt() -> Option<String> {
    if let Ok(home) = std::env::var("GDAL_HOME") {
        let candidate = Path::new(&home).join("bin").join("gdalbuildvrt");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    let probe = Command::new("gdalbuildvrt")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Some("gdalbuildvrt".to_string());
    }
    None
}

/// Resample the DEM raster to match the CLASS reference grid.
///
/// * `class_grid` — reference grid providing target dimensions, CRS, and bounds
/// * `source_path` — path to the source DEM raster (EPSG:32645, int16, NoData=-9999)
/// * `output_path` — path for the output resampled raster (EPSG:4326, float32, NoData=NaN)
///
/// The output raster will have the same dimensions, geotransform, and CRS as
/// the CLASS reference grid.  NoData values (-9999) in the source are converted
/// to NaN in the output.  Bilinear interpolation is used for resampling.
pub fn resample_dem_to_class(
    class_grid: &ClassGrid,
    source_path: &str,
    output_path: &str,
) -> LoaderResult<()> {
    let gdalwarp = find_gdalwarp()
        .ok_or_else(|| LoaderError::Gdal("gdalwarp not found in PATH or GDAL_HOME".into()))?;

    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| LoaderError::Io(e))?;
    }

    let (min_lng, min_lat, max_lng, max_lat) = class_grid.bounds();
    let width = class_grid.width();
    let height = class_grid.height();

    let status = Command::new(&gdalwarp)
        .args([
            "-t_srs",
            "EPSG:4326",
            "-te",
            &format!("{min_lng}"),
            &format!("{min_lat}"),
            &format!("{max_lng}"),
            &format!("{max_lat}"),
            "-ts",
            &format!("{width}"),
            &format!("{height}"),
            "-r",
            "bilinear",
            "-srcnodata",
            "-9999",
            "-dstnodata",
            "nan",
            "-of",
            "GTiff",
            "-ot",
            "Float32",
            "-co",
            "COMPRESS=LZW",
            source_path,
            output_path,
        ])
        .status()
        .map_err(|e| LoaderError::Gdal(format!("failed to spawn gdalwarp: {e}")))?;

    if !status.success() {
        return Err(LoaderError::Gdal(format!(
            "gdalwarp exited with status {}",
            status
        )));
    }

    Ok(())
}

/// Resample all CCDC annual disturbance timing rasters to match the CLASS grid.
///
/// Scans `ccdc_dir` for `YYYY.tif` files, resamples each via `gdalwarp` with
/// nearest-neighbour to preserve disturbance year values (classification data,
/// no interpolation allowed), and writes the result to `output_dir/YYYY.tif`.
///
/// * `class_grid` — reference grid providing target dimensions, bounds, and CRS
/// * `ccdc_dir`   — directory containing source CCDC rasters named `YYYY.tif`
/// * `output_dir` — directory for resampled rasters (created if missing)
///
/// Returns the list of successfully processed years.  Failed years are logged
/// (via `log::warn`) but do **not** halt the remaining batch.
pub fn resample_ccdc_to_class(
    class_grid: &ClassGrid,
    ccdc_dir: &str,
    output_dir: &str,
) -> LoaderResult<Vec<i32>> {
    let gdalwarp = find_gdalwarp()
        .ok_or_else(|| LoaderError::Gdal("gdalwarp not found in PATH or GDAL_HOME".into()))?;

    // Discover years from filesystem.
    let mut years: Vec<i32> = Vec::new();
    let dir_entries = std::fs::read_dir(ccdc_dir)
        .map_err(|e| LoaderError::Io(std::io::Error::new(
            e.kind(),
            format!("cannot read ccdc_dir {ccdc_dir}: {e}"),
        )))?;

    for entry in dir_entries.flatten() {
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        if fname.ends_with(".tif") {
            if let Ok(year) = fname.trim_end_matches(".tif").parse::<i32>() {
                years.push(year);
            }
        }
    }
    years.sort_unstable();

    if years.is_empty() {
        return Err(LoaderError::FileNotFound(format!(
            "no YYYY.tif files found in {ccdc_dir}"
        )));
    }

    // Ensure output directory exists.
    std::fs::create_dir_all(output_dir)
        .map_err(|e| LoaderError::Io(e))?;

    let (min_lng, min_lat, max_lng, max_lat) = class_grid.bounds();
    let width = class_grid.width();
    let height = class_grid.height();

    let mut processed: Vec<i32> = Vec::with_capacity(years.len());

    for year in &years {
        let src_path = format!("{ccdc_dir}/{year}.tif");
        let out_path = format!("{output_dir}/{year}.tif");

        if !Path::new(&src_path).exists() {
            log::warn!("CCDC source file not found, skipping: {src_path}");
            continue;
        }

        let status = Command::new(&gdalwarp)
            .args([
                "-te",
                &format!("{min_lng}"),
                &format!("{min_lat}"),
                &format!("{max_lng}"),
                &format!("{max_lat}"),
                "-ts",
                &format!("{width}"),
                &format!("{height}"),
                "-r",
                "near",
                "-of",
                "GTiff",
                "-ot",
                "Float32",
                "-co",
                "COMPRESS=LZW",
                &src_path,
                &out_path,
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                processed.push(*year);
            }
            Ok(s) => {
                log::warn!(
                    "gdalwarp failed for year {year} (exit code {}), skipping",
                    s.code().map_or(-1, |c| c)
                );
            }
            Err(e) => {
                log::warn!("failed to spawn gdalwarp for year {year}: {e}, skipping");
            }
        }
    }

    Ok(processed)
}

/// Resample all monthly rainfall rasters to match the CLASS grid.
///
/// Scans `rainfall_dir` for `pre??????.tif` files (format `preYYYYMM.tif`),
/// resamples each via `gdalwarp` with bilinear interpolation, and writes the
/// result to `output_dir/<original_filename>`.  Source files are EPSG:4326
/// float32 single‑band GeoTIFFs.
///
/// * `class_grid`   — reference grid providing target dimensions, bounds, and CRS
/// * `rainfall_dir` — directory containing monthly rainfall rasters
/// * `output_dir`   — directory for resampled rasters (created if missing)
///
/// Returns the list of output file paths successfully created.  Failed months
/// are logged (via `log::warn`) but do **not** halt the remaining batch.
/// Existing output files are skipped (resume support).
pub fn resample_rainfall_to_class(
    class_grid: &ClassGrid,
    rainfall_dir: &str,
    output_dir: &str,
) -> LoaderResult<Vec<String>> {
    let gdalwarp = find_gdalwarp()
        .ok_or_else(|| LoaderError::Gdal("gdalwarp not found in PATH or GDAL_HOME".into()))?;

    // Discover rainfall files from filesystem.
    let mut files: Vec<String> = Vec::new();
    let dir_entries = std::fs::read_dir(rainfall_dir).map_err(|e| {
        LoaderError::Io(std::io::Error::new(
            e.kind(),
            format!("cannot read rainfall_dir {rainfall_dir}: {e}"),
        ))
    })?;

    for entry in dir_entries.flatten() {
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        if fname.starts_with("pre") && fname.ends_with(".tif") {
            files.push(fname.to_string());
        }
    }
    files.sort();

    if files.is_empty() {
        return Err(LoaderError::FileNotFound(format!(
            "no pre*.tif files found in {rainfall_dir}"
        )));
    }

    std::fs::create_dir_all(output_dir).map_err(LoaderError::Io)?;

    let (min_lng, min_lat, max_lng, max_lat) = class_grid.bounds();
    let width = class_grid.width();
    let height = class_grid.height();

    let mut processed: Vec<String> = Vec::with_capacity(files.len());

    for fname in &files {
        let src_path = format!("{rainfall_dir}/{fname}");
        let out_path = format!("{output_dir}/{fname}");

        // Resume: skip already-resampled files.
        if Path::new(&out_path).exists() {
            log::info!("rainfall {fname} already resampled, skipping");
            processed.push(out_path.clone());
            continue;
        }

        if !Path::new(&src_path).exists() {
            log::warn!("rainfall source file not found, skipping: {src_path}");
            continue;
        }

        let status = Command::new(&gdalwarp)
            .args([
                "-te",
                &format!("{min_lng}"),
                &format!("{min_lat}"),
                &format!("{max_lng}"),
                &format!("{max_lat}"),
                "-ts",
                &format!("{width}"),
                &format!("{height}"),
                "-r",
                "bilinear",
                "-of",
                "GTiff",
                "-ot",
                "Float32",
                "-co",
                "COMPRESS=LZW",
                &src_path,
                &out_path,
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                processed.push(out_path);
            }
            Ok(s) => {
                log::warn!(
                    "gdalwarp failed for rainfall {fname} (exit code {}), skipping",
                    s.code().map_or(-1, |c| c)
                );
            }
            Err(e) => {
                log::warn!("failed to spawn gdalwarp for rainfall {fname}: {e}, skipping");
            }
        }
    }

    Ok(processed)
}

/// Resample all monthly temperature NetCDF files to match the CLASS grid.
///
/// Scans `temperature_dir` for `tmp_YYYY.nc` files.  Each NetCDF contains 12
/// monthly bands (Int16, NoData=-32768, unit 0.1°C).  The entire multi‑band
/// file is resampled via `gdalwarp` (bilinear), then individual monthly bands
/// are extracted via `gdal_translate` and written as
/// `output_dir/tmp_YYYY_MM.tif` (Float32).
///
/// * `class_grid`       — reference grid providing target dimensions, bounds, and CRS
/// * `temperature_dir`  — directory containing annual NetCDF files
/// * `output_dir`       — directory for resampled rasters (created if missing)
///
/// Returns the list of output file paths successfully created.  Failed months
/// are logged (via `log::warn`) but do **not** halt the remaining batch.
/// Existing output files are skipped (resume support).  Temporary intermediate
/// files are cleaned up.
pub fn resample_temperature_to_class(
    class_grid: &ClassGrid,
    temperature_dir: &str,
    output_dir: &str,
) -> LoaderResult<Vec<String>> {
    let gdalwarp = find_gdalwarp()
        .ok_or_else(|| LoaderError::Gdal("gdalwarp not found in PATH or GDAL_HOME".into()))?;
    let gdal_translate = find_gdal_translate()
        .ok_or_else(|| LoaderError::Gdal("gdal_translate not found in PATH or GDAL_HOME".into()))?;

    // Discover year files from filesystem.
    let mut years: Vec<i32> = Vec::new();
    let dir_entries = std::fs::read_dir(temperature_dir).map_err(|e| {
        LoaderError::Io(std::io::Error::new(
            e.kind(),
            format!("cannot read temperature_dir {temperature_dir}: {e}"),
        ))
    })?;

    for entry in dir_entries.flatten() {
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        if fname.starts_with("tmp_") && fname.ends_with(".nc") {
            let stem = fname.trim_end_matches(".nc").trim_start_matches("tmp_");
            if let Ok(year) = stem.parse::<i32>() {
                years.push(year);
            }
        }
    }
    years.sort_unstable();

    if years.is_empty() {
        return Err(LoaderError::FileNotFound(format!(
            "no tmp_YYYY.nc files found in {temperature_dir}"
        )));
    }

    std::fs::create_dir_all(output_dir).map_err(LoaderError::Io)?;

    let (min_lng, min_lat, max_lng, max_lat) = class_grid.bounds();
    let width = class_grid.width();
    let height = class_grid.height();

    let mut processed: Vec<String> = Vec::with_capacity(years.len() * 12);

    for year in &years {
        let src_nc = format!("{temperature_dir}/tmp_{year}.nc");

        if !Path::new(&src_nc).exists() {
            log::warn!("temperature source file not found, skipping: {src_nc}");
            continue;
        }

        // Check if all 12 months already exist (full resume).
        let all_exist = (1..=12).all(|m| {
            let out_path = format!("{output_dir}/tmp_{year}_{m:02}.tif");
            Path::new(&out_path).exists()
        });
        if all_exist {
            log::info!("temperature year {year} already fully resampled, skipping");
            for m in 1..=12 {
                processed.push(format!("{output_dir}/tmp_{year}_{m:02}.tif"));
            }
            continue;
        }

        // Step 1: gdalwarp entire NetCDF (12 bands) to a temp multi-band GTiff.
        let temp_resampled = format!("{output_dir}/._tmp_resampled_{year}.tif");
        let status = Command::new(&gdalwarp)
            .args([
                "-te",
                &format!("{min_lng}"),
                &format!("{min_lat}"),
                &format!("{max_lng}"),
                &format!("{max_lat}"),
                "-ts",
                &format!("{width}"),
                &format!("{height}"),
                "-r",
                "bilinear",
                "-of",
                "GTiff",
                "-ot",
                "Float32",
                "-co",
                "COMPRESS=LZW",
                &src_nc,
                &temp_resampled,
            ])
            .status();

        match status {
            Ok(s) if !s.success() => {
                log::warn!(
                    "gdalwarp failed for temperature year {year} (exit code {}), skipping",
                    s.code().map_or(-1, |c| c)
                );
                let _ = std::fs::remove_file(&temp_resampled);
                continue;
            }
            Err(e) => {
                log::warn!("failed to spawn gdalwarp for temperature year {year}: {e}, skipping");
                continue;
            }
            Ok(_) => {} // success
        }

        // Step 2: extract individual monthly bands via gdal_translate.
        for month in 1..=12 {
            let out_path = format!("{output_dir}/tmp_{year}_{month:02}.tif");

            // Skip existing output (partial resume).
            if Path::new(&out_path).exists() {
                log::info!("temperature tmp_{year}_{month:02} already exists, skipping");
                processed.push(out_path);
                continue;
            }

            let status = Command::new(&gdal_translate)
                .args([
                    "-b",
                    &format!("{month}"),
                    "-of",
                    "GTiff",
                    "-ot",
                    "Float32",
                    "-co",
                    "COMPRESS=LZW",
                    &temp_resampled,
                    &out_path,
                ])
                .status();

            match status {
                Ok(s) if s.success() => {
                    processed.push(out_path);
                }
                Ok(s) => {
                    log::warn!(
                        "gdal_translate failed for year {year} month {month} (exit code {}), skipping",
                        s.code().map_or(-1, |c| c)
                    );
                }
                Err(e) => {
                    log::warn!(
                        "failed to spawn gdal_translate for year {year} month {month}: {e}, skipping"
                    );
                }
            }
        }

        // Step 3: clean up temp multi-band file.
        let _ = std::fs::remove_file(&temp_resampled);
    }

    Ok(processed)
}

// ── recursive file discovery ────────────────────────────────────────

/// Collect all files under `dir` whose names end with `suffix` (case-sensitive).
fn collect_files_recursive(dir: &Path, suffix: &str) -> LoaderResult<Vec<String>> {
    let mut files: Vec<String> = Vec::new();
    let mut stack: Vec<std::path::PathBuf> = vec![dir.to_path_buf()];

    while let Some(current) = stack.pop() {
        let entries = std::fs::read_dir(&current).map_err(|e| {
            LoaderError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot read directory {}: {e}", current.display()),
            ))
        })?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if fname.ends_with(suffix) {
                    files.push(path.to_string_lossy().to_string());
                }
            }
        }
    }
    Ok(files)
}

// ── landuse resampling ──────────────────────────────────────────────

/// Resample provincial landuse tiles (Albers Equal Area) to the CLASS reference grid.
///
/// Discovers all `.tif` files recursively under `landuse_dir`, builds a VRT
/// mosaic via `gdalbuildvrt`, then reprojects (`-t_srs EPSG:4326`) and resamples
/// to the CLASS grid dimensions with nearest-neighbour (`-r near`) to preserve
/// categorical landuse values (0-255).  Output is written as a single GeoTIFF
/// (`Byte`, LZW compressed) at `output_path`.
///
/// * `class_grid`  — reference grid providing target dimensions, bounds, and CRS
/// * `landuse_dir` — directory containing provincial Albers GeoTIFFs (recursive)
/// * `output_path` — path for the output resampled raster (created if missing)
///
/// Temporary VRT file is cleaned up regardless of success or failure.
pub fn resample_landuse_to_class(
    class_grid: &ClassGrid,
    landuse_dir: &str,
    output_path: &str,
) -> LoaderResult<()> {
    let gdalbuildvrt = find_gdalbuildvrt()
        .ok_or_else(|| LoaderError::Gdal("gdalbuildvrt not found in PATH or GDAL_HOME".into()))?;
    let gdalwarp = find_gdalwarp()
        .ok_or_else(|| LoaderError::Gdal("gdalwarp not found in PATH or GDAL_HOME".into()))?;

    // Discover all provincial tif files recursively.
    let tif_files = collect_files_recursive(Path::new(landuse_dir), ".tif")?;
    if tif_files.is_empty() {
        return Err(LoaderError::FileNotFound(format!(
            "no .tif files found under {landuse_dir}"
        )));
    }
    log::info!("found {} landuse tiles to mosaic", tif_files.len());

    // Create output directory if needed.
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).map_err(LoaderError::Io)?;
    }

    // Build a temporary file-list and VRT.
    let tmp_vrt = format!("{output_path}.vrt");

    // Write file list for gdalbuildvrt (avoids command-line length limits).
    let list_path = format!("{output_path}.filelist");
    std::fs::write(&list_path, tif_files.join("\n")).map_err(LoaderError::Io)?;

    let status = Command::new(&gdalbuildvrt)
        .args([
            "-input_file_list",
            &list_path,
            "-overwrite",
            &tmp_vrt,
        ])
        .status()
        .map_err(|e| LoaderError::Gdal(format!("failed to spawn gdalbuildvrt: {e}")))?;

    // Clean up file list regardless.
    let _ = std::fs::remove_file(&list_path);

    if !status.success() {
        let _ = std::fs::remove_file(&tmp_vrt);
        return Err(LoaderError::Gdal(format!(
            "gdalbuildvrt exited with status {}",
            status
        )));
    }

    // Warp: reproject Albers → EPSG:4326, resample to CLASS grid, nearest-neighbour.
    let (min_lng, min_lat, max_lng, max_lat) = class_grid.bounds();
    let width = class_grid.width();
    let height = class_grid.height();

    let status = Command::new(&gdalwarp)
        .args([
            "-t_srs",
            "EPSG:4326",
            "-te",
            &format!("{min_lng}"),
            &format!("{min_lat}"),
            &format!("{max_lng}"),
            &format!("{max_lat}"),
            "-ts",
            &format!("{width}"),
            &format!("{height}"),
            "-r",
            "near",
            "-of",
            "GTiff",
            "-ot",
            "Byte",
            "-co",
            "COMPRESS=LZW",
            &tmp_vrt,
            output_path,
        ])
        .status()
        .map_err(|e| {
            let _ = std::fs::remove_file(&tmp_vrt);
            LoaderError::Gdal(format!("failed to spawn gdalwarp: {e}"))
        })?;

    // Clean up temp VRT.
    let _ = std::fs::remove_file(&tmp_vrt);

    if !status.success() {
        return Err(LoaderError::Gdal(format!(
            "gdalwarp exited with status {}",
            status
        )));
    }

    log::info!("landuse resampled → {output_path}");
    Ok(())
}

// ── resident resampling ─────────────────────────────────────────────

/// Resample resident population density raster to the CLASS reference grid.
///
/// Source raster is a single-band GeoTIFF with NoData=127.0 representing
/// uninhabited cells.  Bilinear interpolation is used for continuous
/// population density data.  If the source CRS differs from EPSG:4326,
/// `-t_srs EPSG:4326` reprojects as part of the warp.
///
/// * `class_grid`    — reference grid providing target dimensions, bounds, and CRS
/// * `resident_path` — path to the source resident GeoTIFF
/// * `output_path`   — path for the output resampled raster (parent dirs created if
///                      missing)
pub fn resample_resident_to_class(
    class_grid: &ClassGrid,
    resident_path: &str,
    output_path: &str,
) -> LoaderResult<()> {
    let gdalwarp = find_gdalwarp()
        .ok_or_else(|| LoaderError::Gdal("gdalwarp not found in PATH or GDAL_HOME".into()))?;

    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| LoaderError::Io(e))?;
    }

    let (min_lng, min_lat, max_lng, max_lat) = class_grid.bounds();
    let width = class_grid.width();
    let height = class_grid.height();

    let status = Command::new(&gdalwarp)
        .args([
            "-t_srs",
            "EPSG:4326",
            "-te",
            &format!("{min_lng}"),
            &format!("{min_lat}"),
            &format!("{max_lng}"),
            &format!("{max_lat}"),
            "-ts",
            &format!("{width}"),
            &format!("{height}"),
            "-r",
            "bilinear",
            "-srcnodata",
            "127",
            "-dstnodata",
            "nan",
            "-of",
            "GTiff",
            "-ot",
            "Float32",
            "-co",
            "COMPRESS=LZW",
            resident_path,
            output_path,
        ])
        .status()
        .map_err(|e| LoaderError::Gdal(format!("failed to spawn gdalwarp: {e}")))?;

    if !status.success() {
        return Err(LoaderError::Gdal(format!(
            "gdalwarp exited with status {}",
            status
        )));
    }

    log::info!("resident resampled → {output_path}");
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "gdal")]
mod tests {
    use super::*;
    use gdal::raster::Buffer;
    use gdal::DriverManager;
    use tempfile::TempDir;

    fn gdalwarp_available() -> bool {
        find_gdalwarp().is_some()
    }

    fn create_synth_raster(
        path: &str,
        width: usize,
        height: usize,
        data: &[f32],
        nodata: Option<f64>,
        geo_transform: &[f64; 6],
        crs: &str,
    ) {
        let drv = DriverManager::get_driver_by_name("GTiff").unwrap();
        let mut ds = drv
            .create_with_band_type::<f32, _>(path, width, height, 1)
            .unwrap();
        ds.set_geo_transform(geo_transform).unwrap();
        ds.set_projection(crs).unwrap();
        let mut band = ds.rasterband(1).unwrap();
        band.set_no_data_value(nodata).unwrap();
        let mut buf = Buffer::new((width, height), data.to_vec());
        band.write((0, 0), (width, height), &mut buf).unwrap();
    }

    fn create_synth_class_grid(
        path: &str,
        width: usize,
        height: usize,
        bands: usize,
        geo_transform: &[f64; 6],
        crs: &str,
    ) -> ClassGrid {
        let drv = DriverManager::get_driver_by_name("GTiff").unwrap();
        let mut ds = drv
            .create_with_band_type::<i8, _>(path, width, height, bands)
            .unwrap();
        ds.set_geo_transform(geo_transform).unwrap();
        ds.set_projection(crs).unwrap();
        drop(ds);
        ClassGrid::open(path, 2016).unwrap()
    }

    const WGS84_WKT: &str = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]";

    #[test]
    fn test_resample_dem_to_class_basic() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let src_path = tmp.path().join("src_dem.tif");
        let class_path = tmp.path().join("class.tif");
        let out_path = tmp.path().join("resampled_dem.tif");

        let mut src_data = vec![1500.0f32; 100];
        src_data[0] = -9999.0;
        src_data[1] = -9999.0;
        src_data[99] = -9999.0;

        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        create_synth_raster(
            src_path.to_str().unwrap(),
            10,
            10,
            &src_data,
            Some(-9999.0),
            &src_gt,
            WGS84_WKT,
        );

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            5,
            5,
            2,
            &[88.02, 0.01, 0.0, 27.98, 0.0, -0.01],
            WGS84_WKT,
        );

        resample_dem_to_class(
            &cg,
            src_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        )
        .unwrap();

        assert!(out_path.exists(), "output file was not created");

        let ds = gdal::Dataset::open(out_path.to_str().unwrap()).unwrap();
        let (w, h) = ds.raster_size();
        assert_eq!(w as usize, 5, "width mismatch: expected 5, got {w}");
        assert_eq!(h as usize, 5, "height mismatch: expected 5, got {h}");

        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<f32>((0, 0), (5, 5), (5, 5), None)
            .unwrap();
        let arr: Vec<f32> = out_data.data().to_vec();
        let valid_count = arr.iter().filter(|v| v.is_finite()).count();

        assert!(
            valid_count > 0,
            "expected valid elevation values, got all NaN"
        );

        let out_crs = ds.projection();
        assert!(
            out_crs.contains("4326") || out_crs.contains("WGS 84"),
            "output CRS should be EPSG:4326, got: {out_crs}"
        );
    }

    #[test]
    fn test_resample_dem_to_class_all_nodata() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let src_path = tmp.path().join("src_all_nodata.tif");
        let class_path = tmp.path().join("class_nodata.tif");
        let out_path = tmp.path().join("resampled_all_nodata.tif");

        let src_data = vec![-9999.0f32; 25];
        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        create_synth_raster(
            src_path.to_str().unwrap(),
            5,
            5,
            &src_data,
            Some(-9999.0),
            &src_gt,
            WGS84_WKT,
        );

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            3,
            3,
            2,
            &[88.01, 0.01, 0.0, 27.99, 0.0, -0.01],
            WGS84_WKT,
        );

        resample_dem_to_class(
            &cg,
            src_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        )
        .unwrap();

        let ds = gdal::Dataset::open(out_path.to_str().unwrap()).unwrap();
        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<f32>((0, 0), (3, 3), (3, 3), None)
            .unwrap();
        let all_nan = out_data.data().iter().all(|v| v.is_nan());

        assert!(all_nan, "expected all NaN values for all-NoData source");
    }

    #[test]
    fn test_resample_dem_to_class_bounds() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let src_path = tmp.path().join("src_bounds.tif");
        let class_path = tmp.path().join("class_bounds.tif");
        let out_path = tmp.path().join("resampled_bounds.tif");

        let mut src_data = vec![0.0f32; 400];
        for r in 0..20 {
            for c in 0..20 {
                src_data[r * 20 + c] = if (r + c) % 2 == 0 { 500.0 } else { 1000.0 };
            }
        }

        let src_gt = [100.0, 0.01, 0.0, 30.2, 0.0, -0.01];
        create_synth_raster(
            src_path.to_str().unwrap(),
            20,
            20,
            &src_data,
            Some(-9999.0),
            &src_gt,
            WGS84_WKT,
        );

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            10,
            10,
            2,
            &[100.0, 0.02, 0.0, 30.2, 0.0, -0.02],
            WGS84_WKT,
        );

        resample_dem_to_class(
            &cg,
            src_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        )
        .unwrap();

        let ds = gdal::Dataset::open(out_path.to_str().unwrap()).unwrap();
        let (w, h) = ds.raster_size();
        assert_eq!(w as usize, 10, "width mismatch: expected 10, got {w}");
        assert_eq!(h as usize, 10, "height mismatch: expected 10, got {h}");

        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<f32>((0, 0), (10, 10), (10, 10), None)
            .unwrap();
        let min = out_data
            .data()
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max = out_data
            .data()
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(min > 450.0, "min elevation too low: {min}");
        assert!(max < 1050.0, "max elevation too high: {max}");
    }

    #[test]
    fn test_resample_creates_parent_dir() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let src_path = tmp.path().join("src.tif");
        let class_path = tmp.path().join("class_nested.tif");
        let out_path = tmp.path().join("nested").join("subdir").join("dem.tif");

        let src_data = vec![500.0f32; 25];
        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        create_synth_raster(
            src_path.to_str().unwrap(),
            5,
            5,
            &src_data,
            Some(-9999.0),
            &src_gt,
            WGS84_WKT,
        );

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            2,
            2,
            2,
            &[88.01, 0.01, 0.0, 27.99, 0.0, -0.01],
            WGS84_WKT,
        );

        resample_dem_to_class(
            &cg,
            src_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        )
        .unwrap();

        assert!(out_path.exists(), "nested output path not created");
    }

    // ── CCDC resample tests ───────────────────────────────────────────

    #[test]
    fn test_resample_ccdc_to_class_basic() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let ccdc_dir = tmp.path().join("ccdc");
        std::fs::create_dir_all(&ccdc_dir).unwrap();
        let output_dir = tmp.path().join("resampled_ccdc");
        let class_path = tmp.path().join("class_ccdc_test.tif");

        // 3×3 CCDC source with known disturbance year values.
        let src_data = vec![2020.0f32, 2021.0, 2022.0, 2018.0, 2019.0, 2020.0, 2021.0, 2022.0, 2023.0];
        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        let src_path = ccdc_dir.join("2020.tif");
        create_synth_raster(
            src_path.to_str().unwrap(),
            3,
            3,
            &src_data,
            None,
            &src_gt,
            WGS84_WKT,
        );

        // 2×2 CLASS grid covering same extent.
        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            2,
            2,
            2,
            &src_gt,
            WGS84_WKT,
        );

        let processed = resample_ccdc_to_class(
            &cg,
            ccdc_dir.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        )
        .unwrap();

        assert_eq!(processed, vec![2020], "expected year 2020 processed");

        let out_tif = output_dir.join("2020.tif");
        assert!(out_tif.exists(), "output tif not created");

        let ds = gdal::Dataset::open(out_tif.to_str().unwrap()).unwrap();
        let (w, h) = ds.raster_size();
        assert_eq!(w as usize, 2, "width mismatch");
        assert_eq!(h as usize, 2, "height mismatch");

        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<f32>((0, 0), (2, 2), (2, 2), None)
            .unwrap();
        let vals: Vec<f32> = out_data.data().to_vec();

        // Nearest-neighbour must preserve exact integer disturbance years.
        for v in &vals {
            assert!(
                v.fract() == 0.0,
                "value {v} has fractional part — nearest-neighbour violated"
            );
        }
    }

    #[test]
    fn test_resample_ccdc_multi_year() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let ccdc_dir = tmp.path().join("ccdc_multi");
        std::fs::create_dir_all(&ccdc_dir).unwrap();
        let output_dir = tmp.path().join("resampled_ccdc_multi");
        let class_path = tmp.path().join("class_ccdc_multi.tif");

        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        for year in 2018..=2021 {
            let src_data = vec![year as f32; 9];
            let src_path = ccdc_dir.join(format!("{year}.tif"));
            create_synth_raster(src_path.to_str().unwrap(), 3, 3, &src_data, None, &src_gt, WGS84_WKT);
        }

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            2,
            2,
            2,
            &src_gt,
            WGS84_WKT,
        );

        let processed = resample_ccdc_to_class(
            &cg,
            ccdc_dir.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        )
        .unwrap();

        assert_eq!(processed.len(), 4, "expected 4 years processed");

        for year in 2018..=2021 {
            let out_tif = output_dir.join(format!("{year}.tif"));
            assert!(out_tif.exists(), "output for year {year} missing");
            let ds = gdal::Dataset::open(out_tif.to_str().unwrap()).unwrap();
            let (w, h) = ds.raster_size();
            assert_eq!(w as usize, 2);
            assert_eq!(h as usize, 2);
        }
    }

    #[test]
    fn test_resample_ccdc_value_preservation() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let ccdc_dir = tmp.path().join("ccdc_preserve");
        std::fs::create_dir_all(&ccdc_dir).unwrap();
        let output_dir = tmp.path().join("resampled_ccdc_preserve");
        let class_path = tmp.path().join("class_ccdc_preserve.tif");

        // 4×4 source with distinct values — ensure NN picks exact values.
        let mut src_data = vec![0.0f32; 16];
        for i in 0..16 {
            src_data[i] = 2000.0 + i as f32;
        }
        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        let src_path = ccdc_dir.join("2015.tif");
        create_synth_raster(src_path.to_str().unwrap(), 4, 4, &src_data, None, &src_gt, WGS84_WKT);

        // 2×2 CLASS grid covering same extent — output pixels inherit exact source values.
        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            2,
            2,
            2,
            &src_gt,
            WGS84_WKT,
        );

        resample_ccdc_to_class(&cg, ccdc_dir.to_str().unwrap(), output_dir.to_str().unwrap())
            .unwrap();

        let ds = gdal::Dataset::open(output_dir.join("2015.tif").to_str().unwrap()).unwrap();
        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<f32>((0, 0), (2, 2), (2, 2), None)
            .unwrap();
        let vals: Vec<f32> = out_data.data().to_vec();

        // Every output pixel value must exist in the source array.
        for v in &vals {
            assert!(
                src_data.contains(v),
                "output value {v} not present in source — nearest-neighbour violated"
            );
        }
    }

    #[test]
    fn test_resample_ccdc_empty_dir() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let ccdc_dir = tmp.path().join("ccdc_empty");
        std::fs::create_dir_all(&ccdc_dir).unwrap();
        let output_dir = tmp.path().join("resampled_ccdc_empty");
        let class_path = tmp.path().join("class_ccdc_empty.tif");

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            2,
            2,
            2,
            &[88.0, 0.01, 0.0, 28.0, 0.0, -0.01],
            WGS84_WKT,
        );

        let result = resample_ccdc_to_class(
            &cg,
            ccdc_dir.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        );
        assert!(result.is_err(), "empty dir should return error");
    }

    // ── helpers for climate tests ─────────────────────────────────────

    fn create_multi_band_raster(
        path: &str,
        width: usize,
        height: usize,
        bands: usize,
        data: &[f32],
        nodata: Option<f64>,
        geo_transform: &[f64; 6],
        crs: &str,
    ) {
        let drv = DriverManager::get_driver_by_name("GTiff").unwrap();
        let mut ds = drv
            .create_with_band_type::<f32, _>(path, width, height, bands)
            .unwrap();
        ds.set_geo_transform(geo_transform).unwrap();
        ds.set_projection(crs).unwrap();
        for b in 0..bands {
            let mut band = ds.rasterband(b + 1).unwrap();
            band.set_no_data_value(nodata).unwrap();
            let offset = b * width * height;
            let slice = &data[offset..offset + width * height];
            let mut buf = Buffer::new((width, height), slice.to_vec());
            band.write((0, 0), (width, height), &mut buf).unwrap();
        }
    }

    fn gdal_translate_available() -> bool {
        find_gdal_translate().is_some()
    }

    /// Create a NetCDF with a `tmp` variable from a multi-band GeoTIFF.
    /// Uses Python xarray (available in the geo-science conda env) to produce
    /// a file with direct raster bands that gdalwarp can read.
    fn tif_to_nc_via_xarray(src_tif: &str, dst_nc: &str, bands: usize) -> bool {
        let script = format!(
            r#"
import numpy as np, xarray as xr, rasterio
with rasterio.open('{src_tif}') as src:
    data = src.read()  # shape (bands, h, w)
    h, w = src.height, src.width
    gt = src.transform
    lon = np.linspace(gt.c, gt.c + gt.a * w, w)
    lat = np.linspace(gt.f, gt.f + gt.e * h, h)

ds = xr.Dataset(
    {{'tmp': (('time', 'lat', 'lon'), data.astype(np.float32))}},
    coords={{'time': list(range(1, {bands}+1)), 'lat': lat, 'lon': lon}}
)
ds['tmp'].attrs['missing_value'] = -32768
ds['tmp'].attrs['long_name'] = 'monthly mean temperature'
ds['tmp'].attrs['units'] = '0.1 degree centigrade'
ds.to_netcdf('{dst_nc}')
"#
        );
        Command::new("python3")
            .arg("-c")
            .arg(&script)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    // ── rainfall resample tests ───────────────────────────────────────

    #[test]
    fn test_rainfall_resample() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let rainfall_dir = tmp.path().join("rainfall");
        std::fs::create_dir_all(&rainfall_dir).unwrap();
        let output_dir = tmp.path().join("resampled_rainfall");
        let class_path = tmp.path().join("class_rainfall_test.tif");

        // 4×4 rainfall source with known values (e.g. mm/month).
        let src_data: Vec<f32> = (0..16).map(|i| 10.0 + i as f32 * 5.0).collect();
        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        create_synth_raster(
            rainfall_dir.join("pre202001.tif").to_str().unwrap(),
            4,
            4,
            &src_data,
            None,
            &src_gt,
            WGS84_WKT,
        );

        // 8×8 CLASS grid covering same extent (higher resolution).
        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            8,
            8,
            2,
            &src_gt,
            WGS84_WKT,
        );

        let processed = resample_rainfall_to_class(
            &cg,
            rainfall_dir.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        )
        .unwrap();

        assert_eq!(processed.len(), 1, "expected 1 file processed");

        let out_tif = output_dir.join("pre202001.tif");
        assert!(out_tif.exists(), "output tif not created");

        let ds = gdal::Dataset::open(out_tif.to_str().unwrap()).unwrap();
        let (w, h) = ds.raster_size();
        assert_eq!(w as usize, 8, "width mismatch");
        assert_eq!(h as usize, 8, "height mismatch");
        assert_eq!(ds.raster_count(), 1, "should be single-band");

        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<f32>((0, 0), (8, 8), (8, 8), None)
            .unwrap();
        let vals: Vec<f32> = out_data.data().to_vec();
        let valid = vals.iter().filter(|v| v.is_finite()).count();
        assert!(valid > 0, "expected valid rainfall values, got all NaN");
    }

    #[test]
    fn test_rainfall_resample_multi_month() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let rainfall_dir = tmp.path().join("rainfall_multi");
        std::fs::create_dir_all(&rainfall_dir).unwrap();
        let output_dir = tmp.path().join("resampled_rainfall_multi");
        let class_path = tmp.path().join("class_rainfall_multi.tif");

        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        let src_data = vec![50.0f32; 9];
        for (year, month) in [(2020, 1), (2020, 2), (2020, 3)] {
            let fname = format!("pre{year:04}{month:02}.tif");
            create_synth_raster(
                rainfall_dir.join(&fname).to_str().unwrap(),
                3, 3,
                &src_data,
                None,
                &src_gt,
                WGS84_WKT,
            );
        }

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(), 2, 2, 2,
            &src_gt, WGS84_WKT,
        );

        let processed = resample_rainfall_to_class(
            &cg,
            rainfall_dir.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        )
        .unwrap();

        assert_eq!(processed.len(), 3, "expected 3 months processed");
        for (year, month) in [(2020, 1), (2020, 2), (2020, 3)] {
            let out_path = output_dir.join(format!("pre{year:04}{month:02}.tif"));
            assert!(out_path.exists(), "output for {year}-{month:02} missing");
        }
    }

    #[test]
    fn test_rainfall_resample_empty_dir() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let rainfall_dir = tmp.path().join("rainfall_empty");
        std::fs::create_dir_all(&rainfall_dir).unwrap();
        let output_dir = tmp.path().join("resampled_rainfall_empty");
        let class_path = tmp.path().join("class_rainfall_empty.tif");

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(), 2, 2, 2,
            &[88.0, 0.01, 0.0, 28.0, 0.0, -0.01], WGS84_WKT,
        );

        let result = resample_rainfall_to_class(
            &cg,
            rainfall_dir.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        );
        assert!(result.is_err(), "empty dir should return error");
    }

    // ── temperature resample tests ────────────────────────────────────

    #[test]
    fn test_temperature_resample() {
        if !gdalwarp_available() || !gdal_translate_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let temperature_dir = tmp.path().join("temperature");
        std::fs::create_dir_all(&temperature_dir).unwrap();
        let output_dir = tmp.path().join("resampled_temperature");
        let class_path = tmp.path().join("class_temperature_test.tif");

        // Create a 3-band source GeoTIFF (simulating temperature months).
        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        let mut src_data = Vec::with_capacity(4 * 4 * 3);
        for band in 0..3 {
            for i in 0..16 {
                src_data.push(150.0 + (band * 50 + i) as f32);
            }
        }
        let src_tif = temperature_dir.join("_src_multi.tif");
        create_multi_band_raster(
            src_tif.to_str().unwrap(), 4, 4, 3, &src_data,
            Some(-32768.0), &src_gt, WGS84_WKT,
        );

        // Convert to NetCDF with xarray so gdalwarp can read bands directly.
        let nc_path = temperature_dir.join("tmp_2020.nc");
        if !tif_to_nc_via_xarray(src_tif.to_str().unwrap(), nc_path.to_str().unwrap(), 3) {
            eprintln!("xarray NetCDF creation not available, skipping temperature test");
            return;
        }

        // 6×6 CLASS grid.
        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(), 6, 6, 2,
            &src_gt, WGS84_WKT,
        );

        let processed = resample_temperature_to_class(
            &cg,
            temperature_dir.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        )
        .unwrap();

        assert_eq!(processed.len(), 3, "expected 3 months processed");

        for m in 1..=3 {
            let out_path = output_dir.join(format!("tmp_2020_{m:02}.tif"));
            assert!(out_path.exists(), "output for month {m} missing");

            let ds = gdal::Dataset::open(out_path.to_str().unwrap()).unwrap();
            let (w, h) = ds.raster_size();
            assert_eq!(w as usize, 6, "width mismatch month {m}");
            assert_eq!(h as usize, 6, "height mismatch month {m}");
            assert_eq!(ds.raster_count(), 1, "month {m} should be single-band");

            let band = ds.rasterband(1).unwrap();
            let out_data = band
                .read_as::<f32>((0, 0), (6, 6), (6, 6), None)
                .unwrap();
            let valid = out_data.data().iter().filter(|v| v.is_finite()).count();
            assert!(valid > 0, "month {m} has no valid data");
        }
    }

    #[test]
    fn test_temperature_resample_empty_dir() {
        if !gdalwarp_available() || !gdal_translate_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let temperature_dir = tmp.path().join("temperature_empty");
        std::fs::create_dir_all(&temperature_dir).unwrap();
        let output_dir = tmp.path().join("resampled_temperature_empty");
        let class_path = tmp.path().join("class_temperature_empty.tif");

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(), 2, 2, 2,
            &[88.0, 0.01, 0.0, 28.0, 0.0, -0.01], WGS84_WKT,
        );

        let result = resample_temperature_to_class(
            &cg,
            temperature_dir.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        );
        assert!(result.is_err(), "empty dir should return error");
    }

    // ── resident resample tests ─────────────────────────────────────

    #[test]
    fn test_resident_resample() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let src_path = tmp.path().join("src_resident.tif");
        let class_path = tmp.path().join("class_resident.tif");
        let out_path = tmp.path().join("resampled_resident.tif");

        // Source raster with NoData=127.0 (per AGENTS.md resident convention).
        let mut src_data = vec![50.0f32; 100];
        src_data[0] = 127.0; // NoData cell
        src_data[1] = 127.0;
        src_data[99] = 127.0;

        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        create_synth_raster(
            src_path.to_str().unwrap(),
            10,
            10,
            &src_data,
            Some(127.0),
            &src_gt,
            WGS84_WKT,
        );

        // CLASS grid with different extent to test cropping + resampling.
        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            5,
            5,
            2,
            &[88.02, 0.01, 0.0, 27.98, 0.0, -0.01],
            WGS84_WKT,
        );

        resample_resident_to_class(
            &cg,
            src_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        )
        .unwrap();

        assert!(out_path.exists(), "output file was not created");

        let ds = gdal::Dataset::open(out_path.to_str().unwrap()).unwrap();
        let (w, h) = ds.raster_size();
        assert_eq!(w as usize, 5, "width mismatch");
        assert_eq!(h as usize, 5, "height mismatch");

        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<f32>((0, 0), (5, 5), (5, 5), None)
            .unwrap();
        let arr: Vec<f32> = out_data.data().to_vec();
        let valid_count = arr.iter().filter(|v| v.is_finite()).count();

        assert!(
            valid_count > 0,
            "expected valid population values, got all NaN"
        );

        let out_crs = ds.projection();
        assert!(
            out_crs.contains("4326") || out_crs.contains("WGS 84"),
            "output CRS should be EPSG:4326, got: {out_crs}"
        );
    }

    #[test]
    fn test_resident_resample_all_nodata() {
        if !gdalwarp_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let src_path = tmp.path().join("src_resident_all_nodata.tif");
        let class_path = tmp.path().join("class_resident_nodata.tif");
        let out_path = tmp.path().join("resampled_resident_all_nodata.tif");

        let src_data = vec![127.0f32; 25];
        let src_gt = [88.0, 0.01, 0.0, 28.0, 0.0, -0.01];
        create_synth_raster(
            src_path.to_str().unwrap(),
            5,
            5,
            &src_data,
            Some(127.0),
            &src_gt,
            WGS84_WKT,
        );

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            3,
            3,
            2,
            &[88.01, 0.01, 0.0, 27.99, 0.0, -0.01],
            WGS84_WKT,
        );

        resample_resident_to_class(
            &cg,
            src_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        )
        .unwrap();

        let ds = gdal::Dataset::open(out_path.to_str().unwrap()).unwrap();
        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<f32>((0, 0), (3, 3), (3, 3), None)
            .unwrap();
        let all_nan = out_data.data().iter().all(|v| v.is_nan());

        assert!(all_nan, "expected all NaN values for all-NoData source");
    }

    // ── landuse resample tests ──────────────────────────────────────

    fn gdalbuildvrt_available() -> bool {
        find_gdalbuildvrt().is_some()
    }

    #[test]
    fn test_resample_landuse_basic() {
        if !gdalwarp_available() || !gdalbuildvrt_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let landuse_dir = tmp.path().join("landuse_tiles");
        let sub_dir = landuse_dir.join("province_a");
        std::fs::create_dir_all(&sub_dir).unwrap();

        let class_path = tmp.path().join("class_landuse.tif");
        let out_path = tmp.path().join("resampled_landuse.tif");

        // Create two fake provincial tiles with known classification values.
        // Source: 4×4 pixels, EPSG:4326, covering left and right halves.
        let src_gt = [100.0, 0.01, 0.0, 30.0, 0.0, -0.01];

        // Tile A: left side, class values 1, 2, 3
        let mut tile_a = vec![0.0f32; 16];
        for (i, v) in tile_a.iter_mut().enumerate() {
            *v = ((i % 3) + 1) as f32; // 1,2,3,1,2,3,...
        }
        create_synth_raster(
            sub_dir.join("tile_a.tif").to_str().unwrap(),
            4, 4, &tile_a, None, &src_gt, WGS84_WKT,
        );

        // Tile B: right side, class values 7, 8
        let mut tile_b = vec![0.0f32; 16];
        for (i, v) in tile_b.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 7.0 } else { 8.0 };
        }
        create_synth_raster(
            sub_dir.join("tile_b.tif").to_str().unwrap(),
            4, 4, &tile_b, None, &src_gt, WGS84_WKT,
        );

        // CLASS grid: 8×8 covering both tiles.
        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(),
            8, 8, 2,
            &src_gt,
            WGS84_WKT,
        );

        resample_landuse_to_class(
            &cg,
            landuse_dir.to_str().unwrap(),
            out_path.to_str().unwrap(),
        )
        .unwrap();

        assert!(out_path.exists(), "output file was not created");

        let ds = gdal::Dataset::open(out_path.to_str().unwrap()).unwrap();
        let (w, h) = ds.raster_size();
        assert_eq!(w as usize, 8, "width mismatch");
        assert_eq!(h as usize, 8, "height mismatch");

        // Verify nearest-neighbor preserved integer-like values.
        let band = ds.rasterband(1).unwrap();
        let out_data = band
            .read_as::<u8>((0, 0), (8, 8), (8, 8), None)
            .unwrap();
        let vals: Vec<u8> = out_data.data().to_vec();
        let valid_count = vals.iter().filter(|v| **v > 0).count();
        assert!(valid_count > 0, "expected valid landuse class values");

        // All values should be in expected class range (1-8 from our tiles).
        for &v in &vals {
            if v > 0 {
                assert!(
                    v <= 8,
                    "landuse class value {v} out of expected range"
                );
            }
        }
    }

    #[test]
    fn test_resample_landuse_empty_dir() {
        if !gdalwarp_available() || !gdalbuildvrt_available() {
            return;
        }

        let tmp = TempDir::new().unwrap();
        let landuse_dir = tmp.path().join("landuse_empty");
        std::fs::create_dir_all(&landuse_dir).unwrap();
        let class_path = tmp.path().join("class_landuse_empty.tif");
        let out_path = tmp.path().join("resampled_landuse_empty.tif");

        let cg = create_synth_class_grid(
            class_path.to_str().unwrap(), 2, 2, 2,
            &[88.0, 0.01, 0.0, 28.0, 0.0, -0.01], WGS84_WKT,
        );

        let result = resample_landuse_to_class(
            &cg,
            landuse_dir.to_str().unwrap(),
            out_path.to_str().unwrap(),
        );
        assert!(result.is_err(), "empty dir should return error");
    }
}

//! CLASS-centric data preprocessing plan.
//!
//! This crate owns the data_process stage boundary.  It must operate in
//! geospatial coordinates by using [`gdal::GeoGrid`] metadata as the common
//! reference for resampling, clipping, and tiling.

pub mod aoi_mask;
pub mod class_grid;
pub mod feature_extraction;
pub mod logging;
pub mod raster_distance;
pub mod resample;
pub mod river_distance;
pub mod road_distance;
pub mod training_points;

use gdal::{GeoGrid, GridWindow};
use loaders::base::{LoaderError, LoaderResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataProcessSources {
    pub aoi_shapefile: PathBuf,
    pub dem_raster: PathBuf,
    pub ccdc_dir: PathBuf,
    pub rainfall_dir: PathBuf,
    pub temperature_dir: PathBuf,
    pub landuse_dir: PathBuf,
    pub resident_raster: PathBuf,
    pub river_shapefile: PathBuf,
    pub road_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataProcessRequest {
    pub class_raster: PathBuf,
    pub class_start_year: i32,
    pub sources: DataProcessSources,
    pub source_usage: DataSourceUsage,
    pub cache_dir: PathBuf,
    pub tile_size: usize,
    pub processing_years: Vec<i32>,
    pub training_years: Vec<i32>,
    pub prediction_year: i32,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataSourceUsage {
    pub dem: bool,
    pub ccdc: bool,
    pub rainfall: bool,
    pub temperature: bool,
    pub landuse: bool,
    pub resident: bool,
    pub river: bool,
    pub road: bool,
}

impl Default for DataSourceUsage {
    fn default() -> Self {
        Self {
            dem: true,
            ccdc: true,
            rainfall: true,
            temperature: true,
            landuse: true,
            resident: true,
            river: true,
            road: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataProcessPlan {
    pub class_grid: GeoGrid,
    pub processing_years: Vec<i32>,
    pub training_years: Vec<i32>,
    pub prediction_year: i32,
    pub tile_size: usize,
    pub workers: usize,
    pub tiles: Vec<GridWindow>,
}

impl DataProcessPlan {
    pub fn from_request(request: &DataProcessRequest, class_grid: GeoGrid) -> Self {
        let tile_size = request.tile_size;
        let tiles = class_grid.tile_windows(tile_size);
        Self {
            class_grid,
            processing_years: request.processing_years.clone(),
            training_years: request.training_years.clone(),
            prediction_year: request.prediction_year,
            tile_size,
            workers: request.workers,
            tiles,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataProcessRunReport {
    pub class_width: usize,
    pub class_height: usize,
    pub processing_years: Vec<i32>,
    pub training_years: Vec<i32>,
    pub prediction_year: i32,
    pub prediction_tiles: usize,
    pub cache_dir: PathBuf,
}

/// Run the CLASS-centric data_process stage.
///
/// This is the production replacement for the temporary full-basin driver:
/// all source datasets are aligned to the CLASS grid, training samples and
/// features are generated for explicit training years, and full-grid
/// prediction feature tiles are generated for the configured prediction year.
pub fn run_class_basin_data_process(
    request: &DataProcessRequest,
) -> LoaderResult<DataProcessRunReport> {
    crate::status!(
        "[data-process] start: cache={}, workers={}, tile_size={}, prediction_year={}",
        request.cache_dir.display(),
        request.workers,
        request.tile_size,
        request.prediction_year
    );
    let class_path = path_to_string(&request.class_raster);
    let class_grid = class_grid::ClassGrid::open(&class_path, request.class_start_year)?;
    let aoi_mask = aoi_mask::AoiMask::load(&request.sources.aoi_shapefile, &class_grid)?;
    let cache_dir = &request.cache_dir;
    let resampled_dir = cache_dir.join("resampled");
    let features_dir = cache_dir.join("features");

    for dir in [
        resampled_dir.join("ccdc"),
        resampled_dir.join("rainfall"),
        resampled_dir.join("temperature"),
        resampled_dir.join("landuse"),
        resampled_dir.join("road"),
        resampled_dir.join("river"),
        resampled_dir.join("resident"),
        resampled_dir.join("settlement"),
        features_dir.clone(),
    ] {
        std::fs::create_dir_all(dir)?;
    }

    let dem_out = resampled_dir.join("dem.tif");
    if request.source_usage.dem && !dem_out.exists() {
        crate::status!("[data-process] DEM resample -> {}", dem_out.display());
        resample::resample_dem_to_class(
            &class_grid,
            &path_to_string(&request.sources.dem_raster),
            &path_to_string(&dem_out),
        )?;
    } else if request.source_usage.dem {
        crate::status!("[data-process] DEM exists, skip: {}", dem_out.display());
    }

    let ccdc_out = resampled_dir.join("ccdc");
    if request.source_usage.ccdc {
        crate::status!(
            "[data-process] CCDC resample/check -> {}",
            ccdc_out.display()
        );
        let _ = resample::resample_ccdc_to_class_with_workers(
            &class_grid,
            &path_to_string(&request.sources.ccdc_dir),
            &path_to_string(&ccdc_out),
            request.workers,
        )?;
    }

    let processing_years = request.processing_years.clone();
    let year_range = min_max_year(&processing_years);

    let rainfall_out = resampled_dir.join("rainfall");
    if request.source_usage.rainfall {
        crate::status!(
            "[data-process] rainfall resample/check years {:?} -> {}",
            year_range,
            rainfall_out.display()
        );
        let _ = resample::resample_rainfall_to_class_for_years_with_workers(
            &class_grid,
            &path_to_string(&request.sources.rainfall_dir),
            &path_to_string(&rainfall_out),
            year_range,
            request.workers,
            Some(&path_to_string(&request.sources.aoi_shapefile)),
        )?;
    }

    let temperature_out = resampled_dir.join("temperature");
    if request.source_usage.temperature {
        crate::status!(
            "[data-process] temperature resample/check years {:?} -> {}",
            year_range,
            temperature_out.display()
        );
        let _ = resample::resample_temperature_to_class_for_years_with_workers(
            &class_grid,
            &path_to_string(&request.sources.temperature_dir),
            &path_to_string(&temperature_out),
            year_range,
            request.workers,
            Some(&path_to_string(&request.sources.aoi_shapefile)),
        )?;
    }

    if request.source_usage.landuse {
        let available_landuse_years = discover_landuse_years(&request.sources.landuse_dir)?;
        crate::status!(
            "[data-process] landuse available years: {:?}",
            available_landuse_years
        );
        let max_processing_year = processing_years.iter().copied().max().unwrap_or(i32::MAX);
        for &year in &available_landuse_years {
            if year > max_processing_year {
                continue;
            }
            let landuse_out = resampled_dir.join(format!("landuse/landuse_{}.tif", year));
            if landuse_out.exists() {
                crate::status!(
                    "[data-process] landuse {} exists, skip: {}",
                    year,
                    landuse_out.display()
                );
                continue;
            }
            crate::status!(
                "[data-process] landuse {} resample -> {}",
                year,
                landuse_out.display()
            );
            resample::resample_landuse_to_class_for_year(
                &class_grid,
                &path_to_string(&request.sources.landuse_dir),
                &path_to_string(&landuse_out),
                year,
            )?;
        }
    }

    if request.source_usage.resident {
        let resident_dir = request
            .sources
            .resident_raster
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let available_resident = discover_resident_rasters(resident_dir)?;
        crate::status!(
            "[data-process] resident available years: {:?}",
            available_resident
                .iter()
                .map(|(year, _)| *year)
                .collect::<Vec<_>>()
        );
        let max_processing_year = processing_years.iter().copied().max().unwrap_or(i32::MAX);
        for (year, src_path) in &available_resident {
            let year = *year;
            if year > max_processing_year {
                continue;
            }
            let resident_out = resampled_dir.join(format!("resident/resident_{}.tif", year));
            if resident_out.exists() {
                crate::status!(
                    "[data-process] resident {} exists, skip: {}",
                    year,
                    resident_out.display()
                );
                continue;
            }
            crate::status!(
                "[data-process] resident {} resample -> {}",
                year,
                resident_out.display()
            );
            resample::resample_resident_to_class(
                &class_grid,
                &path_to_string(src_path),
                &path_to_string(&resident_out),
            )?;
        }
        for (year, src_path) in &available_resident {
            let year = *year;
            if year > max_processing_year {
                continue;
            }
            for (target_value, prefix, label) in [
                (1u8, "urban_settlement_dist", "urban_settlement"),
                (2u8, "rural_settlement_dist", "rural_settlement"),
            ] {
                let out = resampled_dir.join(format!("settlement/{}_{}.tif", prefix, year));
                if out.exists() {
                    crate::status!(
                        "[data-process] {} {} exists, skip: {}",
                        label,
                        year,
                        out.display()
                    );
                    continue;
                }
                crate::status!(
                    "[data-process] {} {} distance -> {}",
                    label,
                    year,
                    out.display()
                );
                raster_distance::compute_raster_value_distance_raster(
                    &class_grid,
                    &aoi_mask,
                    &request.sources.aoi_shapefile,
                    src_path,
                    target_value,
                    &out,
                    &resampled_dir.join("settlement/_distance_work"),
                    &format!("{}_{}", label, year),
                )?;
            }
        }
    }

    let river_out = resampled_dir.join("river/river_dist.tif");
    if request.source_usage.river && !river_out.exists() {
        crate::status!(
            "[data-process] river distance AOI-masked precompute -> {}",
            river_out.display()
        );
        river_distance::compute_river_distance_raster_rasterized(
            &class_grid,
            &aoi_mask,
            &request.sources.aoi_shapefile,
            &path_to_string(&request.sources.river_shapefile),
            &path_to_string(&river_out),
        )?;
    } else if request.source_usage.river {
        crate::status!(
            "[data-process] river distance exists, skip: {}",
            river_out.display()
        );
    }

    let road_out = resampled_dir.join("road");
    if request.source_usage.road {
        crate::status!(
            "[data-process] road/railway/waterway distance AOI-masked precompute/check -> {}",
            road_out.display()
        );
        let _ = road_distance::compute_road_distance_rasters_rasterized(
            &class_grid,
            &aoi_mask,
            &request.sources.aoi_shapefile,
            &path_to_string(&request.sources.road_dir),
            &path_to_string(&road_out),
        )?;
    }

    let training_points_path = features_dir.join("training_points.parquet");
    if request.source_usage.ccdc && !training_points_path.exists() {
        crate::status!(
            "[data-process] training point generation years {:?} -> {}",
            request.training_years,
            training_points_path.display()
        );
        let mut df = training_points::generate_training_points_for_years(
            &class_path,
            &path_to_string(&ccdc_out),
            &aoi_mask,
            20260604,
            &request.training_years,
        )?;
        training_points::write_training_points(&mut df, &path_to_string(&training_points_path))?;
    } else if training_points_path.exists() {
        crate::status!(
            "[data-process] training points exist, skip: {}",
            training_points_path.display()
        );
    }

    let training_features_path = features_dir.join("training_features.parquet");
    if training_points_path.exists() && !training_features_path.exists() {
        crate::status!(
            "[data-process] training feature extraction -> {}",
            training_features_path.display()
        );
        feature_extraction::extract_training_features_with_workers(
            &path_to_string(&training_points_path),
            &path_to_string(&training_features_path),
            request.workers,
        )?;
    } else if training_features_path.exists() {
        crate::status!(
            "[data-process] training features exist, skip: {}",
            training_features_path.display()
        );
    }

    let prediction_dir = features_dir.join(format!("prediction_{}", request.prediction_year));
    crate::status!(
        "[data-process] prediction feature extraction -> {}",
        prediction_dir.display()
    );
    let prediction_tiles = feature_extraction::extract_prediction_features_with_workers(
        &class_grid,
        &aoi_mask,
        &path_to_string(&prediction_dir),
        request.tile_size,
        &path_to_string(&resampled_dir),
        request.prediction_year,
        request.workers,
    )?;

    crate::status!(
        "[data-process] complete: prediction_tiles={}, cache={}",
        prediction_tiles,
        cache_dir.display()
    );

    Ok(DataProcessRunReport {
        class_width: class_grid.width(),
        class_height: class_grid.height(),
        processing_years,
        training_years: request.training_years.clone(),
        prediction_year: request.prediction_year,
        prediction_tiles,
        cache_dir: cache_dir.clone(),
    })
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn min_max_year(years: &[i32]) -> Option<(i32, i32)> {
    let min_year = years.iter().min().copied()?;
    let max_year = years.iter().max().copied()?;
    Some((min_year, max_year))
}

fn discover_landuse_years(landuse_dir: &Path) -> LoaderResult<Vec<i32>> {
    let mut years = std::collections::BTreeSet::new();
    collect_landuse_years_recursive(landuse_dir, &mut years)?;
    Ok(years.into_iter().collect())
}

fn collect_landuse_years_recursive(
    dir: &Path,
    years: &mut std::collections::BTreeSet<i32>,
) -> LoaderResult<()> {
    for entry in std::fs::read_dir(dir).map_err(LoaderError::Io)? {
        let entry = entry.map_err(LoaderError::Io)?;
        let path = entry.path();
        if path.is_dir() {
            collect_landuse_years_recursive(&path, years)?;
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(year) = parse_year_after_token(name, "CLCD_v01_") {
            years.insert(year);
        }
    }
    Ok(())
}

fn discover_resident_rasters(resident_dir: &Path) -> LoaderResult<Vec<(i32, PathBuf)>> {
    let mut rasters = Vec::new();
    for entry in std::fs::read_dir(resident_dir).map_err(LoaderError::Io)? {
        let entry = entry.map_err(LoaderError::Io)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("GURS_") || !name.ends_with(".tif") {
            continue;
        }
        if let Some(year) = parse_year_after_token(name, "GURS_") {
            rasters.push((year, path));
        }
    }
    rasters.sort_by_key(|(year, _)| *year);
    Ok(rasters)
}

fn parse_year_after_token(name: &str, token: &str) -> Option<i32> {
    let start = name.find(token)? + token.len();
    let year = name.get(start..start + 4)?;
    year.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn grid() -> GeoGrid {
        GeoGrid {
            path: PathBuf::from("class.tif"),
            width: 10,
            height: 10,
            geo_transform: [0.0, 1.0, 0.0, 10.0, 0.0, -1.0],
            crs_wkt: "EPSG:4326".into(),
            band_count: 10,
            start_year: 2016,
        }
    }

    #[test]
    fn plan_uses_expanded_years() {
        let request = DataProcessRequest {
            class_raster: PathBuf::from("class.tif"),
            class_start_year: 2016,
            sources: DataProcessSources {
                aoi_shapefile: PathBuf::from("aoi.shp"),
                dem_raster: PathBuf::from("dem.tif"),
                ccdc_dir: PathBuf::from("ccdc"),
                rainfall_dir: PathBuf::from("rainfall"),
                temperature_dir: PathBuf::from("temperature"),
                landuse_dir: PathBuf::from("landuse"),
                resident_raster: PathBuf::from("resident.tif"),
                river_shapefile: PathBuf::from("river.shp"),
                road_dir: PathBuf::from("road"),
            },
            source_usage: DataSourceUsage::default(),
            cache_dir: PathBuf::from(".cache"),
            tile_size: 1024,
            processing_years: (2016..=2025).collect(),
            training_years: (2017..=2024).collect(),
            prediction_year: 2025,
            workers: 0,
        };
        let plan = DataProcessPlan::from_request(&request, grid());
        assert!(plan.processing_years.contains(&2016));
        assert!(plan.processing_years.contains(&2025));
        assert_eq!(plan.training_years[0], 2017);
    }
}

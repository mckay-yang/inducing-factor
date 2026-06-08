//! CLASS-centric data preprocessing plan.
//!
//! This crate owns the data_process stage boundary.  It must operate in
//! geospatial coordinates by using [`gdal::GeoGrid`] metadata as the common
//! reference for resampling, clipping, and tiling.

pub mod aoi_mask;
pub mod class_grid;
pub mod feature_extraction;
pub mod resample;
pub mod river_distance;
pub mod road_distance;
pub mod training_points;

use gdal::{GeoGrid, GridWindow};
use loaders::base::LoaderResult;
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
        features_dir.clone(),
    ] {
        std::fs::create_dir_all(dir)?;
    }

    let dem_out = resampled_dir.join("dem.tif");
    if request.source_usage.dem && !dem_out.exists() {
        resample::resample_dem_to_class(
            &class_grid,
            &path_to_string(&request.sources.dem_raster),
            &path_to_string(&dem_out),
        )?;
    }

    let ccdc_out = resampled_dir.join("ccdc");
    if request.source_usage.ccdc {
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
        let _ = resample::resample_temperature_to_class_for_years_with_workers(
            &class_grid,
            &path_to_string(&request.sources.temperature_dir),
            &path_to_string(&temperature_out),
            year_range,
            request.workers,
            Some(&path_to_string(&request.sources.aoi_shapefile)),
        )?;
    }

    let landuse_out = resampled_dir.join("landuse/landuse.tif");
    if request.source_usage.landuse && !landuse_out.exists() {
        resample::resample_landuse_to_class(
            &class_grid,
            &path_to_string(&request.sources.landuse_dir),
            &path_to_string(&landuse_out),
        )?;
    }

    let resident_out = resampled_dir.join("resident/resident.tif");
    if request.source_usage.resident && !resident_out.exists() {
        resample::resample_resident_to_class(
            &class_grid,
            &path_to_string(&request.sources.resident_raster),
            &path_to_string(&resident_out),
        )?;
    }

    let river_out = resampled_dir.join("river/river_dist.tif");
    if request.source_usage.river && !river_out.exists() {
        river_distance::compute_river_distance_raster(
            &class_grid,
            &path_to_string(&request.sources.river_shapefile),
            &path_to_string(&river_out),
        )?;
    }

    let road_out = resampled_dir.join("road");
    if request.source_usage.road {
        let _ = road_distance::compute_road_distance_rasters(
            &class_grid,
            &path_to_string(&request.sources.road_dir),
            &path_to_string(&road_out),
        )?;
    }

    let training_points_path = features_dir.join("training_points.parquet");
    if request.source_usage.ccdc && !training_points_path.exists() {
        let mut df = training_points::generate_training_points_for_years(
            &class_path,
            &path_to_string(&ccdc_out),
            &aoi_mask,
            20260604,
            &request.training_years,
        )?;
        training_points::write_training_points(&mut df, &path_to_string(&training_points_path))?;
    }

    let training_features_path = features_dir.join("training_features.parquet");
    if training_points_path.exists() && !training_features_path.exists() {
        feature_extraction::extract_training_features_with_workers(
            &path_to_string(&training_points_path),
            &path_to_string(&training_features_path),
            request.workers,
        )?;
    }

    let prediction_dir = features_dir.join(format!("prediction_{}", request.prediction_year));
    let prediction_tiles = feature_extraction::extract_prediction_features_with_workers(
        &class_grid,
        &aoi_mask,
        &path_to_string(&prediction_dir),
        request.tile_size,
        &path_to_string(&resampled_dir),
        request.prediction_year,
        request.workers,
    )?;

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

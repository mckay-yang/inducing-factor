//! Typed configuration model for the inducing-factor pipeline.
//!
//! Deserialized from YAML via `serde`. Sections mirror the conceptual
//! pipeline stages: paths, chunking, runtime, attribution hyper-parameters,
//! and data-source descriptors.
//!
//! # Path resolution
//! Relative paths stored in [`PathsConfig`] and [`DataSourceConfig`] are
//! resolved against the *directory containing the configuration file* when
//! [`AppConfig::resolve_paths`] is called.  Absolute paths are left unchanged.

use serde::Deserialize;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

// ── Worker count ────────────────────────────────────────────────

/// Controls the number of worker threads used by the pipeline.
///
/// When serialized from YAML the field accepts:
///   - `"auto"` → [`Workers::Auto`] (use all logical CPUs)
///   - a positive integer → [`Workers::Count`] with that many threads
///   - `0`  → rejected by validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Workers {
    /// Use `std::thread::available_parallelism()` at runtime.
    Auto,
    /// Fixed number of workers (≥ 1).
    Count(NonZeroUsize),
}

impl Default for Workers {
    fn default() -> Self {
        Workers::Auto
    }
}

impl<'de> Deserialize<'de> for Workers {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct WorkersVisitor;

        impl<'de> serde::de::Visitor<'de> for WorkersVisitor {
            type Value = Workers;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str(r#"a positive integer or the string "auto""#)
            }

            fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<Workers, E> {
                match s {
                    "auto" => Ok(Workers::Auto),
                    other => Err(E::custom(format!(
                        "unknown workers value: \"{}\"; expected \"auto\" or a positive integer",
                        other
                    ))),
                }
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Workers, E> {
                if v == 0 {
                    Ok(Workers::Auto)
                } else if v < 0 {
                    Err(E::custom(format!(
                        "workers cannot be negative ({}); use 0, \"auto\", or a positive integer",
                        v
                    )))
                } else {
                    Ok(Workers::Count(NonZeroUsize::new(v as usize).unwrap()))
                }
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Workers, E> {
                if v == 0 {
                    Ok(Workers::Auto)
                } else {
                    Ok(Workers::Count(NonZeroUsize::new(v as usize).unwrap()))
                }
            }
        }

        deserializer.deserialize_any(WorkersVisitor)
    }
}

// ── Paths ───────────────────────────────────────────────────────

/// Filesystem locations for cache, output, and (optionally) data root.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PathsConfig {
    /// Root directory for all input datasets.
    /// When set, relative paths in `data_sources` are resolved from this
    /// directory rather than the config-file directory.
    #[serde(default)]
    pub data_dir: Option<PathBuf>,

    /// Directory where chunked intermediate `.npz` files are stored.
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,

    /// Directory where model artifacts and results are written.
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            data_dir: None,
            cache_dir: default_cache_dir(),
            output_dir: default_output_dir(),
        }
    }
}

fn default_cache_dir() -> PathBuf {
    PathBuf::from("cache/chunks")
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("results/models")
}

// ── Chunking ────────────────────────────────────────────────────

/// Parameters that control how the raster extent is split into tiles.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ChunkConfig {
    /// Side-length of a square chunk in pixels (e.g. 1024).
    #[serde(default = "default_tile_size")]
    pub tile_size: u32,

    /// Path to the AOI shapefile used to clip processing.
    /// Relative to the config-file directory (or `paths.data_dir`).
    #[serde(default)]
    pub aoi_shapefile: Option<PathBuf>,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            tile_size: default_tile_size(),
            aoi_shapefile: None,
        }
    }
}

fn default_tile_size() -> u32 {
    1024
}

// ── Pipeline runtime ────────────────────────────────────────────

/// Run-time execution settings: concurrency, memory, years.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    /// Number of parallel worker threads.
    #[serde(default)]
    pub workers: Workers,

    /// Soft memory budget in GiB.  Workers will auto-scale if set.
    #[serde(default)]
    pub memory_limit_gb: Option<f64>,

    /// Fraction of system RAM the pipeline may use (0.0–1.0).
    /// Used when `memory_limit_gb` is not set. Defaults to 0.7.
    #[serde(default = "default_memory_fraction")]
    pub memory_fraction: f64,

    /// Estimated memory per worker in GiB (used for auto-scaling).
    /// If not set, derived from chunk size.
    #[serde(default)]
    pub estimated_worker_memory_gb: Option<f64>,

    /// First year of the analysis window (inclusive).
    pub start_year: i32,

    /// Last year of the analysis window (inclusive).
    pub end_year: i32,
}

fn default_memory_fraction() -> f64 {
    0.7
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            workers: Workers::default(),
            memory_limit_gb: None,
            memory_fraction: default_memory_fraction(),
            estimated_worker_memory_gb: None,
            start_year: 2000,
            end_year: 2020,
        }
    }
}

// ── Attribution hyper-parameters ────────────────────────────────

/// Model hyper-parameters for the factor-mining and Bayesian-attribution
/// stage.  Values correspond to [`super::MANIFEST.pipeline_parameters`]
/// in the golden-baseline corpus.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AttributionConfig {
    /// Years of history used for rolling-window features.
    #[serde(default = "default_lookback_years")]
    pub lookback_years: usize,

    /// Fraction of background point-ids sampled for training.
    #[serde(default = "default_background_sample_rate")]
    pub background_sample_rate: f64,

    /// Maximum number of feature conditions in a single factor pattern.
    #[serde(default = "default_max_pattern_size")]
    pub max_pattern_size: usize,

    /// Minimum event-support fraction for a candidate pattern.
    #[serde(default = "default_min_support")]
    pub min_support: f64,

    /// Minimum event/background risk ratio.
    #[serde(default = "default_min_relative_risk")]
    pub min_relative_risk: f64,

    /// Number of bootstrap resamples for stability estimation.
    #[serde(default = "default_bootstrap_iterations")]
    pub bootstrap_iterations: usize,

    /// Minimum bootstrap retention rate.
    #[serde(default = "default_min_stability")]
    pub min_stability: f64,

    /// Maximum number of samples per bootstrap iteration.
    #[serde(default = "default_max_bootstrap_samples")]
    pub max_bootstrap_samples: usize,

    /// Maximum factors fed to the Bayesian network.
    #[serde(default = "default_max_network_factors")]
    pub max_network_factors: usize,

    /// Discretization strategy: `"supervised_tree"` or `"quantile"`.
    #[serde(default = "default_discretizer_method")]
    pub discretizer_method: String,

    /// Number of bins for continuous features.
    #[serde(default = "default_discretize_bins")]
    pub discretize_bins: usize,

    /// PRNG seed for reproducibility.
    #[serde(default = "default_random_state")]
    pub random_state: u64,

    /// Cap on rows used to fit the discretizer (streaming).
    #[serde(default = "default_sample_limit")]
    pub discretizer_sample_limit: usize,

    /// Cap on rows used for model training (streaming).
    #[serde(default = "default_sample_limit")]
    pub model_sample_limit: usize,

    /// Threshold for single-factor-dominant attribution.
    #[serde(default = "default_dominant_threshold")]
    pub dominant_threshold: f64,

    /// Threshold for multi-factor-synergy attribution.
    #[serde(default = "default_synergy_threshold")]
    pub synergy_threshold: f64,
}

impl Default for AttributionConfig {
    fn default() -> Self {
        Self {
            lookback_years: default_lookback_years(),
            background_sample_rate: default_background_sample_rate(),
            max_pattern_size: default_max_pattern_size(),
            min_support: default_min_support(),
            min_relative_risk: default_min_relative_risk(),
            bootstrap_iterations: default_bootstrap_iterations(),
            min_stability: default_min_stability(),
            max_bootstrap_samples: default_max_bootstrap_samples(),
            max_network_factors: default_max_network_factors(),
            discretizer_method: default_discretizer_method(),
            discretize_bins: default_discretize_bins(),
            random_state: default_random_state(),
            discretizer_sample_limit: default_sample_limit(),
            model_sample_limit: default_sample_limit(),
            dominant_threshold: default_dominant_threshold(),
            synergy_threshold: default_synergy_threshold(),
        }
    }
}

fn default_lookback_years() -> usize {
    3
}
fn default_background_sample_rate() -> f64 {
    0.05
}
fn default_max_pattern_size() -> usize {
    3
}
fn default_min_support() -> f64 {
    0.05
}
fn default_min_relative_risk() -> f64 {
    1.5
}
fn default_bootstrap_iterations() -> usize {
    100
}
fn default_min_stability() -> f64 {
    0.6
}
fn default_max_bootstrap_samples() -> usize {
    3000
}
fn default_max_network_factors() -> usize {
    6
}
fn default_discretizer_method() -> String {
    "supervised_tree".into()
}
fn default_discretize_bins() -> usize {
    3
}
fn default_random_state() -> u64 {
    42
}
fn default_sample_limit() -> usize {
    200_000
}
fn default_dominant_threshold() -> f64 {
    0.55
}
fn default_synergy_threshold() -> f64 {
    0.75
}

// ── Data sources ────────────────────────────────────────────────

/// Declares the on-disk location of every input dataset the pipeline
/// consumes.  Paths may be absolute or relative; relative paths are
/// resolved against *either* the config-file directory or
/// [`PathsConfig::data_dir`] when it is set.
///
/// File-name patterns use `{year}` and `{month}` placeholders that the
/// loaders expand at runtime.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DataSourceConfig {
    /// AOI boundary shapefile.
    pub aoi_shapefile: Option<PathBuf>,

    /// Digital elevation model raster.
    pub dem_raster: Option<PathBuf>,

    /// Directory containing yearly land-use rasters.
    pub landuse_dir: Option<PathBuf>,

    /// Directory containing monthly precipitation rasters.
    pub rainfall_dir: Option<PathBuf>,

    /// Directory containing yearly temperature NetCDF files.
    pub temperature_dir: Option<PathBuf>,

    /// Directory containing yearly residential-density rasters.
    pub resident_dir: Option<PathBuf>,

    /// Directory containing river vector data.
    pub river_dir: Option<PathBuf>,

    /// Directory containing yearly road/railway/waterway shapefiles.
    pub road_dir: Option<PathBuf>,

    /// Directory containing yearly CCDC disturbance rasters.
    pub ccdc_dir: Option<PathBuf>,

    /// Disturbance-classification raster (single file, all years).
    pub disturbance_class_raster: Option<PathBuf>,

    // ── File-name patterns ──────────────────────────────────
    /// Pattern for land-use rasters, e.g. `CLCD_v01_{year}_albert.tif`.
    #[serde(default = "default_landuse_pattern")]
    pub landuse_file_pattern: String,

    /// Pattern for yearly road shapefiles, e.g. `osm_{year}_roads.shp`.
    #[serde(default = "default_road_pattern")]
    pub road_file_pattern: String,

    /// Pattern for yearly railway shapefiles.
    #[serde(default = "default_railway_pattern")]
    pub railway_file_pattern: String,

    /// Pattern for yearly waterway shapefiles.
    #[serde(default = "default_waterway_pattern")]
    pub waterway_file_pattern: String,

    /// Pattern for monthly rainfall rasters, e.g. `pre{year}{month:02d}.tif`.
    /// The `{month:02d}` placeholder expects zero-padded month numbers.
    #[serde(default = "default_rainfall_monthly_pattern")]
    pub rainfall_monthly_pattern: String,

    /// Pattern for yearly temperature files, e.g. `tmp_{year}.nc`.
    #[serde(default = "default_temperature_pattern")]
    pub temperature_file_pattern: String,

    /// Pattern for yearly residential files, e.g. `GURS_{year}.tif`.
    #[serde(default = "default_resident_pattern")]
    pub resident_file_pattern: String,

    /// Pattern for yearly CCDC files, e.g. `{year}.tif`.
    #[serde(default = "default_ccdc_pattern")]
    pub ccdc_file_pattern: String,

    /// First year encoded in the disturbance-class raster.
    #[serde(default = "default_disturbance_class_start_year")]
    pub disturbance_class_start_year: i32,

    /// Sentinel "no-data" value in the DEM raster.
    #[serde(default = "default_dem_nodata")]
    pub dem_nodata: f64,

    /// Sentinel "no-data" value in residential rasters.
    #[serde(default = "default_resident_nodata")]
    pub resident_nodata: f64,
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self {
            aoi_shapefile: None,
            dem_raster: None,
            landuse_dir: None,
            rainfall_dir: None,
            temperature_dir: None,
            resident_dir: None,
            river_dir: None,
            road_dir: None,
            ccdc_dir: None,
            disturbance_class_raster: None,
            landuse_file_pattern: default_landuse_pattern(),
            road_file_pattern: default_road_pattern(),
            railway_file_pattern: default_railway_pattern(),
            waterway_file_pattern: default_waterway_pattern(),
            rainfall_monthly_pattern: default_rainfall_monthly_pattern(),
            temperature_file_pattern: default_temperature_pattern(),
            resident_file_pattern: default_resident_pattern(),
            ccdc_file_pattern: default_ccdc_pattern(),
            disturbance_class_start_year: default_disturbance_class_start_year(),
            dem_nodata: default_dem_nodata(),
            resident_nodata: default_resident_nodata(),
        }
    }
}

fn default_landuse_pattern() -> String {
    "CLCD_v01_{year}_albert_province/CLCD_v01_{year}_albert_xizang.tif".into()
}
fn default_road_pattern() -> String {
    "osm_{year}_roads.shp".into()
}
fn default_railway_pattern() -> String {
    "osm_{year}_railways.shp".into()
}
fn default_waterway_pattern() -> String {
    "osm_{year}_waterways.shp".into()
}
fn default_rainfall_monthly_pattern() -> String {
    "pre{year}{month:02d}.tif".into()
}
fn default_temperature_pattern() -> String {
    "tmp_{year}.nc".into()
}
fn default_resident_pattern() -> String {
    "GURS_{year}.tif".into()
}
fn default_ccdc_pattern() -> String {
    "{year}.tif".into()
}
fn default_disturbance_class_start_year() -> i32 {
    2016
}
fn default_dem_nodata() -> f64 {
    -9999.0
}
fn default_resident_nodata() -> f64 {
    127.0
}

// ── Top-level config ────────────────────────────────────────────

/// Root configuration object deserialized from a YAML file.
///
/// # Example YAML
///
/// ```yaml
/// paths:
///   cache_dir: "cache/chunks"
///   output_dir: "results/models"
///
/// chunks:
///   tile_size: 1024
///
/// pipeline:
///   workers: "auto"
///   start_year: 2016
///   end_year: 2016
///
/// attribution:
///   lookback_years: 1
///   background_sample_rate: 0.5
///
/// data_sources:
///   aoi_shapefile: "data/aoi/LangToBasighat.shp"
///   dem_raster: "data/dem/tibet_dem.tif"
///   landuse_dir: "data/landuse"
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// Filesystem locations.
    #[serde(default)]
    pub paths: PathsConfig,

    /// Chunk / tile settings.
    #[serde(default)]
    pub chunks: ChunkConfig,

    /// Runtime execution settings.
    #[serde(default)]
    pub pipeline: PipelineConfig,

    /// Attribution-model hyper-parameters.
    #[serde(default)]
    pub attribution: AttributionConfig,

    /// Data-source paths and file-name patterns.
    #[serde(default)]
    pub data_sources: DataSourceConfig,

    // ── Internal — set after construction, not deserialized ──
    /// Directory that contained the YAML file (populated by [`Self::from_file`]).
    #[serde(skip)]
    pub(crate) config_dir: Option<PathBuf>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            paths: PathsConfig::default(),
            chunks: ChunkConfig::default(),
            pipeline: PipelineConfig::default(),
            attribution: AttributionConfig::default(),
            data_sources: DataSourceConfig::default(),
            config_dir: None,
        }
    }
}

impl AppConfig {
    /// Load configuration from a YAML file, record its directory, and
    /// validate all fields.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, crate::ConfigError> {
        let path = path.as_ref();
        let yaml = std::fs::read_to_string(path).map_err(|e| crate::ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mut config: AppConfig =
            serde_yaml::from_str(&yaml).map_err(|e| crate::ConfigError::Deserialize {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        // Record the directory so relative-paths can be resolved later.
        if let Some(parent) = path.parent() {
            config.config_dir = Some(parent.to_path_buf());
        } else {
            config.config_dir = Some(PathBuf::from("."));
        }

        // Validate.
        crate::validation::validate(&config, Some(path))?;

        Ok(config)
    }

    /// Resolve every relative path in the configuration against the
    /// config-file directory (or [`PathsConfig::data_dir`] if set).
    ///
    /// Absolute paths are left untouched.  Returns a new `AppConfig` with
    /// all paths resolved — the original is not modified.
    pub fn resolve_paths(&self) -> AppConfig {
        let base = self
            .paths
            .data_dir
            .as_deref()
            .or(self.config_dir.as_deref())
            .unwrap_or_else(|| Path::new("."));

        let resolve = |p: &Option<PathBuf>| -> Option<PathBuf> {
            p.as_ref().map(|p| {
                if p.is_absolute() {
                    p.clone()
                } else {
                    base.join(p)
                }
            })
        };

        AppConfig {
            paths: PathsConfig {
                data_dir: self.paths.data_dir.clone(),
                cache_dir: if self.paths.cache_dir.is_absolute() {
                    self.paths.cache_dir.clone()
                } else {
                    base.join(&self.paths.cache_dir)
                },
                output_dir: if self.paths.output_dir.is_absolute() {
                    self.paths.output_dir.clone()
                } else {
                    base.join(&self.paths.output_dir)
                },
            },
            chunks: ChunkConfig {
                tile_size: self.chunks.tile_size,
                aoi_shapefile: resolve(&self.chunks.aoi_shapefile),
            },
            pipeline: self.pipeline.clone(),
            attribution: self.attribution.clone(),
            data_sources: DataSourceConfig {
                aoi_shapefile: resolve(&self.data_sources.aoi_shapefile),
                dem_raster: resolve(&self.data_sources.dem_raster),
                landuse_dir: resolve(&self.data_sources.landuse_dir),
                rainfall_dir: resolve(&self.data_sources.rainfall_dir),
                temperature_dir: resolve(&self.data_sources.temperature_dir),
                resident_dir: resolve(&self.data_sources.resident_dir),
                river_dir: resolve(&self.data_sources.river_dir),
                road_dir: resolve(&self.data_sources.road_dir),
                ccdc_dir: resolve(&self.data_sources.ccdc_dir),
                disturbance_class_raster: resolve(&self.data_sources.disturbance_class_raster),
                ..self.data_sources.clone()
            },
            config_dir: self.config_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_deserialize_auto() {
        let w: Workers = serde_yaml::from_str("\"auto\"").unwrap();
        assert_eq!(w, Workers::Auto);
    }

    #[test]
    fn workers_deserialize_count() {
        let w: Workers = serde_yaml::from_str("4").unwrap();
        assert_eq!(w, Workers::Count(NonZeroUsize::new(4).unwrap()));
    }

    #[test]
    fn workers_deserialize_zero_is_auto() {
        let w: Workers = serde_yaml::from_str("0").unwrap();
        assert_eq!(w, Workers::Auto);
    }

    #[test]
    fn appconfig_default_has_sane_values() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.chunks.tile_size, 1024);
        assert_eq!(cfg.pipeline.workers, Workers::Auto);
        assert_eq!(cfg.attribution.lookback_years, 3);
        assert_eq!(cfg.attribution.random_state, 42);
    }
}

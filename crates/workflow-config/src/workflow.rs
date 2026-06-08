//! JSON workflow configuration for the CLASS-centric Rust architecture.
//!
//! It describes the three high-level stages that the CLI orchestrates:
//!
//! - data_process: align all geospatial inputs to the CLASS grid
//! - train: train one basin-wide model from explicit training years
//! - predict: generate one or more products for a prediction year

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::ConfigError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WorkflowConfig {
    pub class_grid: ClassGridConfig,
    pub data_sources: WorkflowDataSources,
    pub data_usage: WorkflowDataUsage,
    pub data_process: DataProcessConfig,
    pub train: TrainConfig,
    pub predict: PredictionConfig,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            class_grid: ClassGridConfig::default(),
            data_sources: WorkflowDataSources::default(),
            data_usage: WorkflowDataUsage::default(),
            data_process: DataProcessConfig::default(),
            train: TrainConfig::default(),
            predict: PredictionConfig::default(),
        }
    }
}

impl WorkflowConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let cfg: WorkflowConfig =
            serde_json::from_str(&raw).map_err(|e| ConfigError::Deserialize {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.train.training_years.is_empty() {
            return Err(ConfigError::field(
                "train.training_years",
                "must list every training year explicitly",
            ));
        }

        let mut seen = BTreeSet::new();
        for &year in &self.train.training_years {
            if !seen.insert(year) {
                return Err(ConfigError::field(
                    "train.training_years",
                    format!("duplicate year {year}"),
                ));
            }
            if !(1900..=2100).contains(&year) {
                return Err(ConfigError::field(
                    "train.training_years",
                    format!("year {year} is outside the supported range 1900..=2100"),
                ));
            }
        }

        if self.predict.prediction_year < 1900 || self.predict.prediction_year > 2100 {
            return Err(ConfigError::field(
                "predict.prediction_year",
                format!(
                    "year {} is outside the supported range 1900..=2100",
                    self.predict.prediction_year
                ),
            ));
        }

        if self.data_process.temporal_context_years < 0 {
            return Err(ConfigError::field(
                "data_process.temporal_context_years",
                "must be >= 0",
            ));
        }

        if self.data_process.tile_size == 0 {
            return Err(ConfigError::field("data_process.tile_size", "must be >= 1"));
        }

        Ok(())
    }

    /// Years that data processing should prepare for time-dependent datasets.
    ///
    /// This expands the explicit training years by `temporal_context_years` on
    /// both sides and includes the prediction year.  For example,
    /// training_years=[2017, 2020] and context=1 yields
    /// [2016, 2017, 2018, 2019, 2020, 2021].
    pub fn processing_years(&self) -> Vec<i32> {
        let min_train = self.train.training_years.iter().min().copied();
        let max_train = self.train.training_years.iter().max().copied();
        let context = self.data_process.temporal_context_years;

        let mut years = BTreeSet::new();
        if let (Some(min_year), Some(max_year)) = (min_train, max_train) {
            for year in (min_year - context)..=(max_year + context) {
                years.insert(year);
            }
        }
        years.insert(self.predict.prediction_year);
        years.into_iter().collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ClassGridConfig {
    pub raster: PathBuf,
    pub start_year: i32,
}

impl Default for ClassGridConfig {
    fn default() -> Self {
        Self {
            raster: PathBuf::from("data/disturbance_class/disturbance_class_res.tif"),
            start_year: 2016,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WorkflowDataSources {
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

impl Default for WorkflowDataSources {
    fn default() -> Self {
        Self {
            aoi_shapefile: PathBuf::from("data/aoi/LangToBasighat.shp"),
            dem_raster: PathBuf::from("data/dem/tibet_dem.tif"),
            ccdc_dir: PathBuf::from("data/ccdc_res"),
            rainfall_dir: PathBuf::from("data/rainfall"),
            temperature_dir: PathBuf::from("data/temperature"),
            landuse_dir: PathBuf::from("data/landuse"),
            resident_raster: PathBuf::from("data/resident/GURS_2020.tif"),
            river_shapefile: PathBuf::from("data/river/As_15s_Chimaps.shp"),
            road_dir: PathBuf::from("data/road"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WorkflowDataUsage {
    pub production: DataUsageConfig,
    pub smoke_test: DataUsageConfig,
}

impl Default for WorkflowDataUsage {
    fn default() -> Self {
        Self {
            production: DataUsageConfig::default(),
            smoke_test: DataUsageConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DataUsageConfig {
    pub dem: bool,
    pub ccdc: bool,
    pub rainfall: bool,
    pub temperature: bool,
    pub landuse: bool,
    pub resident: bool,
    pub river: bool,
    pub road: bool,
}

impl Default for DataUsageConfig {
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
#[serde(default)]
pub struct DataProcessConfig {
    pub cache_dir: PathBuf,
    pub tile_size: usize,
    pub temporal_context_years: i32,
    pub workers: usize,
}

impl Default for DataProcessConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".cache"),
            tile_size: 1024,
            temporal_context_years: 1,
            workers: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrainConfig {
    pub training_years: Vec<i32>,
    pub model_dir: PathBuf,
    pub missing_data_policy: ModelMissingDataPolicy,
    pub discretizer_method: String,
    pub discretize_bins: usize,
    pub max_pattern_size: usize,
    pub min_support: f64,
    pub min_relative_risk: f64,
    pub min_stability: f64,
    pub bootstrap_iterations: usize,
    pub max_bootstrap_samples: usize,
    pub max_network_factors: usize,
    pub random_state: u64,
    pub workers: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            training_years: vec![2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            model_dir: PathBuf::from("results/models"),
            missing_data_policy: ModelMissingDataPolicy::NativeMissingCategory,
            discretizer_method: "supervised_tree".into(),
            discretize_bins: 3,
            max_pattern_size: 3,
            min_support: 0.05,
            min_relative_risk: 1.5,
            min_stability: 0.6,
            bootstrap_iterations: 100,
            max_bootstrap_samples: 3000,
            max_network_factors: 6,
            random_state: 42,
            workers: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelMissingDataPolicy {
    /// Preserve missing feature values as explicit "missing" states so the
    /// model can still score pixels without a complete covariate stack.
    NativeMissingCategory,
    /// Add synthetic feature-dropout rows during training.  This is a planned
    /// extension; the config supports it before implementation is complete.
    FeatureDropout,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PredictionConfig {
    pub prediction_year: i32,
    pub output_dir: PathBuf,
    pub products: Vec<OutputProduct>,
    pub workers: usize,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            prediction_year: 2025,
            output_dir: PathBuf::from("results/predictions"),
            products: vec![OutputProduct::Probability],
            workers: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputProduct {
    Probability,
    DominantFactor,
    DisturbanceType,
    DisturbanceTypeProbability,
    FactorActivation,
    MissingDataMask,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processing_years_expand_training_span() {
        let cfg = WorkflowConfig {
            train: TrainConfig {
                training_years: vec![2017, 2020],
                ..TrainConfig::default()
            },
            predict: PredictionConfig {
                prediction_year: 2020,
                ..PredictionConfig::default()
            },
            data_process: DataProcessConfig {
                temporal_context_years: 1,
                ..DataProcessConfig::default()
            },
            ..WorkflowConfig::default()
        };
        assert_eq!(
            cfg.processing_years(),
            vec![2016, 2017, 2018, 2019, 2020, 2021]
        );
    }

    #[test]
    fn json_roundtrip_uses_explicit_training_years() {
        let raw = r#"{
          "data_usage": {
            "production": {"road": false},
            "smoke_test": {"river": false, "road": false}
          },
          "train": {"training_years": [2017, 2018, 2019]},
          "predict": {"prediction_year": 2025}
        }"#;
        let cfg: WorkflowConfig = serde_json::from_str(raw).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.train.training_years, vec![2017, 2018, 2019]);
        assert!(cfg.processing_years().contains(&2025));
        assert!(!cfg.data_usage.production.road);
        assert!(!cfg.data_usage.smoke_test.river);
        assert!(cfg.data_usage.production.dem);
    }
}

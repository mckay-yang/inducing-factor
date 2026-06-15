//! Shared model artifact protocol.
//!
//! `train` writes this format and `predict` reads it.  Keeping the
//! artifact contract in its own crate prevents train and predict from depending
//! on each other.

use factor_core::discretize::AdaptiveDiscretizer;
use factor_core::model::{
    DisturbanceClassEngine, DisturbanceTypeDiagnostics, OccurrenceProbabilityEngine,
};
use factor_core::types::InducingFactor;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ModelArtifactError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinModelArtifact {
    pub version: String,
    pub feature_columns: Vec<String>,
    pub discretizer: AdaptiveDiscretizer,
    pub selected_factors: Vec<InducingFactor>,
    pub occurrence_engine: OccurrenceProbabilityEngine,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_engine: Option<DisturbanceClassEngine>,
    pub n_rows: usize,
    pub n_events: usize,
    pub n_background: usize,
    pub backend: String,
    pub missing_data_policy: String,
    pub random_state: u64,
    #[serde(default)]
    pub type_diagnostics: Option<DisturbanceTypeDiagnostics>,
    #[serde(default)]
    pub prediction_feature_assumptions: serde_json::Value,
}

impl BasinModelArtifact {
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), ModelArtifactError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let raw = serde_json::to_string_pretty(self)?;
        std::fs::write(path, raw)?;
        Ok(())
    }
}

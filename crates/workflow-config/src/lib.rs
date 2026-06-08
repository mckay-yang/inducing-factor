//! JSON workflow configuration for the CLASS-centric Rust architecture.

pub mod workflow;

use std::path::PathBuf;

pub use workflow::{
    ClassGridConfig, DataProcessConfig, DataUsageConfig, ModelMissingDataPolicy, OutputProduct,
    PredictionConfig, TrainConfig, WorkflowConfig, WorkflowDataSources, WorkflowDataUsage,
};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file `{}`: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to deserialize config file `{}`: {message}", path.display())]
    Deserialize { path: PathBuf, message: String },

    #[error("`{field}`: {message}")]
    Field { field: String, message: String },
}

impl ConfigError {
    pub fn field(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Field {
            field: field.into(),
            message: message.into(),
        }
    }
}

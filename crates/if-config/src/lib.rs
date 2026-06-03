//! Configuration loading, typed deserialization, and validation for the
//! inducing-factor pipeline.
//!
//! # Quick start
//!
//! ```no_run
//! use if_config::AppConfig;
//!
//! let cfg = AppConfig::from_file("configs/example.yaml")?;
//! let resolved = cfg.resolve_paths();
//! # Ok::<(), if_config::ConfigError>(())
//! ```
//!
//! # Path resolution
//!
//! Relative paths in [`PathsConfig`] and [`DataSourceConfig`] are resolved
//! against the directory containing the config file when
//! [`AppConfig::resolve_paths`] is called.  Absolute paths are left
//! unchanged.  When [`PathsConfig::data_dir`] is set it overrides the
//! config-file directory as the resolution base.

pub mod config;
pub mod validation;

pub use config::{
    AppConfig, AttributionConfig, ChunkConfig, DataSourceConfig, PathsConfig, PipelineConfig,
    Workers,
};
pub use validation::ConfigError;

use std::path::{Path, PathBuf};

use crate::config::AppConfig;

/// Errors produced during config loading and validation.
///
/// Every variant that relates to a config field includes the dot-joined
/// field path (e.g. `pipeline.start_year`) so callers can report exactly
/// which value is invalid.
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
        ConfigError::Field {
            field: field.into(),
            message: message.into(),
        }
    }
}

/// Run all validation checks on `config`, returning the first error
/// encountered.
pub(crate) fn validate(config: &AppConfig, config_path: Option<&Path>) -> Result<(), ConfigError> {
    validate_pipeline(&config.pipeline)?;
    validate_attribution(&config.attribution)?;
    validate_data_sources(&config.data_sources, config_path)?;
    Ok(())
}

fn validate_pipeline(p: &crate::config::PipelineConfig) -> Result<(), ConfigError> {
    if p.start_year > p.end_year {
        return Err(ConfigError::field(
            "pipeline.start_year",
            format!(
                "start_year ({}) must not be greater than end_year ({})",
                p.start_year, p.end_year
            ),
        ));
    }

    if p.start_year < 1900 {
        return Err(ConfigError::field(
            "pipeline.start_year",
            format!(
                "start_year ({}) is implausibly early; expected ≥ 1900",
                p.start_year
            ),
        ));
    }

    if p.end_year > 2100 {
        return Err(ConfigError::field(
            "pipeline.end_year",
            format!(
                "end_year ({}) is implausibly far in the future; expected ≤ 2100",
                p.end_year
            ),
        ));
    }

    if let Some(gb) = p.memory_limit_gb {
        if gb <= 0.0 {
            return Err(ConfigError::field(
                "pipeline.memory_limit_gb",
                format!("memory_limit_gb ({}) must be positive when set", gb),
            ));
        }
    }

    if p.memory_fraction <= 0.0 || p.memory_fraction > 1.0 {
        return Err(ConfigError::field(
            "pipeline.memory_fraction",
            format!("memory_fraction ({}) must be in (0, 1]", p.memory_fraction),
        ));
    }

    if let Some(gb) = p.estimated_worker_memory_gb {
        if gb <= 0.0 {
            return Err(ConfigError::field(
                "pipeline.estimated_worker_memory_gb",
                format!(
                    "estimated_worker_memory_gb ({}) must be positive when set",
                    gb
                ),
            ));
        }
    }

    Ok(())
}

fn validate_attribution(a: &crate::config::AttributionConfig) -> Result<(), ConfigError> {
    if a.lookback_years == 0 {
        return Err(ConfigError::field(
            "attribution.lookback_years",
            "lookback_years must be ≥ 1 to produce meaningful lag features",
        ));
    }

    if a.background_sample_rate <= 0.0 || a.background_sample_rate > 1.0 {
        return Err(ConfigError::field(
            "attribution.background_sample_rate",
            format!(
                "background_sample_rate ({}) must be in (0, 1]",
                a.background_sample_rate
            ),
        ));
    }

    if a.max_pattern_size == 0 {
        return Err(ConfigError::field(
            "attribution.max_pattern_size",
            "max_pattern_size must be ≥ 1",
        ));
    }

    if a.min_support <= 0.0 || a.min_support > 1.0 {
        return Err(ConfigError::field(
            "attribution.min_support",
            format!("min_support ({}) must be in (0, 1]", a.min_support),
        ));
    }

    if a.min_relative_risk < 1.0 {
        return Err(ConfigError::field(
            "attribution.min_relative_risk",
            format!("min_relative_risk ({}) must be ≥ 1.0", a.min_relative_risk),
        ));
    }

    if a.min_stability < 0.0 || a.min_stability > 1.0 {
        return Err(ConfigError::field(
            "attribution.min_stability",
            format!("min_stability ({}) must be in [0, 1]", a.min_stability),
        ));
    }

    if a.bootstrap_iterations == 0 && a.min_stability > 0.0 {
        return Err(ConfigError::field(
            "attribution.bootstrap_iterations",
            "bootstrap_iterations must be ≥ 1 when min_stability > 0",
        ));
    }

    if a.max_network_factors < 2 {
        return Err(ConfigError::field(
            "attribution.max_network_factors",
            format!(
                "max_network_factors ({}) must be ≥ 2 to build a useful network",
                a.max_network_factors
            ),
        ));
    }

    if a.discretize_bins < 2 {
        return Err(ConfigError::field(
            "attribution.discretize_bins",
            format!(
                "discretize_bins ({}) must be ≥ 2 to create meaningful bins",
                a.discretize_bins
            ),
        ));
    }

    let method = a.discretizer_method.as_str();
    if method != "supervised_tree" && method != "quantile" {
        return Err(ConfigError::field(
            "attribution.discretizer_method",
            format!(
                "unknown discretizer_method \"{}\"; expected \"supervised_tree\" or \"quantile\"",
                method
            ),
        ));
    }

    if a.dominant_threshold < 0.0 || a.dominant_threshold > 1.0 {
        return Err(ConfigError::field(
            "attribution.dominant_threshold",
            format!(
                "dominant_threshold ({}) must be in [0, 1]",
                a.dominant_threshold
            ),
        ));
    }

    if a.synergy_threshold < 0.0 || a.synergy_threshold > 1.0 {
        return Err(ConfigError::field(
            "attribution.synergy_threshold",
            format!(
                "synergy_threshold ({}) must be in [0, 1]",
                a.synergy_threshold
            ),
        ));
    }

    if a.discretizer_sample_limit == 0 {
        return Err(ConfigError::field(
            "attribution.discretizer_sample_limit",
            "discretizer_sample_limit must be ≥ 1",
        ));
    }

    if a.model_sample_limit == 0 {
        return Err(ConfigError::field(
            "attribution.model_sample_limit",
            "model_sample_limit must be ≥ 1",
        ));
    }

    Ok(())
}

fn validate_data_sources(
    ds: &crate::config::DataSourceConfig,
    _config_path: Option<&Path>,
) -> Result<(), ConfigError> {
    // Required data sources — the pipeline cannot run without these.
    let required: &[(&str, &Option<PathBuf>)] = &[
        ("data_sources.landuse_dir", &ds.landuse_dir),
        ("data_sources.rainfall_dir", &ds.rainfall_dir),
        ("data_sources.river_dir", &ds.river_dir),
        ("data_sources.road_dir", &ds.road_dir),
        ("data_sources.ccdc_dir", &ds.ccdc_dir),
    ];

    for (field, value) in required {
        if value.is_none() {
            return Err(ConfigError::field(
                *field,
                format!(
                    "{} is required but missing; set it to the path of the corresponding dataset directory",
                    field
                ),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn valid_appconfig() -> AppConfig {
        AppConfig {
            data_sources: DataSourceConfig {
                landuse_dir: Some("data/landuse".into()),
                rainfall_dir: Some("data/rainfall".into()),
                river_dir: Some("data/river".into()),
                road_dir: Some("data/road".into()),
                ccdc_dir: Some("data/ccdc_res".into()),
                ..DataSourceConfig::default()
            },
            ..AppConfig::default()
        }
    }

    #[test]
    fn valid_config_passes() {
        let cfg = valid_appconfig();
        assert!(validate(&cfg, None).is_ok());
    }

    #[test]
    fn start_year_after_end_year_fails() {
        let mut cfg = valid_appconfig();
        cfg.pipeline.start_year = 2025;
        cfg.pipeline.end_year = 2020;
        let err = validate(&cfg, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("pipeline.start_year"), "msg: {}", msg);
        assert!(msg.contains("2025"), "msg: {}", msg);
        assert!(msg.contains("2020"), "msg: {}", msg);
    }

    #[test]
    fn zero_lookback_rejected() {
        let mut cfg = valid_appconfig();
        cfg.attribution.lookback_years = 0;
        let err = validate(&cfg, None).unwrap_err();
        assert!(err.to_string().contains("attribution.lookback_years"));
    }

    #[test]
    fn bad_background_rate_rejected() {
        for rate in &[-0.1, 0.0, 1.5] {
            let mut cfg = valid_appconfig();
            cfg.attribution.background_sample_rate = *rate;
            let err = validate(&cfg, None).unwrap_err();
            assert!(err
                .to_string()
                .contains("attribution.background_sample_rate"));
        }
    }

    #[test]
    fn low_relative_risk_rejected() {
        let mut cfg = valid_appconfig();
        cfg.attribution.min_relative_risk = 0.5;
        let err = validate(&cfg, None).unwrap_err();
        assert!(err.to_string().contains("attribution.min_relative_risk"));
    }

    #[test]
    fn unknown_discretizer_rejected() {
        let mut cfg = valid_appconfig();
        cfg.attribution.discretizer_method = "kmeans".into();
        let err = validate(&cfg, None).unwrap_err();
        assert!(err.to_string().contains("attribution.discretizer_method"));
        assert!(err.to_string().contains("kmeans"));
    }

    #[test]
    fn missing_required_data_source_field() {
        let mut cfg = valid_appconfig();
        cfg.data_sources.landuse_dir = None;
        let err = validate(&cfg, None).unwrap_err();
        assert!(err.to_string().contains("data_sources.landuse_dir"));
        assert!(err.to_string().contains("required"));
    }

    #[test]
    fn missing_multiple_data_sources_reports_first() {
        let mut cfg = valid_appconfig();
        cfg.data_sources.landuse_dir = None;
        cfg.data_sources.rainfall_dir = None;
        let err = validate(&cfg, None).unwrap_err();
        // We check required fields in order, so landuse_dir should be first.
        assert!(err.to_string().contains("data_sources.landuse_dir"));
    }

    #[test]
    fn out_of_range_stability_rejected() {
        for val in &[-0.1, 1.5] {
            let mut cfg = valid_appconfig();
            cfg.attribution.min_stability = *val;
            let err = validate(&cfg, None).unwrap_err();
            assert!(err.to_string().contains("attribution.min_stability"));
        }
    }

    #[test]
    fn too_few_bins_rejected() {
        let mut cfg = valid_appconfig();
        cfg.attribution.discretize_bins = 1;
        let err = validate(&cfg, None).unwrap_err();
        assert!(err.to_string().contains("attribution.discretize_bins"));
    }

    #[test]
    fn too_few_network_factors_rejected() {
        let mut cfg = valid_appconfig();
        cfg.attribution.max_network_factors = 1;
        let err = validate(&cfg, None).unwrap_err();
        assert!(err.to_string().contains("attribution.max_network_factors"));
    }

    #[test]
    fn memory_fraction_out_of_range_rejected() {
        for val in &[-0.1, 0.0, 1.1, 2.0] {
            let mut cfg = valid_appconfig();
            cfg.pipeline.memory_fraction = *val;
            let err = validate(&cfg, None).unwrap_err();
            assert!(
                err.to_string().contains("pipeline.memory_fraction"),
                "expected memory_fraction error for {}, got: {}",
                val,
                err
            );
        }
    }

    #[test]
    fn negative_memory_limit_rejected() {
        let mut cfg = valid_appconfig();
        cfg.pipeline.memory_limit_gb = Some(-5.0);
        let err = validate(&cfg, None).unwrap_err();
        assert!(err.to_string().contains("pipeline.memory_limit_gb"));
    }

    #[test]
    fn negative_estimated_worker_memory_rejected() {
        let mut cfg = valid_appconfig();
        cfg.pipeline.estimated_worker_memory_gb = Some(-0.1);
        let err = validate(&cfg, None).unwrap_err();
        assert!(err.to_string().contains("estimated_worker_memory_gb"));
    }
}

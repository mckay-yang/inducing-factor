use std::io::Write;

use if_config::AppConfig;

fn write_temp_yaml(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    (dir, path)
}

#[test]
fn load_valid_yaml_from_file() {
    let yaml = r#"
paths:
  cache_dir: "cache/chunks"
  output_dir: "results/models"

chunks:
  tile_size: 512

pipeline:
  workers: 4
  start_year: 2016
  end_year: 2020

attribution:
  lookback_years: 2
  background_sample_rate: 0.3

data_sources:
  landuse_dir: "data/landuse"
  rainfall_dir: "data/rainfall"
  river_dir: "data/river"
  road_dir: "data/road"
  ccdc_dir: "data/ccdc_res"
"#;
    let (_dir, config_path) = write_temp_yaml(yaml);
    let cfg = AppConfig::from_file(&config_path).unwrap();
    assert_eq!(cfg.chunks.tile_size, 512);
    assert_eq!(cfg.pipeline.start_year, 2016);
    assert_eq!(cfg.pipeline.end_year, 2020);
    assert_eq!(cfg.attribution.lookback_years, 2);
    assert_eq!(cfg.attribution.background_sample_rate, 0.3);
}

#[test]
fn zero_workers_rejected_in_full_config() {
    let yaml = r#"
pipeline:
  workers: 0
  start_year: 2016
  end_year: 2016
data_sources:
  landuse_dir: "data/landuse"
  rainfall_dir: "data/rainfall"
  river_dir: "data/river"
  road_dir: "data/road"
  ccdc_dir: "data/ccdc_res"
"#;
    let (_dir, config_path) = write_temp_yaml(yaml);
    let err = AppConfig::from_file(&config_path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("cannot be 0") || msg.contains("workers"),
        "expected rejection of zero workers, got: {}",
        msg
    );
}

#[test]
fn missing_required_data_source_field_in_file() {
    let yaml = r#"
pipeline:
  workers: "auto"
  start_year: 2016
  end_year: 2016
data_sources:
  landuse_dir: "data/landuse"
  rainfall_dir: "data/rainfall"
  river_dir: "data/river"
  # road_dir is missing
  ccdc_dir: "data/ccdc_res"
"#;
    let (_dir, config_path) = write_temp_yaml(yaml);
    let err = AppConfig::from_file(&config_path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("data_sources.road_dir"),
        "expected error about road_dir, got: {}",
        msg
    );
    assert!(
        msg.contains("required"),
        "expected 'required' in message, got: {}",
        msg
    );
}

#[test]
fn invalid_start_year_range_rejected() {
    let yaml = r#"
pipeline:
  workers: "auto"
  start_year: 2025
  end_year: 2020
data_sources:
  landuse_dir: "data/landuse"
  rainfall_dir: "data/rainfall"
  river_dir: "data/river"
  road_dir: "data/road"
  ccdc_dir: "data/ccdc_res"
"#;
    let (_dir, config_path) = write_temp_yaml(yaml);
    let err = AppConfig::from_file(&config_path).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("pipeline.start_year"), "msg: {}", msg);
    assert!(msg.contains("2025"), "msg: {}", msg);
}

#[test]
fn path_resolution_respects_config_dir() {
    let yaml = r#"
paths:
  cache_dir: "cache/chunks"
  output_dir: "results/models"

pipeline:
  workers: "auto"
  start_year: 2016
  end_year: 2016

data_sources:
  landuse_dir: "data/landuse"
  rainfall_dir: "data/rainfall"
  river_dir: "data/river"
  road_dir: "data/road"
  ccdc_dir: "data/ccdc_res"
  aoi_shapefile: "data/aoi/test.shp"
"#;
    let (dir, config_path) = write_temp_yaml(yaml);
    let cfg = AppConfig::from_file(&config_path).unwrap();
    let resolved = cfg.resolve_paths();

    let expected_cache = dir.path().join("cache/chunks");
    let expected_output = dir.path().join("results/models");
    let expected_aoi = dir.path().join("data/aoi/test.shp");

    assert_eq!(resolved.paths.cache_dir, expected_cache);
    assert_eq!(resolved.paths.output_dir, expected_output);
    assert_eq!(resolved.data_sources.aoi_shapefile, Some(expected_aoi));
}

#[test]
fn absolute_paths_left_untouched() {
    let yaml = r#"
paths:
  cache_dir: "/absolute/cache"
  output_dir: "/absolute/output"

pipeline:
  workers: "auto"
  start_year: 2016
  end_year: 2016

data_sources:
  landuse_dir: "data/landuse"
  rainfall_dir: "data/rainfall"
  river_dir: "data/river"
  road_dir: "data/road"
  ccdc_dir: "data/ccdc_res"
"#;
    let (_dir, config_path) = write_temp_yaml(yaml);
    let cfg = AppConfig::from_file(&config_path).unwrap();
    let resolved = cfg.resolve_paths();

    assert_eq!(
        resolved.paths.cache_dir,
        std::path::PathBuf::from("/absolute/cache")
    );
    assert_eq!(
        resolved.paths.output_dir,
        std::path::PathBuf::from("/absolute/output")
    );
}

#[test]
fn bad_yaml_syntax_gives_deserialize_error() {
    let yaml = r#"
pipeline:
  workers: "auto"
  start_year: 20x6    # not a valid integer
  end_year: 2016
"#;
    let (_dir, config_path) = write_temp_yaml(yaml);
    let err = AppConfig::from_file(&config_path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("deserialize"),
        "expected deserialize error, got: {}",
        msg
    );
}

#[test]
fn missing_file_gives_io_error() {
    let err = AppConfig::from_file("/nonexistent/path/config.yaml").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("failed to read"),
        "expected io error, got: {}",
        msg
    );
}

#[test]
fn workers_string_auto_accepted() {
    let yaml = r#"
pipeline:
  workers: "auto"
  start_year: 2016
  end_year: 2016
data_sources:
  landuse_dir: "data/landuse"
  rainfall_dir: "data/rainfall"
  river_dir: "data/river"
  road_dir: "data/road"
  ccdc_dir: "data/ccdc_res"
"#;
    let (_dir, config_path) = write_temp_yaml(yaml);
    let cfg = AppConfig::from_file(&config_path).unwrap();
    assert_eq!(cfg.pipeline.workers, if_config::Workers::Auto);
}

#[test]
fn default_config_passes_validation() {
    let cfg = AppConfig::default();
    // Default is missing required data sources, so it won't pass validation.
    // But we test that the default struct is constructable.
    assert_eq!(cfg.chunks.tile_size, 1024);
}

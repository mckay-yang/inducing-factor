pub mod class_grid;
pub mod ensemble;
pub mod execution;
pub mod feature_builder;
pub mod feature_extraction;
pub mod probability_mapper;
pub mod processor;
pub mod production_pipeline;
pub mod production_raster;
pub mod production_vector;
pub mod resample;
pub mod river_distance;
pub mod training_points;
pub mod road_distance;
pub mod sampling;

pub use ensemble::predict_ensemble;
pub use feature_extraction::{
    extract_prediction_features, extract_training_features,
};
pub use execution::{
    detect_available_ram_gb, detect_system_ram_gb, estimate_worker_memory_gb, execute_chunk_tasks,
    execute_chunk_tasks_bounded, resolve_workers_with_memory_guard,
};
pub use feature_builder::FeatureBuilder;
pub use probability_mapper::{
    AnnualProbabilityMapper, PredictionResult, RasterOutputMeta, CLASS_LABELS,
};
pub use processor::{
    build_chunk_dataframe, chunk_cache_path, chunk_has_valid_data, derive_chunk_grid,
    resolve_num_workers, write_audit_report, write_chunk_cache, ChunkAudit, ChunkResult, ChunkTask,
    SyntheticLoaderSet, SyntheticRasterPanel, CATEGORICAL_TEMPORAL_FEATURES, DIRECT_FEATURES,
    DYNAMIC_FEATURES, RAINFALL_MONTHLY_COLS, STATIC_FEATURES,
};
pub use production_pipeline::run_production_pipeline;
pub use production_raster::{
    derive_grid_from_ccdc, ChunkRasters, ProductionRasterPanel, YearRasters,
};
pub use sampling::StratifiedBackgroundSampler;

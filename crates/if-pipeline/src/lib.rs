pub mod processor;
pub mod feature_builder;
pub mod sampling;
pub mod execution;

pub use processor::{
    build_chunk_dataframe, chunk_cache_path, chunk_has_valid_data, derive_chunk_grid,
    resolve_num_workers, write_audit_report, write_chunk_cache, ChunkAudit, ChunkResult, ChunkTask,
    SyntheticLoaderSet, SyntheticRasterPanel, CATEGORICAL_TEMPORAL_FEATURES, DIRECT_FEATURES,
    DYNAMIC_FEATURES, RAINFALL_MONTHLY_COLS, STATIC_FEATURES,
};
pub use feature_builder::FeatureBuilder;
pub use sampling::StratifiedBackgroundSampler;
pub use execution::{execute_chunk_tasks, execute_chunk_tasks_bounded};

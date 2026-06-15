from inducing_factor.config.loader import load_config


def test_production_config_uses_safe_parallel_defaults():
    config = load_config("configs/production.yaml")

    assert config.pipeline.chunk_size == 1024
    assert config.pipeline.num_workers == 0
    assert config.pipeline.max_auto_workers == 4
    assert config.pipeline.memory_target_fraction == 0.7
    assert config.pipeline.estimated_worker_gb == 3.0
    assert config.attribution.run_num_workers == 0
    assert config.attribution.predict_num_workers == 0


def test_production_fresh_config_uses_parallel_cache_dir():
    config = load_config("configs/production_fresh.yaml")

    assert config.paths.cache_dir == "data/cache/chunks"
    assert config.paths.out_dir == "data/result/models"
    assert config.pipeline.chunk_size == 1024
    assert config.pipeline.num_workers == 0
    assert config.attribution.run_num_workers == 0
    assert config.attribution.predict_num_workers == 0

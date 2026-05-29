"""Quick end-to-end test using a few small chunks to verify Stage 2+3 pipeline."""
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEST_CACHE = "data/.cache/quick_test_chunks"
SRC_CACHE = "data/.cache/chunks_aoi_v3_subset"
OUT_DIR = "data/.cache/quick_test_output"
LOG_FILE = "logs/run/quick_test.log"

SMALL_CHUNKS = [
    "chunk_74.npz",
    "chunk_245.npz",
    "chunk_16.npz",
    "chunk_181.npz",
    "chunk_19.npz",
    "chunk_10.npz",
    "chunk_202.npz",
    "chunk_31.npz",
    "chunk_20.npz",
    "chunk_52.npz",
]


def setup():
    os.makedirs(TEST_CACHE, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    for name in SMALL_CHUNKS:
        src = os.path.join(SRC_CACHE, name)
        dst = os.path.join(TEST_CACHE, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  copied {name}")
        else:
            print(f"  exists {name}")


def run():
    from inducing_factor.config import load_config
    from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem

    config = load_config("configs/production_subset.yaml")
    config.paths.cache_dir = TEST_CACHE
    config.paths.out_dir = OUT_DIR
    config.attribution.run_num_workers = 1
    config.attribution.run_max_auto_workers = 1
    config.attribution.max_network_factors = 6
    config.attribution.bootstrap_iterations = 10
    config.attribution.bn_bootstrap_iterations = 3

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="w"),
        ],
    )

    system = EcoDisturbanceAttributionSystem(config=config)
    try:
        result = system.run()
        print("\n=== SUCCESS ===")
        print(f"Model backend: {system.model_backend}")
        print(f"Feature columns: {system.feature_columns}")
        print(f"Selected factors: {len(system.selected_factors)}")
        for f in system.selected_factors:
            print(f"  {f.factor_id}: items={f.items}, rr={f.relative_risk:.2f}")
    except Exception as e:
        print(f"\n=== FAILED: {type(e).__name__}: {e} ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Setting up test chunks...")
    setup()
    print("\nRunning quick test...")
    run()

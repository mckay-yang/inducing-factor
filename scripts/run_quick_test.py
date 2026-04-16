#!/usr/bin/env python
"""
Quick test of the full pipeline with AOI-driven processing and attribution.
Uses limited years and chunk size for speed.
"""
import logging
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inducing_factor.config.loader import load_config
from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem
from inducing_factor.pipeline.processor import PipelineProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QuickTest")

def main():
    config_path = project_root / "configs" / "quick_test.yaml"
    logger.info("Loading configuration from %s", config_path)
    try:
        config = load_config(str(config_path))
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        traceback.print_exc()
        return 1

    logger.info("Starting pipeline processing (limited extent)...")
    try:
        processor = PipelineProcessor(config=config)
        processor.process_all_chunks()
    except Exception as e:
        logger.error("Pipeline processing failed: %s", e)
        traceback.print_exc()
        # If failure is due to missing data, we can still try attribution with existing cache
        logger.info("Continuing with attribution using existing cache (if any)...")

    logger.info("Starting attribution system...")
    try:
        attribution = EcoDisturbanceAttributionSystem(config=config)
        attribution.run()
        logger.info("Attribution completed successfully.")
    except Exception as e:
        logger.error("Attribution failed: %s", e)
        traceback.print_exc()
        return 1

    logger.info("Quick test finished. Output in %s", config.paths.out_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
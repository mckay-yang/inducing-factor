import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inducing_factor.config.loader import load_config
from inducing_factor.pipeline.processor import PipelineProcessor

config_path = "configs/quick_multi.yaml"
print(f"Loading config from {config_path}")
config = load_config(config_path)

print("Initializing pipeline processor...")
processor = PipelineProcessor(config=config)

print("Running pipeline...")
processor.process_all_chunks()

print("Pipeline completed.")
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem
from inducing_factor.config.loader import load_config

config_path = "configs/quick_multi.yaml"
print(f"Loading config from {config_path}")
config = load_config(config_path)

# Override paths to use multi cache
config.paths.cache_dir = "results/cache_multi"
config.paths.out_dir = "results/output_multi"
print(f"Using cache_dir: {config.paths.cache_dir}")
print(f"Using out_dir: {config.paths.out_dir}")

print("Initializing attribution system...")
system = EcoDisturbanceAttributionSystem(config=config)

print("Running attribution...")
system.run()

print("Attribution completed.")
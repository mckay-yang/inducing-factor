import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem
from inducing_factor.config.loader import load_config

config_path = "configs/quick_test.yaml"
print(f"Loading config from {config_path}")
config = load_config(config_path)

print("Initializing attribution system...")
system = EcoDisturbanceAttributionSystem(config=config)

print("Running attribution...")
system.run()

print("Attribution completed.")
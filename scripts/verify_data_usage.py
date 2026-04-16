import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inducing_factor import EcoDisturbanceAttributionSystem
from inducing_factor.utils import build_synthetic_panel, save_as_npz_chunks


def main():
    cache_dir = "results/cache"
    out_dir = "results/verify"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    save_as_npz_chunks(build_synthetic_panel(), cache_dir)

    total_rows = 0
    for name in sorted(f for f in os.listdir(cache_dir) if f.endswith(".npz")):
        data = np.load(os.path.join(cache_dir, name))
        total_rows += len(data[data.files[0]])

    system = EcoDisturbanceAttributionSystem(
        cache_dir=cache_dir,
        out_dir=out_dir,
        lookback_years=2,
        background_sample_rate=0.3,
        max_pattern_size=3,
        min_support=0.08,
        min_relative_risk=1.2,
        bootstrap_iterations=5,
        min_stability=0.4,
        max_bootstrap_samples=1000,
        max_network_factors=6,
    )
    samples = system.build_training_samples()
    factors = system.mine_inducing_factors(samples)
    print(f"raw_rows={total_rows}")
    print(f"training_rows={len(samples)}")
    print(f"factors={len(factors)}")
    print(f"stability_eq_1={(factors['stability'] == 1.0).sum()}")


if __name__ == "__main__":
    main()

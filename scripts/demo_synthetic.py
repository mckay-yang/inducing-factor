import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inducing_factor import EcoDisturbanceAttributionSystem
from inducing_factor.utils import build_synthetic_panel, save_as_npz_chunks


def main():
    df = build_synthetic_panel()
    cache_dir = "results/cache"
    out_dir = "results/output"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    save_as_npz_chunks(df, cache_dir)
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
    result = system.run()
    print(f"training_samples={len(result['training_samples'])}")
    print(f"factors={len(result['factors'])}")
    print(result["factors"][["factor_id", "semantics", "relative_risk", "stability"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

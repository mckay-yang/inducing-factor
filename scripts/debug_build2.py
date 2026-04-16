import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem

# Load cache
cache_dir = "results/cache_real"
files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.npz')]
print(f"Found {len(files)} cache files")
for f in files[:1]:
    print(f"Loading {f}")
    data = np.load(f)
    df = pd.DataFrame({k: data[k] for k in data.files})
    print(f"Chunk shape: {df.shape}")
    print(f"Disturbance type >0: {(df['disturbance_type'] > 0).sum()}")
    # Initialize system with default config
    system = EcoDisturbanceAttributionSystem(
        cache_dir=cache_dir,
        out_dir="results/output_real",
        lookback_years=1,
        background_sample_rate=0.5,
        discretizer_method="supervised_tree",
        discretize_bins=3,
        max_pattern_size=3,
        min_support=0.05,
        min_relative_risk=1.1,
        bootstrap_iterations=5,
        min_stability=0.4,
        max_bootstrap_samples=1000,
        max_network_factors=6,
        dominant_threshold=0.55,
        synergy_threshold=0.75,
        random_state=42
    )
    # Call internal method
    try:
        df_samples = system._build_chunk_training_samples(df)
        print(f"Built samples shape: {df_samples.shape}")
        if df_samples.empty:
            print("Samples are empty")
            # Let's examine intermediate steps
            df_prepared = system._assign_reference_months(df)
            print(f"reference_month distribution:")
            print(df_prepared['reference_month'].value_counts().sort_index())
            # Use system.feature_builder
            df_feat, required_cols = system.feature_builder.build_chunk_features(df_prepared)
            print(f"Required cols: {required_cols}")
            print(f"Number of required cols: {len(required_cols)}")
            if required_cols:
                history_mask = df_feat[required_cols].notna().all(axis=1)
                print(f"History mask true count: {history_mask.sum()}")
                df_ready = df_feat.loc[history_mask, system.metadata_columns + required_cols].copy()
                print(f"df_ready shape: {df_ready.shape}")
                print(f"Target distribution in df_ready: {df_ready[system.target].value_counts() if not df_ready.empty else 'empty'}")
            else:
                print("No required columns")
        else:
            print(f"Samples columns: {df_samples.columns.tolist()}")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
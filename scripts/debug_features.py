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
    print(f"\n--- Processing {os.path.basename(f)} ---")
    data = np.load(f)
    df = pd.DataFrame({k: data[k] for k in data.files})
    print(f"Chunk shape: {df.shape}")
    
    # Initialize system with relaxed parameters
    system = EcoDisturbanceAttributionSystem(
        cache_dir=cache_dir,
        out_dir="results/output_real",
        lookback_years=1,  # Reduced from 3 to 1
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
    
    # Step 1: Assign reference months
    df_prepared = system._assign_reference_months(df)
    print(f"\n1. After assign_reference_months:")
    print(f"   reference_month unique: {df_prepared['reference_month'].unique()}")
    
    # Step 2: Build features
    df_feat, required_cols = system.feature_builder.build_chunk_features(df_prepared)
    print(f"\n2. After build_chunk_features:")
    print(f"   Required columns ({len(required_cols)}): {required_cols}")
    
    # Check each required column for NaN
    print(f"\n3. NaN analysis per required column:")
    for col in required_cols:
        nan_count = df_feat[col].isna().sum()
        total = len(df_feat)
        pct = nan_count / total * 100
        print(f"   {col}: {nan_count}/{total} NaN ({pct:.1f}%)")
    
    # Check history mask
    print(f"\n4. History mask analysis:")
    history_mask = df_feat[required_cols].notna().all(axis=1)
    print(f"   history_mask true: {history_mask.sum()}/{len(df_feat)} ({history_mask.sum()/len(df_feat)*100:.1f}%)")
    
    if history_mask.sum() > 0:
        df_ready = df_feat.loc[history_mask, system.metadata_columns + required_cols].copy()
        print(f"\n5. Ready data (shape={df_ready.shape}):")
        print(f"   Target distribution: {df_ready[system.target].value_counts()}")
        
        # Check sample building
        df_dist = df_ready[df_ready[system.target] == 1]
        df_bg = df_ready[df_ready[system.target] == 0]
        print(f"   Disturbance samples: {len(df_dist)}")
        print(f"   Background samples: {len(df_bg)}")
    else:
        print(f"\n5. No samples pass history mask")
        
        # Let's see which columns are causing issues
        print(f"\n6. Column-wise completeness:")
        for col in required_cols:
            col_complete = df_feat[col].notna().sum()
            col_total = len(df_feat)
            print(f"   {col}: {col_complete}/{col_total} ({col_complete/col_total*100:.1f}%)")
            
            # If column has many NaN, show first few rows
            if col_complete < col_total and col_complete > 0:
                non_nan_sample = df_feat[col].dropna().iloc[:3].tolist()
                print(f"     Sample values: {non_nan_sample}")
            elif col_complete == 0:
                print(f"     ALL NaN!")
                
    # Also check the feature building logic for resident
    print(f"\n7. Feature builder configuration:")
    print(f"   dynamic_features: {system.dynamic_features}")
    print(f"   categorical_temporal_features: {system.categorical_temporal_features}")
    print(f"   direct_features: {system.direct_features}")
    print(f"   lookback_years: {system.feature_builder.lookback_years}")
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
from inducing_factor.utils.synthetic import load_chunk_cache

cache_dir = "results/cache_real"
print("Loading cache from", cache_dir)
df = load_chunk_cache(cache_dir)
print("DataFrame shape:", df.shape)
if df is not None and len(df) > 0:
    print("Columns:", df.columns.tolist())
    print("Years:", df['year'].unique())
    if 'disturbance_type' in df.columns:
        print("Disturbance type counts:")
        print(df['disturbance_type'].value_counts().head())
        print("Non-zero disturbance points:", (df['disturbance_type'] > 0).sum())
    if 'disturbance_month' in df.columns:
        print("Disturbance month unique:", df['disturbance_month'].unique())
    # Check rainfall columns
    rain_cols = [c for c in df.columns if c.startswith('rainfall')]
    print("Rainfall columns:", rain_cols)
    # Show sample rows
    print("\nFirst 5 rows:")
    print(df.head())
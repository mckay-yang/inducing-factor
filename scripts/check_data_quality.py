import numpy as np
import pandas as pd
import os
import glob

cache_dir = "results/cache_real"
files = glob.glob(os.path.join(cache_dir, "chunk_*.npz"))
print(f"Found {len(files)} cache files")

for f in files[:2]:  # Check first two files
    print(f"\n--- Analyzing {os.path.basename(f)} ---")
    data = np.load(f)
    df = pd.DataFrame({k: data[k] for k in data.files})
    
    # Check resident column
    if 'resident' in df.columns:
        resident_vals = df['resident']
        print(f"Resident column: dtype={resident_vals.dtype}")
        print(f"NaN count: {resident_vals.isna().sum()}/{len(resident_vals)}")
        print(f"Unique values: {resident_vals.unique()[:10]}")  # Show first 10 unique values
        print(f"Min: {resident_vals.min()}, Max: {resident_vals.max()}")
    else:
        print("No resident column found")
    
    # Check temperature column
    if 'temperature' in df.columns:
        temp_vals = df['temperature']
        print(f"\nTemperature column: dtype={temp_vals.dtype}")
        print(f"NaN count: {temp_vals.isna().sum()}/{len(temp_vals)}")
        print(f"Unique values: {temp_vals.unique()[:10]}")
        print(f"Min: {temp_vals.min()}, Max: {temp_vals.max()}")
    else:
        print("No temperature column found")
    
    # Check rainfall columns
    rain_cols = [c for c in df.columns if c.startswith('rainfall')]
    print(f"\nRainfall columns: {len(rain_cols)}")
    for col in ['rainfall_month_01', 'rainfall_month_07', 'rainfall_month_12']:
        if col in df.columns:
            vals = df[col]
            print(f"  {col}: NaN={vals.isna().sum()}/{len(vals)}, Min={vals.min():.2f}, Max={vals.max():.2f}")
    
    # Check disturbance data
    if 'disturbance_type' in df.columns:
        dist = df['disturbance_type']
        print(f"\nDisturbance type >0: {(dist > 0).sum()}/{len(dist)}")
        if (dist > 0).sum() > 0:
            print(f"  Disturbance values: {np.unique(dist[dist > 0])[:10]}")
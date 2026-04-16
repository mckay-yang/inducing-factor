import numpy as np
import pandas as pd
import os
import glob

cache_dir = "results/cache_real"
files = glob.glob(os.path.join(cache_dir, "chunk_*.npz"))
print(f"Found {len(files)} cache files")
all_data = []
for f in files:
    data = np.load(f)
    # Convert to dict of columns
    df = pd.DataFrame({k: data[k] for k in data.files})
    all_data.append(df)
if all_data:
    df = pd.concat(all_data, ignore_index=True)
    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nDisturbance type distribution:")
    print(df['disturbance_type'].value_counts())
    print("\nDisturbance month distribution:")
    print(df['disturbance_month'].value_counts())
    print("\nYear distribution:")
    print(df['year'].value_counts().sort_index())
    # Check rainfall columns
    rain_cols = [c for c in df.columns if c.startswith('rainfall')]
    print(f"\n{len(rain_cols)} rainfall columns: {rain_cols}")
    # Check if any disturbance_type > 0
    disturbance_mask = df['disturbance_type'] > 0
    print(f"\nDisturbance points: {disturbance_mask.sum()} out of {len(df)}")
    if disturbance_mask.any():
        print("Sample disturbance rows:")
        print(df[disturbance_mask].head())
    else:
        print("No disturbance points found.")
else:
    print("No data loaded.")
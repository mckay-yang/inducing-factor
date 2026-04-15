import numpy as np
import pandas as pd
import sys
import os

chunk_path = "results/cache_quick/chunk_21.npz"
if not os.path.exists(chunk_path):
    print(f"File not found: {chunk_path}")
    sys.exit(1)

data = np.load(chunk_path, allow_pickle=True)
print("Keys in npz:", list(data.keys()))
for key in data:
    val = data[key]
    if isinstance(val, np.ndarray):
        print(f"{key}: shape={val.shape}, dtype={val.dtype}")
        if val.ndim == 1 and len(val) > 0:
            print(f"  first few values: {val[:5]}")
    else:
        print(f"{key}: type={type(val)}, value={val}")

# Try to load as dataframe
if 'data' in data:
    df = data['data']
    if isinstance(df, pd.DataFrame):
        print("\nDataFrame columns:", df.columns.tolist())
        print(f"Rows: {len(df)}")
        if len(df) > 0:
            print("\nFirst few rows:")
            print(df.head())
    else:
        print(f"'data' is not a DataFrame, it's {type(df)}")
        # maybe it's a dictionary?
        if isinstance(df, dict):
            print("Dictionary keys:", df.keys())
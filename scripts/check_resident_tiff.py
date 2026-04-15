#!/usr/bin/env python3
import rasterio
import numpy as np
import sys

tiff_path = "/Volumes/T9/Datasets/resident/方创琳教授团队发布全球首套100 m城市和农村居住区数据集(2000-2020年)/GURS_2015.tif"
try:
    with rasterio.open(tiff_path) as src:
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Shape: {src.shape}")
        print(f"Dtype: {src.dtypes[0]}")
        print(f"NoData value: {src.nodata}")
        
        # Read a sample window (1000x1000) to check values
        window = rasterio.windows.Window(0, 0, min(1000, src.width), min(1000, src.height))
        data = src.read(1, window=window)
        print(f"\nSample window shape: {data.shape}")
        print(f"Sample unique values: {np.unique(data)}")
        print(f"Sample value range: {np.min(data)} to {np.max(data)}")
        
        # Check if 127 appears
        count_127 = np.sum(data == 127)
        count_total = data.size
        print(f"Count of value 127: {count_127}/{count_total} ({count_127/count_total*100:.2f}%)")
        
        # Check other values
        unique_vals = np.unique(data)
        for val in unique_vals:
            count = np.sum(data == val)
            print(f"  Value {val}: {count} pixels ({count/count_total*100:.2f}%)")
        
        # Check if this might be a binary mask (0/1) or continuous values
        if len(unique_vals) < 10:
            print(f"\nFew unique values ({len(unique_vals)}), likely categorical:")
            for val in unique_vals:
                print(f"  {val}")
        else:
            print(f"\nMany unique values ({len(unique_vals)}), likely continuous")
            
except Exception as e:
    print(f"Error reading {tiff_path}: {e}")
    import traceback
    traceback.print_exc()
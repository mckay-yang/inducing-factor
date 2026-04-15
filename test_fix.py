#!/usr/bin/env python
"""Quick test of the DEM loader fix."""
import warnings
import sys
sys.path.insert(0, '.')

from src.data_loader.dem_loader import DEMLoader
import rasterio
from rasterio.windows import Window

# Use a small window to avoid heavy I/O
dem_path = "/Volumes/T9/Datasets/dem/tibet/tibet_dem.tif"
print(f"Testing DEM: {dem_path}")

loader = DEMLoader(dem_path)
window = Window(1000, 1000, 100, 100)

print("Loading elevation...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    data, (res_x, res_y), nodata = loader.load(window)
    if w:
        print(f"Warnings during load: {[str(warning.message) for warning in w]}")
    else:
        print("No warnings during load.")

print(f"Resolution: res_x={res_x}, res_y={res_y}")
print(f"Data shape: {data.shape}, nodata={nodata}")

print("Calculating slope and aspect...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    slope, aspect = loader.calculate_slope_aspect(data, res_x, res_y, nodata=nodata)
    if w:
        print(f"Warnings during slope/aspect calculation:")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")
    else:
        print("No warnings during slope/aspect calculation.")

print(f"Slope min/max: {slope.min():.2f}, {slope.max():.2f}")
print(f"Aspect min/max: {aspect.min():.2f}, {aspect.max():.2f}")

# Check for NaN/inf
import numpy as np
print(f"Valid slope pixels: {np.isfinite(slope).sum()}/{slope.size}")
print(f"Valid aspect pixels: {np.isfinite(aspect).sum()}/{aspect.size}")

if res_x == 0 or res_y == 0:
    print("ERROR: Zero resolution detected!")
    sys.exit(1)
else:
    print("SUCCESS: Resolution looks correct.")
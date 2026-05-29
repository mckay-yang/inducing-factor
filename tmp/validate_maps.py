"""Validate probability maps: check value range, spatial distribution, and data quality."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CLASS_LABELS = {1: "fire", 2: "logging", 3: "geo_hazard", 4: "road_construction"}
_ABS_FILE_SHORT = {1: "fire", 2: "logging", 3: "geo_hazard", 4: "road"}

def validate_map(path, name, expected_range=(0, 1)):
    import rasterio
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return False
    
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        valid = data[data != nodata]
    
    if len(valid) == 0:
        print(f"  EMPTY: {name} - no valid pixels")
        return False
    
    lo, hi = expected_range
    out_of_range = ((valid < lo) | (valid > hi)).sum()
    pct_valid = len(valid) / data.size * 100
    
    print(f"  {name}:")
    print(f"    shape={data.shape}, valid_pixels={len(valid)} ({pct_valid:.1f}%), nodata={nodata}")
    print(f"    range=[{valid.min():.4f}, {valid.max():.4f}], mean={valid.mean():.4f}, std={valid.std():.4f}")
    print(f"    out_of_range={out_of_range} ({out_of_range/len(valid)*100:.2f}%)")
    
    # Spatial autocorrelation: neighbor correlation as proxy for patch-likeness
    if len(valid) > 100:
        mask = data != nodata
        from scipy.ndimage import uniform_filter
        smoothed = uniform_filter(np.where(mask, data, 0), size=3)
        valid_smooth = smoothed[mask]
        if len(valid_smooth) > 0:
            corr = np.corrcoef(valid, valid_smooth)[0, 1]
            print(f"    spatial_autocorr={corr:.4f} {'(patch-like)' if corr > 0.7 else '(scattered)'}")
    
    ok = out_of_range == 0 and valid.min() >= lo and valid.max() <= hi
    if not ok:
        print(f"    FAIL: values outside [{lo}, {hi}]")
    return ok


def main():
    import glob
    
    out_dirs = sorted(glob.glob("data/.cache/patent_system_aoi_v3_subset"))
    if not out_dirs:
        print("No output directory found")
        sys.exit(1)
    
    out_dir = out_dirs[-1]
    map_dir = os.path.join(out_dir, "probability_maps")
    
    if not os.path.exists(map_dir):
        print(f"No probability_maps dir yet: {map_dir}")
        print("Run predict-annual-map first")
        sys.exit(1)
    
    print(f"Validating maps in: {map_dir}\n")
    
    all_ok = True
    
    dist_maps = sorted(glob.glob(os.path.join(map_dir, "disturbance_probability_*.tif")))
    for path in dist_maps:
        year = os.path.basename(path).replace(".tif", "").split("_")[-1]
        ok = validate_map(path, f"disturbance_probability_{year}")
        all_ok = all_ok and ok
    
    print()
    for cls_id, cls_name in CLASS_LABELS.items():
        file_short = _ABS_FILE_SHORT.get(cls_id, cls_name)
        for path in sorted(glob.glob(os.path.join(map_dir, f"prob_{file_short}_*.tif"))):
            year = os.path.basename(path).replace(".tif", "").split("_")[-1]
            ok = validate_map(path, f"P({cls_name})_{year}")
            all_ok = all_ok and ok
    
    print()
    for cls_id in [1, 2, 3, 4]:
        for path in sorted(glob.glob(os.path.join(map_dir, f"class_probability_*_{cls_id}.tif"))):
            ok = validate_map(path, f"P(class={CLASS_LABELS[cls_id]}|disturbance)")
            all_ok = all_ok and ok
    
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

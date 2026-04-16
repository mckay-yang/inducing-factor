#!/usr/bin/env python3
import rasterio
import geopandas as gpd
import numpy as np
import os
import sys
from rasterio.mask import mask
from shapely.geometry import box

def main():
    # Paths
    aoi_path = "data/aoi/LangToBasighat.shp"
    resident_dir = "data/resident"
    
    # Check if files exist
    if not os.path.exists(aoi_path):
        print(f"AOI shapefile not found: {aoi_path}")
        # Try alternative spelling
        aoi_path = "data/aoi/LangToBasighat.shp"
        if not os.path.exists(aoi_path):
            print(f"Alternative AOI also not found: {aoi_path}")
            return
    
    print(f"Loading AOI from: {aoi_path}")
    try:
        aoi_gdf = gpd.read_file(aoi_path)
        print(f"AOI CRS: {aoi_gdf.crs}")
        print(f"AOI bounds: {aoi_gdf.total_bounds}")
        print(f"AOI geometry type: {aoi_gdf.geometry.type.unique()}")
    except Exception as e:
        print(f"Error loading AOI: {e}")
        return
    
    # Check resident files
    resident_files = [f for f in os.listdir(resident_dir) if f.endswith('.tif')]
    print(f"\nFound {len(resident_files)} resident TIFF files:")
    for f in sorted(resident_files):
        print(f"  {f}")
    
    # Test reading resident data for year 2015 (closest to our study period)
    resident_2015 = os.path.join(resident_dir, "GURS_2015.tif")
    if not os.path.exists(resident_2015):
        print(f"\nResident 2015 file not found: {resident_2015}")
        return
    
    print(f"\n--- Testing resident data for 2015 ---")
    with rasterio.open(resident_2015) as src:
        print(f"Source CRS: {src.crs}")
        print(f"Source bounds: {src.bounds}")
        print(f"Source shape: {src.shape}")
        print(f"Source dtype: {src.dtypes[0]}")
        print(f"Source NoData: {src.nodata}")
        
        # Reproject AOI to resident CRS if needed
        if aoi_gdf.crs != src.crs:
            print(f"Reprojecting AOI from {aoi_gdf.crs} to {src.crs}")
            aoi_gdf = aoi_gdf.to_crs(src.crs)
        
        # Get combined geometry (union of all features)
        aoi_geom = aoi_gdf.geometry.unary_union
        print(f"AOI geometry after union: {aoi_geom.geom_type}")
        
        # Try to mask and read data within AOI
        try:
            out_image, out_transform = mask(src, [aoi_geom], crop=True, filled=False)
            data = out_image[0]  # First band
            print(f"\nMasked data shape: {data.shape}")
            print(f"Unique values in masked area: {np.unique(data)}")
            
            # Count values
            unique, counts = np.unique(data, return_counts=True)
            total_pixels = data.size
            for val, cnt in zip(unique, counts):
                if np.isnan(val):
                    print(f"  NaN: {cnt}/{total_pixels} ({cnt/total_pixels*100:.2f}%)")
                else:
                    print(f"  Value {val}: {cnt}/{total_pixels} ({cnt/total_pixels*100:.2f}%)")
            
            # Check specifically for values 1 and 2
            val1 = np.sum(data == 1)
            val2 = np.sum(data == 2)
            print(f"\nValue 1 (urban): {val1} pixels")
            print(f"Value 2 (rural): {val2} pixels")
            
        except Exception as e:
            print(f"Error during masking: {e}")
            # Try alternative: read a sample window
            print("\nTrying sample window read instead...")
            window = rasterio.windows.Window(0, 0, min(1000, src.width), min(1000, src.height))
            data = src.read(1, window=window)
            print(f"Sample window shape: {data.shape}")
            print(f"Sample unique values: {np.unique(data)}")
    
    # Also check chunk coordinates from cache
    print("\n--- Checking chunk coordinates ---")
    cache_dir = "results/cache_real"
    if os.path.exists(cache_dir):
        import glob
        cache_files = glob.glob(os.path.join(cache_dir, "chunk_*.npz"))
        if cache_files:
            import numpy as np
            data = np.load(cache_files[0])
            if 'x' in data and 'y' in data:
                x_coords = data['x']
                y_coords = data['y']
                print(f"Chunk {os.path.basename(cache_files[0])}: {len(x_coords)} points")
                print(f"  X range: {x_coords.min():.6f} to {x_coords.max():.6f}")
                print(f"  Y range: {y_coords.min():.6f} to {y_coords.max():.6f}")
                
                # Check if chunk bounds overlap with AOI bounds
                chunk_bounds = box(x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())
                if aoi_geom.intersects(chunk_bounds):
                    print("  Chunk overlaps with AOI")
                else:
                    print("  WARNING: Chunk does NOT overlap with AOI")
            else:
                print("No x,y coordinates in cache file")

if __name__ == "__main__":
    main()
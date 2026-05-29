#!/usr/bin/env python3
"""
Process a single chunk (first chunk) of real data for testing.
"""
import os
import sys
import logging
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd

# Add parent directory to path to import inducing_factor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inducing_factor.data.loaders import CCDCLoader, ClimateLoader, DEMLoader, LandUseLoader, VectorLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SingleChunkProcessor")

def decimal_year_to_month(values, years):
    """Convert decimal year to month (1-12)."""
    months = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return months
    value_years = np.floor(values[valid]).astype(int)
    matching = value_years == years[valid].astype(int)
    if not np.any(matching):
        return months
    matched_values = values[valid][matching]
    matched_years = value_years[matching]
    fractions = np.clip(matched_values - matched_years, 0.0, 0.999999)
    # Simple approximation: fraction * 12 + 1 (month)
    decoded_months = (fractions * 12 + 1).astype(int)
    months[np.where(valid)[0][matching]] = np.clip(decoded_months, 1, 12)
    return months

def main():
    chunk_size = 1024
    start_year = 2015
    end_year = 2021
    cache_dir = "results/cache_real"
    os.makedirs(cache_dir, exist_ok=True)
    
    dem_path = "data/dem/tibet_dem.tif"
    road_path = "data/road/osm_2014_roads.shp"
    river_path = "data/river/As_15s_Chimaps.shp"
    rainfall_dir = "data/rainfall"
    temp_dir = "data/temperature"
    landuse_dir = "data/landuse"
    resident_path = "data/resident"
    ccdc_dir = "data/ccdc_res"
    
    logger.info("Initializing loaders...")
    dem_loader = DEMLoader(dem_path)
    road_loader = VectorLoader(road_path)
    river_loader = VectorLoader(river_path)
    rainfall_loader = ClimateLoader(rainfall_dir)
    temp_loader = ClimateLoader(temp_dir, file_pattern="tmp_{year}.nc")
    landuse_loader = LandUseLoader(landuse_dir, is_temporal=True, file_pattern="CLCD_v01_{year}_albert_province/CLCD_v01_{year}_albert_xizang.tif", data_type="Land use")
    resident_loader = VectorLoader(resident_path)
    ccdc_loader = CCDCLoader(ccdc_dir, file_pattern="{year}.tif")
    
    logger.info("Loading vector data...")
    road_loader.load()
    river_loader.load()
    resident_loader.load()
    
    with rasterio.open(dem_path) as src_dem:
        width = src_dem.width
        height = src_dem.height
        transform = src_dem.transform
        crs = src_dem.crs
        logger.info("DEM dimensions: %d x %d, CRS: %s", width, height, crs)
        
        # Process only first chunk at (0,0)
        row_off = 0
        col_off = 0
        w = min(chunk_size, width - col_off)
        h = min(chunk_size, height - row_off)
        window = Window(col_off, row_off, w, h)
        logger.info("Processing window: col_off=%d, row_off=%d, width=%d, height=%d", col_off, row_off, w, h)
        
        dem_data, (res_x, res_y), dem_nodata = dem_loader.load(window)
        if np.all(dem_data == dem_nodata) or np.all(np.isnan(dem_data)):
            logger.info("No valid data in window")
            return
        
        slope_data, aspect_data = dem_loader.calculate_slope_aspect(dem_data, res_x, res_y, nodata=dem_nodata)
        valid_mask = (dem_data != dem_nodata) & (~np.isnan(dem_data))
        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            logger.info("No valid pixels")
            return
        
        x_coords, y_coords = rasterio.transform.xy(transform, rows + row_off, cols + col_off)
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        
        logger.info("Calculating distances...")
        dist_road = road_loader.get_distance(x_coords, y_coords, crs)
        dist_river = river_loader.get_distance(x_coords, y_coords, crs)
        dist_resident = resident_loader.get_distance(x_coords, y_coords, crs)
        
        point_ids = np.arange(len(rows))
        df_static = pd.DataFrame({
            "point_id": point_ids,
            "x": x_coords,
            "y": y_coords,
            "elevation": dem_data[rows, cols],
            "slope": slope_data[rows, cols],
            "aspect": aspect_data[rows, cols],
            "dist_to_road": dist_road,
            "dist_to_river": dist_river,
            "dist_to_resident": dist_resident,
        })
        
        rainfall_loader.clear_state()
        temp_loader.clear_state()
        landuse_loader.clear_state()
        
        chunk_records = []
        for year in range(start_year, end_year + 1):
            logger.info("Processing year %d", year)
            
            rainfall_data = rainfall_loader.load(year, crs, transform, width, height, window)
            temp_data = temp_loader.load(year, crs, transform, width, height, window)
            landuse_data = landuse_loader.load(year, crs, transform, width, height, window)
            
            # Extract values at valid pixels
            rainfall_vals = rainfall_data[rows, cols] if rainfall_data is not None else np.full(len(rows), np.nan)
            temp_vals = temp_data[rows, cols] if temp_data is not None else np.full(len(rows), np.nan)
            landuse_vals = landuse_data[rows, cols] if landuse_data is not None else np.full(len(rows), np.nan)
            
            # Load monthly rainfall (12 months)
            monthly_rainfall = {}
            for month in range(1, 13):
                month_data = rainfall_loader.load_month(year, month, crs, transform, width, height, window)
                if month_data is not None:
                    monthly_rainfall[f"rainfall_month_{month:02d}"] = month_data[rows, cols]
                else:
                    monthly_rainfall[f"rainfall_month_{month:02d}"] = np.full(len(rows), np.nan)
            
            df_year = df_static.copy()
            df_year["year"] = year
            df_year["rainfall"] = rainfall_vals
            df_year["temperature"] = temp_vals
            df_year["landuse"] = landuse_vals
            for month_col, vals in monthly_rainfall.items():
                df_year[month_col] = vals
            
            # Disturbance data
            disturbance_data = ccdc_loader.load(year, crs, transform, width, height, window)
            if disturbance_data is not None:
                disturbance_vals = disturbance_data[rows, cols]
                disturbance_mask = ~np.isnan(disturbance_vals)
                disturbance_years = np.floor(np.where(disturbance_mask, disturbance_vals, np.nan)).astype(float)
                year_match = disturbance_years == float(year)
                disturbance_vals = np.where(disturbance_mask & year_match, disturbance_vals, 0.0)
                df_year["disturbance_type"] = disturbance_vals
                df_year["disturbance_month"] = decimal_year_to_month(disturbance_vals, df_year["year"].to_numpy())
            else:
                df_year["disturbance_type"] = 0.0
                df_year["disturbance_month"] = np.nan
            
            chunk_records.append(df_year)
        
        if chunk_records:
            df_chunk = pd.concat(chunk_records, ignore_index=True)
            df_chunk["point_id"] = pd.to_numeric(df_chunk["point_id"], downcast="integer")
            df_chunk["year"] = pd.to_numeric(df_chunk["year"], downcast="integer")
            cache_file = os.path.join(cache_dir, "chunk_0.npz")
            np.savez_compressed(cache_file, **{col: df_chunk[col].values for col in df_chunk.columns})
            logger.info("Saved chunk data to %s, shape: %s", cache_file, df_chunk.shape)
            logger.info("Columns: %s", df_chunk.columns.tolist())
        else:
            logger.info("No records saved")

if __name__ == "__main__":
    main()
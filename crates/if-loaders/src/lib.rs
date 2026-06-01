//! Data loaders for geospatial raster and vector formats:
//! DEM (GeoTIFF), CCDC disturbance, climate NetCDF, land-use rasters,
//! and OSM vector data (roads, railways, waterways).
//!
//! ## Features
//!
//! - `gdal`: enables all raster and vector loaders (DEM, landuse, CCDC,
//!   climate, resident, disturbance-class, vector).
//! - `netcdf`: enables temperature NetCDF loading in [`ClimateLoader`].

pub mod base;

#[cfg(feature = "gdal")]
pub mod ccdc;
#[cfg(feature = "gdal")]
pub mod climate;
#[cfg(feature = "gdal")]
pub mod dem;
#[cfg(feature = "gdal")]
pub mod disturbance_class;
#[cfg(feature = "gdal")]
pub mod landuse;
#[cfg(feature = "gdal")]
pub mod resident;
#[cfg(feature = "gdal")]
pub mod vector;

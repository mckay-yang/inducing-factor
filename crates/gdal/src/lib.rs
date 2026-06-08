//! Geospatial primitives and GDAL boundary code.
//!
//! Higher-level crates should depend on this crate for raster/grid metadata and
//! coordinate-safe planning.  The goal is to keep direct GDAL usage contained
//! here instead of spreading it through data processing, training, prediction,
//! and CLI code.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum GeoError {
    #[error("GDAL error: {0}")]
    Gdal(String),
    #[error("invalid raster grid: {0}")]
    InvalidGrid(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct GeoBounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeoGrid {
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    pub geo_transform: [f64; 6],
    pub crs_wkt: String,
    pub band_count: usize,
    pub start_year: i32,
}

impl GeoGrid {
    #[cfg(feature = "gdal")]
    pub fn open(path: impl AsRef<Path>, start_year: i32) -> Result<Self, GeoError> {
        let path = path.as_ref();
        let ds = gdal_crate::Dataset::open(path)
            .map_err(|e| GeoError::Gdal(format!("open {}: {e}", path.display())))?;
        let (width, height) = ds.raster_size();
        let geo_transform = ds
            .geo_transform()
            .map_err(|e| GeoError::Gdal(format!("geo_transform {}: {e}", path.display())))?;
        let crs_wkt = ds.projection();
        let band_count = ds.raster_count();

        let grid = Self {
            path: path.to_path_buf(),
            width,
            height,
            geo_transform,
            crs_wkt,
            band_count,
            start_year,
        };
        grid.validate()?;
        Ok(grid)
    }

    #[cfg(not(feature = "gdal"))]
    pub fn open(_path: impl AsRef<Path>, _start_year: i32) -> Result<Self, GeoError> {
        Err(GeoError::Gdal(
            "GeoGrid::open requires the 'gdal' feature".to_string(),
        ))
    }

    pub fn validate(&self) -> Result<(), GeoError> {
        if self.width == 0 || self.height == 0 {
            return Err(GeoError::InvalidGrid(format!(
                "empty dimensions {}x{}",
                self.width, self.height
            )));
        }
        if self.band_count == 0 {
            return Err(GeoError::InvalidGrid("band_count must be >= 1".into()));
        }
        if self.geo_transform[1] == 0.0 || self.geo_transform[5] == 0.0 {
            return Err(GeoError::InvalidGrid(
                "pixel size in geotransform must be non-zero".into(),
            ));
        }
        Ok(())
    }

    pub fn end_year(&self) -> i32 {
        self.start_year + self.band_count as i32 - 1
    }

    pub fn years(&self) -> Vec<i32> {
        (self.start_year..=self.end_year()).collect()
    }

    pub fn pixel_center_to_geo(&self, row: usize, col: usize) -> (f64, f64) {
        let gt = &self.geo_transform;
        let x = gt[0] + (col as f64 + 0.5) * gt[1] + (row as f64 + 0.5) * gt[2];
        let y = gt[3] + (col as f64 + 0.5) * gt[4] + (row as f64 + 0.5) * gt[5];
        (x, y)
    }

    /// Bounds in the grid CRS, derived from pixel-center corner coordinates.
    pub fn bounds(&self) -> GeoBounds {
        let (x0, y0) = self.pixel_center_to_geo(0, 0);
        let (x1, y1) = self.pixel_center_to_geo(self.height - 1, self.width - 1);
        GeoBounds {
            min_x: x0.min(x1),
            min_y: y0.min(y1),
            max_x: x0.max(x1),
            max_y: y0.max(y1),
        }
    }

    pub fn tile_windows(&self, tile_size: usize) -> Vec<GridWindow> {
        assert!(tile_size > 0, "tile_size must be > 0");
        let mut out = Vec::new();
        for row_off in (0..self.height).step_by(tile_size) {
            for col_off in (0..self.width).step_by(tile_size) {
                out.push(GridWindow {
                    row_off,
                    col_off,
                    width: tile_size.min(self.width - col_off),
                    height: tile_size.min(self.height - row_off),
                });
            }
        }
        out
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct GridWindow {
    pub row_off: usize,
    pub col_off: usize,
    pub width: usize,
    pub height: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid() -> GeoGrid {
        GeoGrid {
            path: PathBuf::from("class.tif"),
            width: 10,
            height: 8,
            geo_transform: [100.0, 0.01, 0.0, 30.0, 0.0, -0.01],
            crs_wkt: "EPSG:4326".into(),
            band_count: 10,
            start_year: 2016,
        }
    }

    #[test]
    fn years_follow_band_count() {
        let grid = test_grid();
        assert_eq!(grid.end_year(), 2025);
        assert_eq!(grid.years()[0], 2016);
    }

    #[test]
    fn tile_windows_cover_edges() {
        let grid = test_grid();
        let tiles = grid.tile_windows(6);
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[3].width, 4);
        assert_eq!(tiles[3].height, 2);
    }
}

//! Core traits and types shared across all data loaders.
//!
//! Defines the `RasterWindowLoader` trait for windowed raster reads,
//! `Window` for specifying spatial sub-regions, and unified error types.

use ndarray::Array2;
use thiserror::Error;

/// Errors that can occur during data loading operations.
#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GDAL error: {0}")]
    Gdal(String),

    #[error("NetCDF error: {0}")]
    NetCdf(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("No valid data in window")]
    NoValidData,

    #[error("Invalid window: {0}")]
    InvalidWindow(String),

    #[error("Missing feature: {0}")]
    MissingFeature(String),

    #[error("CRS error: {0}")]
    CrsError(String),

    #[error("Vector read error: {0}")]
    VectorRead(String),
}

#[cfg(feature = "gdal")]
impl From<gdal::errors::GdalError> for LoaderError {
    fn from(e: gdal::errors::GdalError) -> Self {
        LoaderError::Gdal(e.to_string())
    }
}

/// Result type alias for loader operations.
pub type LoaderResult<T> = Result<T, LoaderError>;

/// Spatial window specification for raster subset reads.
///
/// Coordinates are pixel offsets and sizes in the source raster grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Column offset (x origin) in pixels.
    pub col_off: usize,
    /// Row offset (y origin) in pixels.
    pub row_off: usize,
    /// Window width in pixels.
    pub width: usize,
    /// Window height in pixels.
    pub height: usize,
}

impl Window {
    /// Create a new window specification.
    pub fn new(col_off: usize, row_off: usize, width: usize, height: usize) -> Self {
        Self {
            col_off,
            row_off,
            width,
            height,
        }
    }

    /// Total number of pixels in this window.
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Returns true if this window has zero area.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Trait for raster data loaders that support windowed reads.
///
/// Each implementation wraps a specific data source (DEM, climate,
/// land-use, CCDC, etc.) and provides spatial-window access via
/// `load_window`.  The returned `Array2<f32>` is row-major with
/// shape `(height, width)`.
pub trait RasterWindowLoader {
    /// Read a spatial window from the raster, returning pixel values
    /// as a 2-D array.  NoData pixels are converted to `f32::NAN`.
    fn load_window(&self, window: &Window) -> LoaderResult<Array2<f32>>;

    /// Return the NoData sentinel value for this raster, if any.
    fn nodata_value(&self) -> Option<f32>;
}

/// Trait for vector data loaders that compute point-to-feature distances.
pub trait VectorDistanceLoader {
    /// Compute the shortest geometric distance from each (x, y) point
    /// to any vector feature (road, river, etc.).
    ///
    /// Distances are in the same units as the input coordinates
    /// (typically meters after reprojection).  There is **no maximum
    /// distance cap** — a point arbitrarily far from all features
    /// still receives a finite, correct distance.
    fn get_distance(&self, x_coords: &[f64], y_coords: &[f64]) -> LoaderResult<Vec<f64>>;
}

/// Helper: replace NoData values in an ndarray with NaN.
pub fn apply_nodata_mask(arr: &mut Array2<f32>, nodata: f32) {
    arr.mapv_inplace(|v| {
        if (v - nodata).abs() < f32::EPSILON || v.is_nan() {
            f32::NAN
        } else {
            v
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_pixel_count() {
        let w = Window::new(0, 0, 100, 50);
        assert_eq!(w.pixel_count(), 5000);
    }

    #[test]
    fn test_window_is_empty() {
        assert!(Window::new(0, 0, 0, 10).is_empty());
        assert!(Window::new(0, 0, 10, 0).is_empty());
        assert!(!Window::new(0, 0, 10, 10).is_empty());
    }

    #[test]
    fn test_apply_nodata_mask() {
        let mut arr = Array2::from_shape_vec((2, 2), vec![1.0, -9999.0, 3.0, f32::NAN]).unwrap();
        apply_nodata_mask(&mut arr, -9999.0);
        assert!(arr[[0, 0]].is_finite());
        assert!(arr[[0, 1]].is_nan());
        assert!(arr[[1, 0]].is_finite());
        assert!(arr[[1, 1]].is_nan());
    }
}

//! ClassGrid — single reference grid for resampling and prediction.
//!
//! Wraps the multi-band disturbance classification raster
//! (`data/disturbance_class/disturbance_class_res.tif`) which provides the
//! canonical grid for all output rasters.  The raster is 62166×35095 pixels
//! in EPSG:4326 with 10 bands (class_2016 … class_2025) stored as Int8.

#[cfg(feature = "gdal")]
use gdal_crate::Dataset;

use loaders::base::{LoaderError, LoaderResult};
use ndarray::Array2;

/// Canonical reference grid derived from the disturbance classification raster.
///
/// Provides dimensions, geotransform, CRS, pixel↔geographic conversion, and
/// per‑year band access.
#[derive(Debug, Clone)]
pub struct ClassGrid {
    /// Raster width (columns).
    pub width: usize,
    /// Raster height (rows).
    pub height: usize,
    /// GDAL affine geotransform `[origin_x, px_w, row_rot, origin_y, col_rot, px_h]`.
    pub geo_transform: [f64; 6],
    /// CRS as WKT string.
    pub crs: String,
    /// Number of bands (years) in the raster.
    pub band_count: usize,
    /// First year represented by band 1.
    pub start_year: i32,
    /// File-system path to the raster.
    path: String,
}

impl ClassGrid {
    #[cfg(test)]
    pub(crate) fn synthetic(
        width: usize,
        height: usize,
        geo_transform: [f64; 6],
        crs: String,
        band_count: usize,
        start_year: i32,
    ) -> Self {
        Self {
            width,
            height,
            geo_transform,
            crs,
            band_count,
            start_year,
            path: "synthetic.tif".into(),
        }
    }

    /// Open the CLASS raster and extract metadata.
    ///
    /// The file must be a valid GDAL dataset with at least one band.
    /// Band 1 is assumed to correspond to `start_year` (2016 for the
    /// canonical CLASS raster `class_2016 … class_2025`).
    #[cfg(feature = "gdal")]
    pub fn open(path: &str, start_year: i32) -> LoaderResult<Self> {
        let ds = Dataset::open(path).map_err(|e| LoaderError::Gdal(format!("open {path}: {e}")))?;
        let (width, height) = ds.raster_size();
        let geo_transform = ds
            .geo_transform()
            .map_err(|e| LoaderError::Gdal(format!("geo_transform {path}: {e}")))?;
        let crs = ds.projection();
        let band_count = ds.raster_count() as usize;
        drop(ds);

        if width == 0 || height == 0 || band_count == 0 {
            return Err(LoaderError::InvalidWindow(format!(
                "CLASS raster {path} is empty (dims {width}×{height}, {band_count} bands)"
            )));
        }

        Ok(Self {
            width: width as usize,
            height: height as usize,
            geo_transform,
            crs,
            band_count,
            start_year,
            path: path.to_string(),
        })
    }

    /// Stub implementation when `gdal` feature is disabled.
    #[cfg(not(feature = "gdal"))]
    pub fn open(_path: &str, _start_year: i32) -> LoaderResult<Self> {
        Err(LoaderError::Gdal(
            "ClassGrid::open requires the 'gdal' feature".into(),
        ))
    }

    // ── accessors ──────────────────────────────────────────────────────

    /// Raster width (columns).
    pub fn width(&self) -> usize {
        self.width
    }
    /// Raster height (rows).
    pub fn height(&self) -> usize {
        self.height
    }
    /// GDAL affine geotransform (6-element array).
    pub fn geo_transform(&self) -> &[f64; 6] {
        &self.geo_transform
    }
    /// CRS as WKT string.
    pub fn crs(&self) -> &str {
        &self.crs
    }
    /// Total number of bands.
    pub fn band_count(&self) -> usize {
        self.band_count
    }
    /// First year (band 1).
    pub fn start_year(&self) -> i32 {
        self.start_year
    }
    /// Last year (last band).
    pub fn end_year(&self) -> i32 {
        self.start_year + self.band_count as i32 - 1
    }

    /// Years available in this raster.
    pub fn years(&self) -> Vec<i32> {
        (self.start_year..=self.end_year()).collect()
    }

    /// Map a year to a 1‑based band index, or `None` if out of range.
    fn year_to_band(&self, year: i32) -> Option<usize> {
        if year < self.start_year {
            return None;
        }
        let band = (year - self.start_year) as usize + 1;
        if band > self.band_count {
            return None;
        }
        Some(band)
    }

    // ── pixel / geo conversions ────────────────────────────────────────

    /// Convert pixel coordinates (row, col) to geographic (longitude, latitude).
    ///
    /// Uses the pixel‑center convention: `x = gt[0] + (col+0.5)*gt[1] + (row+0.5)*gt[2]`.
    pub fn pixel_to_geo(&self, row: usize, col: usize) -> (f64, f64) {
        let gt = &self.geo_transform;
        let lng = gt[0] + (col as f64 + 0.5) * gt[1] + (row as f64 + 0.5) * gt[2];
        let lat = gt[3] + (col as f64 + 0.5) * gt[4] + (row as f64 + 0.5) * gt[5];
        (lng, lat)
    }

    /// Convert geographic coordinates to pixel indices, clamped to valid range.
    ///
    /// Returns `(row, col)`.  Out‑of‑bounds coordinates are clamped to
    /// `[0, height)` and `[0, width)`.
    pub fn geo_to_pixel(&self, lng: f64, lat: f64) -> (usize, usize) {
        let gt = &self.geo_transform;
        // Full affine inverse (handles rotation when gt[2] or gt[4] ≠ 0).
        let det = gt[1] * gt[5] - gt[2] * gt[4];
        let col_f = (gt[5] * (lng - gt[0]) - gt[2] * (lat - gt[3])) / det - 0.5;
        let row_f = (gt[1] * (lat - gt[3]) - gt[4] * (lng - gt[0])) / det - 0.5;

        let col = col_f.round() as isize;
        let row = row_f.round() as isize;

        (
            row.clamp(0, self.height as isize - 1) as usize,
            col.clamp(0, self.width as isize - 1) as usize,
        )
    }

    // ── bounding box ───────────────────────────────────────────────────

    /// Return `(min_lng, min_lat, max_lng, max_lat)` computed from geotransform
    /// and raster dimensions.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let (lng_top_left, lat_top_left) = self.pixel_to_geo(0, 0);
        let (lng_bot_right, lat_bot_right) = self.pixel_to_geo(self.height - 1, self.width - 1);

        (
            lng_top_left.min(lng_bot_right),
            lat_top_left.min(lat_bot_right),
            lng_top_left.max(lng_bot_right),
            lat_top_left.max(lat_bot_right),
        )
    }

    // ── data access ────────────────────────────────────────────────────

    /// Read an entire band for *year* as an `Array2<i8>`.
    ///
    /// Returns an error if *year* is not covered by the raster.
    #[cfg(feature = "gdal")]
    pub fn read_band(&self, year: i32) -> LoaderResult<Array2<i8>> {
        let band = self.year_to_band(year).ok_or_else(|| {
            LoaderError::MissingFeature(format!(
                "year {year} outside [{}, {}]",
                self.start_year,
                self.end_year()
            ))
        })?;

        let ds = Dataset::open(&self.path)
            .map_err(|e| LoaderError::Gdal(format!("open {}: {e}", self.path)))?;
        let raster_band = ds
            .rasterband(band)
            .map_err(|e| LoaderError::Gdal(format!("rasterband {band}: {e}")))?;

        let data = raster_band
            .read_as::<i8>(
                (0, 0),
                (self.width, self.height),
                (self.width, self.height),
                None,
            )
            .map_err(|e| LoaderError::Gdal(e.to_string()))?;

        Array2::from_shape_vec((self.height, self.width), data.data().to_vec())
            .map_err(|e| LoaderError::Gdal(format!("shape mismatch: {e}")))
    }

    /// Stub implementation when `gdal` feature is disabled.
    #[cfg(not(feature = "gdal"))]
    pub fn read_band(&self, _year: i32) -> LoaderResult<Array2<i8>> {
        Err(LoaderError::Gdal(
            "ClassGrid::read_band requires the 'gdal' feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn class_path() -> String {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
            .join("disturbance_class")
            .join("disturbance_class_res.tif");
        p.to_string_lossy().to_string()
    }

    #[cfg(feature = "gdal")]
    #[test]
    fn test_open() {
        let path = class_path();
        let cg = ClassGrid::open(&path, 2016).expect("failed to open CLASS raster");
        assert_eq!(cg.width(), 62166, "width");
        assert_eq!(cg.height(), 35095, "height");
        assert_eq!(cg.band_count(), 10, "band count");
        assert_eq!(cg.start_year(), 2016);
        assert_eq!(cg.end_year(), 2025);
        assert!(!cg.crs().is_empty(), "CRS should be non-empty");
        let gt = cg.geo_transform();
        // Pixel width should be positive (EPSG:4326 eastward).
        assert!(
            gt[1] > 0.0,
            "pixel width should be positive (got {})",
            gt[1]
        );
        // Pixel height should be negative (EPSG:4326 southward).
        assert!(
            gt[5] < 0.0,
            "pixel height should be negative (got {})",
            gt[5]
        );
    }

    #[cfg(feature = "gdal")]
    #[test]
    fn test_pixel_geo_roundtrip() {
        let cg = ClassGrid::open(&class_path(), 2016).unwrap();
        // Centre of raster.
        let row = 17000;
        let col = 30000;
        let (lng, lat) = cg.pixel_to_geo(row, col);
        let (r2, c2) = cg.geo_to_pixel(lng, lat);
        let dr = (row as isize - r2 as isize).abs();
        let dc = (col as isize - c2 as isize).abs();
        assert!(dr <= 1, "row drift {dr}");
        assert!(dc <= 1, "col drift {dc}");
    }

    #[cfg(feature = "gdal")]
    #[test]
    fn test_read_band_2020() {
        let cg = ClassGrid::open(&class_path(), 2016).unwrap();
        let arr = cg.read_band(2020).expect("failed to read band 2020");
        assert_eq!(arr.shape(), &[35095, 62166], "band 2020 shape");
    }

    #[cfg(feature = "gdal")]
    #[test]
    fn test_invalid_year() {
        let cg = ClassGrid::open(&class_path(), 2016).unwrap();
        let err = cg.read_band(1999).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("1999"),
            "error should mention the invalid year: {msg}"
        );
    }

    /// Ensure the non‑gdal stubs return errors, not panics.
    #[cfg(not(feature = "gdal"))]
    #[test]
    fn test_stubs_return_errors() {
        let err = ClassGrid::open("dummy.tif", 2016).unwrap_err();
        assert!(err.to_string().contains("gdal"));
    }
}

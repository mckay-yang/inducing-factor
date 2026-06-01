use gdal::Dataset;
use ndarray::Array2;
use std::path::Path;

use crate::base::{apply_nodata_mask, LoaderError, LoaderResult, Window};

const RESIDENT_NODATA: f32 = 127.0;

pub struct ResidentLoader {
    path: String,
}

impl ResidentLoader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }

    pub fn load(&self, window: &Window) -> LoaderResult<Array2<f32>> {
        let ds = Dataset::open(&self.path)?;
        let band = ds.rasterband(1)?;

        let data = band
            .read_as::<f32>(
                (window.col_off as isize, window.row_off as isize),
                (window.width, window.height),
                (window.width, window.height),
                None,
            )
            .map_err(|e| LoaderError::Gdal(e.to_string()))?;

        let mut arr = Array2::from_shape_vec((window.height, window.width), data.data)
            .map_err(|e| LoaderError::Gdal(format!("shape: {e}")))?;

        apply_nodata_mask(&mut arr, RESIDENT_NODATA);
        Ok(arr)
    }

    pub fn nodata_value(&self) -> f32 {
        RESIDENT_NODATA
    }

    pub fn is_available(&self) -> bool {
        Path::new(&self.path).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::DriverManager;
    use tempfile::TempDir;

    fn create_synthetic_resident(path: &str, data: &[f32], w: usize, h: usize) {
        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let ds = driver.create(path, w as isize, h as isize, 1).unwrap();
        let mut band = ds.rasterband(1).unwrap();
        let buf = gdal::raster::Buffer::new((w, h), data.to_vec());
        band.write((0, 0), (w, h), &buf).unwrap();
        drop(band);
        drop(ds);
    }

    #[test]
    fn test_resident_nodata() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("resident.tif");
        let path_str = path.to_str().unwrap().to_string();

        let data: Vec<f32> = vec![1.0, 127.0, 50.0, 127.0];
        create_synthetic_resident(&path_str, &data, 2, 2);

        let loader = ResidentLoader::new(&path_str);
        let w = Window::new(0, 0, 2, 2);
        let arr = loader.load(&w).unwrap();

        assert_eq!(arr[[0, 0]], 1.0);
        assert!(arr[[0, 1]].is_nan());
        assert_eq!(arr[[1, 0]], 50.0);
        assert!(arr[[1, 1]].is_nan());
    }

    #[test]
    fn test_resident_nodata_value() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("resident.tif");
        create_synthetic_resident(path.to_str().unwrap(), &[1.0], 1, 1);

        let loader = ResidentLoader::new(path.to_str().unwrap());
        assert_eq!(loader.nodata_value(), 127.0);
    }

    #[test]
    fn test_resident_is_available() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("exists.tif");
        create_synthetic_resident(path.to_str().unwrap(), &[1.0], 1, 1);

        let loader = ResidentLoader::new(path.to_str().unwrap());
        assert!(loader.is_available());

        let missing = ResidentLoader::new("/nonexistent/path.tif");
        assert!(!missing.is_available());
    }
}

use gdal::Dataset;
use ndarray::Array2;
use std::path::Path;

use crate::base::{LoaderError, LoaderResult, Window};

/// Loader for Land Use / Land Cover categorical raster data.
pub struct LandUseLoader {
    path: String,
    is_temporal: bool,
    file_pattern: String,
}

impl LandUseLoader {
    pub fn new_static(path: &str) -> Self {
        Self {
            path: path.to_string(),
            is_temporal: false,
            file_pattern: String::new(),
        }
    }

    pub fn new_temporal(dir: &str, file_pattern: &str) -> Self {
        Self {
            path: dir.to_string(),
            is_temporal: true,
            file_pattern: file_pattern.to_string(),
        }
    }

    fn target_path(&self, year: Option<i32>) -> String {
        if !self.is_temporal {
            return self.path.clone();
        }
        let y = year.expect("year required for temporal land-use");
        let fname = self.file_pattern.replace("{year}", &y.to_string());
        Path::new(&self.path)
            .join(fname)
            .to_str()
            .unwrap_or("")
            .to_string()
    }

    /// Load land-use categorical values for a given year and window.
    /// Returns `None` if the file is missing.
    pub fn load(&self, year: Option<i32>, window: &Window) -> LoaderResult<Option<Array2<f32>>> {
        let path = self.target_path(year);
        if !Path::new(&path).exists() {
            return Ok(None);
        }

        let ds = Dataset::open(&path)?;
        let band = ds.rasterband(1)?;

        let data = band
            .read_as::<f32>(
                (window.col_off as isize, window.row_off as isize),
                (window.width, window.height),
                (window.width, window.height),
                None,
            )
            .map_err(|e| LoaderError::Gdal(e.to_string()))?;

        let mut arr = Array2::from_shape_vec((window.height, window.width), data.data().to_vec())
            .map_err(|e| LoaderError::Gdal(format!("shape: {e}")))?;

        if let Some(nd) = band.no_data_value() {
            let nd_f32 = nd as f32;
            arr.mapv_inplace(|v| {
                if (v - nd_f32).abs() < f32::EPSILON {
                    f32::NAN
                } else {
                    v
                }
            });
        }

        Ok(Some(arr))
    }

    /// Load windowed land-use data (static mode).
    pub fn load_window_static(&self, window: &Window) -> LoaderResult<Array2<f32>> {
        match self.load(None, window)? {
            Some(arr) => Ok(arr),
            None => Err(LoaderError::FileNotFound(self.path.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::DriverManager;
    use tempfile::TempDir;

    fn create_synthetic_landuse(path: &str, data: &[f32], w: usize, h: usize) {
        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let ds = driver.create(path, w, h, 1).unwrap();
        let mut band = ds.rasterband(1).unwrap();
        let mut buf = gdal::raster::Buffer::new((w, h), data.to_vec());
        band.write((0, 0), (w, h), &mut buf).unwrap();
        drop(band);
        drop(ds);
    }

    #[test]
    fn test_landuse_categorical() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("landuse.tif");
        let path_str = path.to_str().unwrap().to_string();

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        create_synthetic_landuse(&path_str, &data, 2, 2);

        let loader = LandUseLoader::new_static(&path_str);
        let w = Window::new(0, 0, 2, 2);
        let arr = loader.load(None, &w).unwrap().unwrap();

        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[0, 1]], 2.0);
        assert_eq!(arr[[1, 0]], 3.0);
        assert_eq!(arr[[1, 1]], 4.0);
    }

    #[test]
    fn test_landuse_temporal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("2020.tif");
        create_synthetic_landuse(path.to_str().unwrap(), &[10.0, 20.0, 30.0, 40.0], 2, 2);

        let loader = LandUseLoader::new_temporal(dir.path().to_str().unwrap(), "{year}.tif");
        let w = Window::new(0, 0, 2, 2);
        let arr = loader.load(Some(2020), &w).unwrap().unwrap();

        assert_eq!(arr[[0, 0]], 10.0);
        assert_eq!(arr[[1, 1]], 40.0);
    }

    #[test]
    fn test_landuse_missing_year() {
        let dir = TempDir::new().unwrap();
        let loader = LandUseLoader::new_temporal(dir.path().to_str().unwrap(), "{year}.tif");
        let w = Window::new(0, 0, 1, 1);
        let result = loader.load(Some(2025), &w).unwrap();
        assert!(result.is_none());
    }
}

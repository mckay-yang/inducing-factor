use gdal::Dataset;
use ndarray::Array2;
use std::collections::BTreeSet;
use std::path::Path;

use crate::base::{LoaderError, LoaderResult, Window};

pub struct CCDCLoader {
    dir_path: String,
    file_pattern: String,
}

impl CCDCLoader {
    pub fn new(dir_path: &str, file_pattern: &str) -> Self {
        Self {
            dir_path: dir_path.to_string(),
            file_pattern: file_pattern.to_string(),
        }
    }

    pub fn new_default(dir_path: &str) -> Self {
        Self::new(dir_path, "{year}.tif")
    }

    fn build_path(&self, year: i32) -> String {
        let fname = self.file_pattern.replace("{year}", &year.to_string());
        Path::new(&self.dir_path)
            .join(fname)
            .to_str()
            .unwrap_or("")
            .to_string()
    }

    /// Discover available years by scanning the directory for matching files.
    pub fn discover_years(&self) -> Vec<i32> {
        let mut years = BTreeSet::new();
        if let Ok(entries) = std::fs::read_dir(&self.dir_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".tif") {
                    let stem = name.trim_end_matches(".tif");
                    if let Ok(y) = stem.parse::<i32>() {
                        years.insert(y);
                    }
                }
            }
        }

        if years.is_empty() {
            let (prefix, suffix) = self.file_pattern.split_once("{year}").unwrap_or(("", ""));
            let suffix = suffix.trim_end_matches(".tif");
            if let Ok(entries) = std::fs::read_dir(&self.dir_path) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with(prefix) && name.ends_with(&format!("{suffix}.tif")) {
                        let inner = &name[prefix.len()..name.len() - suffix.len() - 4];
                        if let Ok(y) = inner.parse::<i32>() {
                            years.insert(y);
                        }
                    }
                }
            }
        }

        years.into_iter().collect()
    }

    /// Load disturbance data for a specific year and spatial window.
    pub fn load(
        &self,
        year: i32,
        window: &Window,
    ) -> LoaderResult<Option<(Array2<u8>, Option<f64>)>> {
        let path = self.build_path(year);
        if !Path::new(&path).exists() {
            return Ok(None);
        }

        let ds = Dataset::open(&path)?;
        let band = ds.rasterband(1)?;
        let nodata = band.no_data_value();

        let data = band
            .read_as::<u8>(
                (window.col_off as isize, window.row_off as isize),
                (window.width, window.height),
                (window.width, window.height),
                None,
            )
            .map_err(|e| LoaderError::Gdal(e.to_string()))?;

        let arr = Array2::from_shape_vec((window.height, window.width), data.data)
            .map_err(|e| LoaderError::Gdal(format!("shape: {e}")))?;

        Ok(Some((arr, nodata)))
    }

    /// Load disturbance data as f32, with NoData values replaced by NaN.
    pub fn load_f32(&self, year: i32, window: &Window) -> LoaderResult<Option<Array2<f32>>> {
        let (arr, nodata) = match self.load(year, window)? {
            Some(v) => v,
            None => return Ok(None),
        };

        let mut f32_arr = arr.mapv(|v| v as f32);
        if let Some(nd) = nodata {
            let nd_f32 = nd as f32;
            f32_arr.mapv_inplace(|v| {
                if (v - nd_f32).abs() < f32::EPSILON {
                    f32::NAN
                } else {
                    v
                }
            });
        }
        Ok(Some(f32_arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::DriverManager;
    use tempfile::TempDir;

    fn create_synthetic_ccdc(dir: &TempDir, year: i32, data: &[u8], w: usize, h: usize) {
        let path = dir.path().join(format!("{year}.tif"));
        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let ds = driver
            .create(path.to_str().unwrap(), w as isize, h as isize, 1)
            .unwrap();
        let mut band = ds.rasterband(1).unwrap();
        band.set_no_data_value(Some(0.0)).unwrap();
        let buf = gdal::raster::Buffer::new((w, h), data.to_vec());
        band.write((0, 0), (w, h), &buf).unwrap();
        drop(band);
        drop(ds);
    }

    #[test]
    fn test_ccdc_annual_year_discovery() {
        let dir = TempDir::new().unwrap();
        create_synthetic_ccdc(&dir, 2018, &[1, 2, 3, 4], 2, 2);
        create_synthetic_ccdc(&dir, 2019, &[1, 2, 3, 4], 2, 2);
        create_synthetic_ccdc(&dir, 2020, &[1, 2, 3, 4], 2, 2);

        let loader = CCDCLoader::new_default(dir.path().to_str().unwrap());
        let years = loader.discover_years();
        assert_eq!(years, vec![2018, 2019, 2020]);
    }

    #[test]
    fn test_ccdc_load() {
        let dir = TempDir::new().unwrap();
        create_synthetic_ccdc(&dir, 2021, &[1, 2, 3, 4], 2, 2);

        let loader = CCDCLoader::new_default(dir.path().to_str().unwrap());
        let w = Window::new(0, 0, 2, 2);
        let (arr, nodata) = loader.load(2021, &w).unwrap().unwrap();

        assert_eq!(arr[[0, 0]], 1);
        assert_eq!(arr[[0, 1]], 2);
        assert_eq!(arr[[1, 0]], 3);
        assert_eq!(arr[[1, 1]], 4);
        assert_eq!(nodata, Some(0.0));
    }

    #[test]
    fn test_ccdc_missing_year() {
        let dir = TempDir::new().unwrap();
        let loader = CCDCLoader::new_default(dir.path().to_str().unwrap());
        let w = Window::new(0, 0, 1, 1);
        let result = loader.load(2022, &w).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_ccdc_load_f32_nodata() {
        let dir = TempDir::new().unwrap();
        create_synthetic_ccdc(&dir, 2023, &[0, 1, 2, 255], 2, 2);

        let loader = CCDCLoader::new_default(dir.path().to_str().unwrap());
        let w = Window::new(0, 0, 2, 2);
        let arr = loader.load_f32(2023, &w).unwrap().unwrap();

        assert!(arr[[0, 0]].is_nan());
        assert_eq!(arr[[0, 1]], 1.0);
        assert_eq!(arr[[1, 0]], 2.0);
        assert_eq!(arr[[1, 1]], 255.0);
    }
}

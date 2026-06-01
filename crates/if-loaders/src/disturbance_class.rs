use gdal::Dataset;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

use crate::base::{LoaderError, LoaderResult, Window};

pub struct DisturbanceClassLoader {
    path: String,
    #[allow(dead_code)]
    start_year: i32,
    year_band_map: HashMap<i32, usize>,
    #[allow(dead_code)]
    band_year_map: HashMap<usize, i32>,
}

impl DisturbanceClassLoader {
    pub fn open(path: &str, start_year: i32) -> LoaderResult<Self> {
        if !Path::new(path).exists() {
            return Err(LoaderError::FileNotFound(path.to_string()));
        }

        let ds = Dataset::open(path)?;
        let band_count = ds.raster_count() as usize;
        drop(ds);

        let mut year_band_map = HashMap::new();
        let mut band_year_map = HashMap::new();

        for band in 1..=band_count {
            let year = start_year + (band as i32) - 1;
            year_band_map.insert(year, band);
            band_year_map.insert(band, year);
        }

        Ok(Self {
            path: path.to_string(),
            start_year,
            year_band_map,
            band_year_map,
        })
    }

    pub fn available_years(&self) -> Vec<i32> {
        let mut years: Vec<i32> = self.year_band_map.keys().copied().collect();
        years.sort();
        years
    }

    /// Load disturbance class data for a specific year and window.
    pub fn load(
        &self,
        year: i32,
        window: &Window,
    ) -> LoaderResult<Option<(Array2<u8>, Option<f64>)>> {
        let band = match self.year_band_map.get(&year) {
            Some(b) => *b,
            None => return Ok(None),
        };

        let ds = Dataset::open(&self.path)?;
        let raster_band = ds.rasterband(band as isize)?;
        let nodata = raster_band.no_data_value();

        let data = raster_band
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

    /// Load disturbance class data as f32 with NoData replaced by NaN.
    pub fn load_f32(
        &self,
        year: i32,
        window: &Window,
    ) -> LoaderResult<Option<Array2<f32>>> {
        let (arr_u8, nodata) = match self.load(year, window)? {
            Some(v) => v,
            None => return Ok(None),
        };

        let mut arr = arr_u8.mapv(|v| v as f32);
        if let Some(nd) = nodata {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::DriverManager;
    use tempfile::TempDir;

    fn create_synthetic_multiband(path: &str, bands_data: &[Vec<u8>], w: usize, h: usize) {
        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let ds = driver
            .create(path, w as isize, h as isize, bands_data.len() as isize)
            .unwrap();

        for (i, data) in bands_data.iter().enumerate() {
            let mut band = ds.rasterband((i + 1) as isize).unwrap();
            let buf = gdal::raster::Buffer::new((w, h), data.clone());
            band.write((0, 0), (w, h), &buf).unwrap();
        }
        drop(ds);
    }

    #[test]
    fn test_disturbance_class_years() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("classes.tif");
        let path_str = path.to_str().unwrap().to_string();

        let band1 = vec![1u8, 2, 3, 4];
        let band2 = vec![2u8, 1, 4, 3];
        let band3 = vec![3u8, 4, 1, 2];
        create_synthetic_multiband(&path_str, &[band1, band2, band3], 2, 2);

        let loader = DisturbanceClassLoader::open(&path_str, 2016).unwrap();
        assert_eq!(loader.available_years(), vec![2016, 2017, 2018]);
    }

    #[test]
    fn test_disturbance_class_load() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("classes.tif");
        let path_str = path.to_str().unwrap().to_string();

        let band1 = vec![1u8, 2, 3, 4];
        let band2 = vec![2u8, 1, 4, 3];
        create_synthetic_multiband(&path_str, &[band1, band2], 2, 2);

        let loader = DisturbanceClassLoader::open(&path_str, 2016).unwrap();

        let w = Window::new(0, 0, 2, 2);
        let (arr, _) = loader.load(2016, &w).unwrap().unwrap();
        assert_eq!(arr[[0, 0]], 1);
        assert_eq!(arr[[1, 1]], 4);

        let (arr2, _) = loader.load(2017, &w).unwrap().unwrap();
        assert_eq!(arr2[[0, 0]], 2);
        assert_eq!(arr2[[1, 1]], 3);
    }

    #[test]
    fn test_disturbance_class_missing_year() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("classes.tif");
        let path_str = path.to_str().unwrap().to_string();

        create_synthetic_multiband(&path_str, &[vec![1u8; 4]], 2, 2);
        let loader = DisturbanceClassLoader::open(&path_str, 2016).unwrap();

        let w = Window::new(0, 0, 1, 1);
        let result = loader.load(2020, &w).unwrap();
        assert!(result.is_none());
    }
}

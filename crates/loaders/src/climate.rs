use gdal::Dataset;
use ndarray::Array2;

use crate::base::{LoaderError, LoaderResult, Window};

/// Loads monthly rainfall GeoTIFF files and annual temperature NetCDF files.
pub struct ClimateLoader {
    rainfall_dir: Option<String>,
    temperature_dir: Option<String>,
    rainfall_pattern: String,
    temperature_pattern: String,
}

impl ClimateLoader {
    pub fn new(rainfall_dir: Option<&str>, temperature_dir: Option<&str>) -> Self {
        Self {
            rainfall_dir: rainfall_dir.map(String::from),
            temperature_dir: temperature_dir.map(String::from),
            rainfall_pattern: "pre{year:04}{month:02}.tif".to_string(),
            temperature_pattern: "tmp_{year}.nc".to_string(),
        }
    }

    pub fn with_patterns(
        rainfall_dir: Option<&str>,
        temperature_dir: Option<&str>,
        rainfall_pattern: &str,
        temperature_pattern: &str,
    ) -> Self {
        Self {
            rainfall_dir: rainfall_dir.map(String::from),
            temperature_dir: temperature_dir.map(String::from),
            rainfall_pattern: rainfall_pattern.to_string(),
            temperature_pattern: temperature_pattern.to_string(),
        }
    }

    fn build_monthly_path(&self, year: i32, month: u32) -> Option<String> {
        let dir = self.rainfall_dir.as_ref()?;
        let fname = self
            .rainfall_pattern
            .replace("{year:04}", &format!("{year:04}"))
            .replace("{month:02}", &format!("{month:02}"))
            .replace("{year}", &year.to_string())
            .replace("{month}", &month.to_string());
        let path = std::path::Path::new(dir).join(&fname);
        if path.exists() {
            Some(path.to_str()?.to_string())
        } else {
            None
        }
    }

    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    fn build_temperature_path(&self, year: i32) -> Option<String> {
        let dir = self.temperature_dir.as_ref()?;
        let fname = self
            .temperature_pattern
            .replace("{year:04}", &format!("{year:04}"))
            .replace("{year}", &year.to_string());
        let path = std::path::Path::new(dir).join(&fname);
        if path.exists() {
            Some(path.to_str()?.to_string())
        } else {
            None
        }
    }

    /// Load monthly rainfall for a given year/month and spatial window.
    pub fn load_rainfall_month(
        &self,
        year: i32,
        month: u32,
        window: &Window,
    ) -> LoaderResult<Option<Array2<f32>>> {
        let path = match self.build_monthly_path(year, month) {
            Some(p) => p,
            None => return Ok(None),
        };

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

        let arr = Array2::from_shape_vec((window.height, window.width), data.data().to_vec())
            .map_err(|e| LoaderError::Gdal(format!("shape: {e}")))?;

        Ok(Some(arr))
    }

    /// Load annual temperature for a given year and spatial window.
    #[cfg(feature = "netcdf")]
    pub fn load_temperature_year(
        &self,
        year: i32,
        window: &Window,
    ) -> LoaderResult<Option<Array2<f32>>> {
        let path = match self.build_temperature_path(year) {
            Some(p) => p,
            None => return Ok(None),
        };

        let file = netcdf::open(&path).map_err(|e| LoaderError::NetCdf(e.to_string()))?;

        let var = file
            .variable("tmp")
            .map_err(|e| LoaderError::NetCdf(format!("variable 'tmp': {e}")))?;

        let values = var
            .values::<f32>(None, None)
            .map_err(|e| LoaderError::NetCdf(e.to_string()))?;

        let shape = var.shape();
        let (rows, cols) = if shape.len() == 3 {
            (shape[1], shape[2])
        } else if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            return Err(LoaderError::NetCdf(format!("unexpected shape: {shape:?}")));
        };

        let arr = Array2::from_shape_vec((rows, cols), values.into_raw())
            .map_err(|e| LoaderError::NetCdf(format!("reshape: {e}")))?;

        let end_row = (window.row_off + window.height).min(rows);
        let end_col = (window.col_off + window.width).min(cols);
        let slice = arr.slice(ndarray::s![
            window.row_off..end_row,
            window.col_off..end_col
        ]);
        Ok(Some(slice.to_owned()))
    }

    /// Load annual temperature (stub when netcdf feature disabled).
    #[cfg(not(feature = "netcdf"))]
    pub fn load_temperature_year(
        &self,
        _year: i32,
        _window: &Window,
    ) -> LoaderResult<Option<Array2<f32>>> {
        Err(LoaderError::MissingFeature(
            "netcdf feature not enabled".into(),
        ))
    }

    /// Discover available rainfall years by scanning directory.
    pub fn discover_rainfall_years(&self) -> Vec<i32> {
        let dir = match &self.rainfall_dir {
            Some(d) => d,
            None => return vec![],
        };

        let mut years = std::collections::BTreeSet::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("pre") && name.ends_with(".tif") && name.len() >= 8 {
                    if let Ok(y) = name[3..7].parse::<i32>() {
                        years.insert(y);
                    }
                }
            }
        }
        years.into_iter().collect()
    }

    /// Discover available temperature years by scanning directory.
    pub fn discover_temperature_years(&self) -> Vec<i32> {
        let dir = match &self.temperature_dir {
            Some(d) => d,
            None => return vec![],
        };

        let mut years = std::collections::BTreeSet::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("tmp_") && name.ends_with(".nc") && name.len() >= 9 {
                    if let Ok(y) = name[4..8].parse::<i32>() {
                        years.insert(y);
                    }
                }
            }
        }
        years.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::DriverManager;
    use tempfile::TempDir;

    fn create_synthetic_rainfall(path: &str, data: &[f32], w: usize, h: usize) {
        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let ds = driver.create(path, w, h, 1).unwrap();
        let mut band = ds.rasterband(1).unwrap();
        let mut buf = gdal::raster::Buffer::new((w, h), data.to_vec());
        band.write((0, 0), (w, h), &mut buf).unwrap();
        drop(band);
        drop(ds);
    }

    #[test]
    fn test_rainfall_monthly_load() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pre202007.tif");
        let path_str = path.to_str().unwrap().to_string();

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        create_synthetic_rainfall(&path_str, &data, 2, 2);

        let loader = ClimateLoader::new(Some(dir.path().to_str().unwrap()), None);

        let w = Window::new(0, 0, 2, 2);
        let result = loader.load_rainfall_month(2020, 7, &w).unwrap().unwrap();
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 1]], 4.0);
    }

    #[test]
    fn test_rainfall_missing_month() {
        let dir = TempDir::new().unwrap();
        let loader = ClimateLoader::new(Some(dir.path().to_str().unwrap()), None);
        let w = Window::new(0, 0, 1, 1);
        let result = loader.load_rainfall_month(2020, 6, &w).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_discover_rainfall_years() {
        let dir = TempDir::new().unwrap();
        create_synthetic_rainfall(
            dir.path().join("pre201801.tif").to_str().unwrap(),
            &[1.0],
            1,
            1,
        );
        create_synthetic_rainfall(
            dir.path().join("pre201901.tif").to_str().unwrap(),
            &[1.0],
            1,
            1,
        );
        create_synthetic_rainfall(
            dir.path().join("pre202001.tif").to_str().unwrap(),
            &[1.0],
            1,
            1,
        );

        let loader = ClimateLoader::new(Some(dir.path().to_str().unwrap()), None);
        let years = loader.discover_rainfall_years();
        assert_eq!(years, vec![2018, 2019, 2020]);
    }
}

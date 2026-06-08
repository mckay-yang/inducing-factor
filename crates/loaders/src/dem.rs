use gdal::Dataset;
use ndarray::Array2;

use crate::base::{apply_nodata_mask, LoaderError, LoaderResult, RasterWindowLoader, Window};

const DEM_NODATA: f32 = -9999.0;

pub struct DEMLoader {
    path: String,
    nodata: f32,
    res_x: f64,
    res_y: f64,
}

impl DEMLoader {
    pub fn open(path: &str) -> LoaderResult<Self> {
        let ds = Dataset::open(path)?;
        let geo_transform = ds.geo_transform()?;
        let band = ds.rasterband(1)?;
        let nodata = band.no_data_value().map(|v| v as f32).unwrap_or(DEM_NODATA);

        Ok(Self {
            path: path.to_string(),
            nodata,
            res_x: geo_transform[1].abs(),
            res_y: geo_transform[5].abs(),
        })
    }

    /// Return the DEM resolution as (res_x, res_y).
    pub fn resolution(&self) -> (f64, f64) {
        (self.res_x, self.res_y)
    }

    /// Read raw elevation from a window without computing slope/aspect.
    pub fn read_elevation(&self, window: &Window) -> LoaderResult<Array2<f32>> {
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

        let rows = window.height;
        let cols = window.width;
        let mut arr = Array2::from_shape_vec((rows, cols), data.data().to_vec()).map_err(|e| {
            LoaderError::Gdal(format!("shape mismatch: expected ({rows}, {cols}): {e}"))
        })?;

        apply_nodata_mask(&mut arr, self.nodata);
        Ok(arr)
    }

    /// Calculate slope from an elevation array using numpy-style gradient.
    ///
    /// Returns slope in degrees.  NoData pixels produce NaN output.
    pub fn calculate_slope(elev: &Array2<f32>, res_x: f64, res_y: f64) -> Array2<f32> {
        let (rows, cols) = elev.dim();
        let mut slope = Array2::<f32>::zeros((rows, cols));

        for r in 0..rows {
            for c in 0..cols {
                if elev[[r, c]].is_nan() {
                    slope[[r, c]] = f32::NAN;
                    continue;
                }
                let dz_dx = central_diff_x(elev, r, c, res_x);
                let dz_dy = central_diff_y(elev, r, c, res_y);
                let rise = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt();
                slope[[r, c]] = rise.atan().to_degrees();
            }
        }
        slope
    }

    /// Calculate aspect from an elevation array using numpy-style gradient.
    ///
    /// Returns aspect in degrees (0-360, clockwise from north).
    /// NoData pixels produce NaN output.
    pub fn calculate_aspect(elev: &Array2<f32>, res_x: f64, res_y: f64) -> Array2<f32> {
        let (rows, cols) = elev.dim();
        let mut aspect = Array2::<f32>::zeros((rows, cols));

        for r in 0..rows {
            for c in 0..cols {
                if elev[[r, c]].is_nan() {
                    aspect[[r, c]] = f32::NAN;
                    continue;
                }
                let dz_dx = central_diff_x(elev, r, c, res_x);
                let dz_dy = central_diff_y(elev, r, c, res_y);
                let mut deg = dz_dy.atan2(-dz_dx).to_degrees();

                if deg < 0.0 {
                    deg = 90.0 - deg;
                } else if deg > 90.0 {
                    deg = 360.0 - deg + 90.0;
                } else {
                    deg = 90.0 - deg;
                }
                aspect[[r, c]] = deg;
            }
        }
        aspect
    }

    /// Convenience: load elevation + slope + aspect for a window.
    pub fn load_slope_aspect(
        &self,
        window: &Window,
    ) -> LoaderResult<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let elev = self.read_elevation(window)?;
        let slope = Self::calculate_slope(&elev, self.res_x, self.res_y);
        let aspect = Self::calculate_aspect(&elev, self.res_x, self.res_y);
        Ok((elev, slope, aspect))
    }
}

impl RasterWindowLoader for DEMLoader {
    fn load_window(&self, window: &Window) -> LoaderResult<Array2<f32>> {
        self.read_elevation(window)
    }

    fn nodata_value(&self) -> Option<f32> {
        Some(self.nodata)
    }
}

fn central_diff_x(arr: &Array2<f32>, r: usize, c: usize, res: f64) -> f32 {
    let (_, cols) = arr.dim();
    let res = res as f32;
    if c == 0 {
        (arr[[r, 1]] - arr[[r, 0]]) / res
    } else if c == cols - 1 {
        (arr[[r, c]] - arr[[r, c - 1]]) / res
    } else {
        (arr[[r, c + 1]] - arr[[r, c - 1]]) / (2.0 * res)
    }
}

fn central_diff_y(arr: &Array2<f32>, r: usize, c: usize, res: f64) -> f32 {
    let (rows, _) = arr.dim();
    let res = res as f32;
    if r == 0 {
        (arr[[1, c]] - arr[[0, c]]) / res
    } else if r == rows - 1 {
        (arr[[r, c]] - arr[[r - 1, c]]) / res
    } else {
        (arr[[r + 1, c]] - arr[[r - 1, c]]) / (2.0 * res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::DriverManager;
    use tempfile::TempDir;

    fn create_synthetic_dem() -> (TempDir, String) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_dem.tif");
        let path_str = path.to_str().unwrap().to_string();

        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let mut ds = driver.create(&path_str, 5, 5, 1).unwrap();
        ds.set_geo_transform(&[0.0, 30.0, 0.0, 0.0, 0.0, -30.0])
            .unwrap();
        let mut band = ds.rasterband(1).unwrap();
        band.set_no_data_value(Some(-9999.0)).unwrap();

        let data: Vec<f32> = vec![
            100.0, 110.0, 120.0, -9999.0, 90.0, 105.0, 115.0, 125.0, 95.0, 85.0, 110.0, 120.0,
            130.0, 100.0, 80.0, 115.0, 125.0, 135.0, 105.0, 75.0, -9999.0, 130.0, 140.0, 110.0,
            70.0,
        ];
        let mut buf = gdal::raster::Buffer::new((5, 5), data);
        band.write((0, 0), (5, 5), &mut buf).unwrap();
        drop(band);
        drop(ds);

        (dir, path_str)
    }

    #[test]
    fn test_dem_window_read() {
        let (_dir, path) = create_synthetic_dem();
        let loader = DEMLoader::open(&path).unwrap();

        let w = Window::new(1, 1, 3, 3);
        let elev = loader.read_elevation(&w).unwrap();

        assert_eq!(elev.dim(), (3, 3));
        assert!((elev[[0, 0]] - 115.0).abs() < 0.01);
        assert!((elev[[1, 1]] - 130.0).abs() < 0.01);
    }

    #[test]
    fn test_dem_nodata() {
        let (_dir, path) = create_synthetic_dem();
        let loader = DEMLoader::open(&path).unwrap();

        let w = Window::new(0, 0, 5, 5);
        let elev = loader.read_elevation(&w).unwrap();

        assert!(elev[[2, 2]].is_finite());
    }

    #[test]
    fn test_dem_nodata_direct() {
        let mut arr = ndarray::Array2::from_shape_vec(
            (2, 2),
            vec![100.0f32, -9999.0f32, -9999.0f32, 50.0f32],
        )
        .unwrap();
        apply_nodata_mask(&mut arr, -9999.0);
        assert!(arr[[0, 0]].is_finite());
        assert!(arr[[0, 1]].is_nan());
        assert!(arr[[1, 0]].is_nan());
        assert!(arr[[1, 1]].is_finite());
    }

    #[test]
    fn test_slope_computation() {
        let (_dir, path) = create_synthetic_dem();
        let loader = DEMLoader::open(&path).unwrap();
        let (res_x, res_y) = loader.resolution();

        let w = Window::new(1, 1, 3, 3);
        let elev = loader.read_elevation(&w).unwrap();
        let slope = DEMLoader::calculate_slope(&elev, res_x, res_y);

        assert_eq!(slope.dim(), (3, 3));
        assert!(slope[[1, 1]].is_finite());
    }

    #[test]
    fn test_aspect_computation() {
        let (_dir, path) = create_synthetic_dem();
        let loader = DEMLoader::open(&path).unwrap();
        let (res_x, res_y) = loader.resolution();

        let w = Window::new(1, 1, 3, 3);
        let elev = loader.read_elevation(&w).unwrap();
        let aspect = DEMLoader::calculate_aspect(&elev, res_x, res_y);

        assert_eq!(aspect.dim(), (3, 3));
        assert!((0.0..=360.0).contains(&aspect[[1, 1]]) || aspect[[1, 1]].is_nan());
    }

    #[test]
    fn test_load_slope_aspect() {
        let (_dir, path) = create_synthetic_dem();
        let loader = DEMLoader::open(&path).unwrap();

        let w = Window::new(0, 0, 5, 5);
        let (elev, slope, aspect) = loader.load_slope_aspect(&w).unwrap();

        assert_eq!(elev.dim(), (5, 5));
        assert_eq!(slope.dim(), (5, 5));
        assert_eq!(aspect.dim(), (5, 5));
    }

    #[test]
    fn test_raster_window_loader_trait() {
        let (_dir, path) = create_synthetic_dem();
        let loader = DEMLoader::open(&path).unwrap();

        assert_eq!(loader.nodata_value(), Some(-9999.0));

        let w = Window::new(0, 0, 2, 2);
        let data = loader.load_window(&w).unwrap();
        assert_eq!(data.dim(), (2, 2));
    }
}

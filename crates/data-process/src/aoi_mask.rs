//! AOI geometry mask in the CLASS grid CRS.

use std::path::Path;

use gdal_crate::spatial_ref::{AxisMappingStrategy, SpatialRef};
use gdal_crate::vector::LayerAccess;
use gdal_crate::Dataset;
use geo::{BoundingRect, Contains, Coord, Intersects, Point, Rect};
use loaders::base::{LoaderError, LoaderResult};
use std::convert::TryInto;

use crate::class_grid::ClassGrid;

#[derive(Debug, Clone)]
pub struct AoiMask {
    geometries: Vec<geo::Geometry<f64>>,
    bounds: Rect<f64>,
}

impl AoiMask {
    pub fn from_grid_bounds(class_grid: &ClassGrid) -> Self {
        let last_row = class_grid.height().saturating_sub(1);
        let last_col = class_grid.width().saturating_sub(1);
        let corners = [
            class_grid.pixel_to_geo(0, 0),
            class_grid.pixel_to_geo(0, last_col),
            class_grid.pixel_to_geo(last_row, 0),
            class_grid.pixel_to_geo(last_row, last_col),
        ];
        let min_x = corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::INFINITY, f64::min);
        let max_x = corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::INFINITY, f64::min);
        let max_y = corners
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max);
        let bounds = Rect::new(Coord { x: min_x, y: min_y }, Coord { x: max_x, y: max_y });
        Self {
            geometries: vec![geo::Geometry::Rect(bounds)],
            bounds,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_rect(bounds: Rect<f64>) -> Self {
        Self {
            geometries: vec![geo::Geometry::Rect(bounds)],
            bounds,
        }
    }

    pub fn load(path: &Path, class_grid: &ClassGrid) -> LoaderResult<Self> {
        let ds = Dataset::open(path)
            .map_err(|e| LoaderError::Gdal(format!("open AOI {}: {e}", path.display())))?;
        let mut target_srs = if class_grid.crs().is_empty() {
            None
        } else {
            Some(
                SpatialRef::from_wkt(class_grid.crs())
                    .map_err(|e| LoaderError::Gdal(format!("CLASS CRS WKT: {e}")))?,
            )
        };
        if let Some(srs) = target_srs.as_mut() {
            srs.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
        }

        let mut geometries = Vec::new();
        let mut bounds: Option<Rect<f64>> = None;

        for layer_idx in 0..ds.layer_count() {
            let mut layer = ds
                .layer(layer_idx)
                .map_err(|e| LoaderError::Gdal(format!("AOI layer {layer_idx}: {e}")))?;
            let mut layer_srs = layer.spatial_ref();
            if let Some(srs) = layer_srs.as_mut() {
                srs.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
            }

            for feature in layer.features() {
                let Some(ogr_geom) = feature.geometry() else {
                    continue;
                };
                if ogr_geom.is_empty() {
                    continue;
                }

                let mut geom = ogr_geom.clone();
                if let Some(srs) = layer_srs.clone() {
                    geom.set_spatial_ref(srs);
                }
                if let Some(target) = target_srs.as_ref() {
                    geom = geom.transform_to(target).map_err(|e| {
                        LoaderError::Gdal(format!("transform AOI to CLASS CRS: {e}"))
                    })?;
                }

                let json = geom
                    .json()
                    .map_err(|e| LoaderError::Gdal(format!("AOI GeoJSON export: {e}")))?;
                let gj: geojson::Geometry = json
                    .parse()
                    .map_err(|e| LoaderError::Gdal(format!("AOI GeoJSON parse: {e}")))?;
                let geo_geom: geo::Geometry<f64> = gj
                    .try_into()
                    .map_err(|e| LoaderError::Gdal(format!("AOI geo conversion: {e}")))?;

                if let Some(rect) = geo_geom.bounding_rect() {
                    bounds = Some(match bounds {
                        Some(existing) => union_rect(existing, rect),
                        None => rect,
                    });
                }
                geometries.push(geo_geom);
            }
        }

        let bounds = bounds.ok_or_else(|| {
            LoaderError::VectorRead(format!("AOI {} has no usable geometry", path.display()))
        })?;
        Ok(Self { geometries, bounds })
    }

    pub fn contains_pixel(&self, class_grid: &ClassGrid, row: usize, col: usize) -> bool {
        let (x, y) = class_grid.pixel_to_geo(row, col);
        self.contains_xy(x, y)
    }

    pub fn contains_xy(&self, x: f64, y: f64) -> bool {
        if x < self.bounds.min().x
            || x > self.bounds.max().x
            || y < self.bounds.min().y
            || y > self.bounds.max().y
        {
            return false;
        }
        let point = Point::new(x, y);
        self.geometries
            .iter()
            .any(|geom| geom.contains(&point) || geom.intersects(&point))
    }

    pub fn tile_mask(
        &self,
        class_grid: &ClassGrid,
        row_off: usize,
        col_off: usize,
        width: usize,
        height: usize,
    ) -> Vec<bool> {
        let mut mask = Vec::with_capacity(width * height);
        for dy in 0..height {
            for dx in 0..width {
                mask.push(self.contains_pixel(class_grid, row_off + dy, col_off + dx));
            }
        }
        mask
    }

    pub fn window_intersects_bounds(
        &self,
        class_grid: &ClassGrid,
        row_off: usize,
        col_off: usize,
        width: usize,
        height: usize,
    ) -> bool {
        let row_last = row_off + height.saturating_sub(1);
        let col_last = col_off + width.saturating_sub(1);
        let corners = [
            class_grid.pixel_to_geo(row_off, col_off),
            class_grid.pixel_to_geo(row_off, col_last),
            class_grid.pixel_to_geo(row_last, col_off),
            class_grid.pixel_to_geo(row_last, col_last),
        ];
        let min_x = corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::INFINITY, f64::min);
        let max_x = corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::INFINITY, f64::min);
        let max_y = corners
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max);
        let window_bounds = Rect::new(Coord { x: min_x, y: min_y }, Coord { x: max_x, y: max_y });
        self.bounds.intersects(&window_bounds)
    }
}

fn union_rect(a: Rect<f64>, b: Rect<f64>) -> Rect<f64> {
    Rect::new(
        Coord {
            x: a.min().x.min(b.min().x),
            y: a.min().y.min(b.min().y),
        },
        Coord {
            x: a.max().x.max(b.max().x),
            y: a.max().y.max(b.max().y),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn project_path(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(rel)
    }

    #[test]
    fn loads_formal_aoi_and_matches_class_center() {
        let class_path = project_path("data/input/disturbance_class/disturbance_class_res.tif");
        let aoi_path = project_path("data/input/aoi/LangToBasighat.shp");
        let class_grid = ClassGrid::open(&class_path.to_string_lossy(), 2016).unwrap();
        let mask = AoiMask::load(&aoi_path, &class_grid).unwrap();

        let row = class_grid.height() / 2;
        let col = class_grid.width() / 2;
        assert!(mask.contains_pixel(&class_grid, row, col));
    }
}

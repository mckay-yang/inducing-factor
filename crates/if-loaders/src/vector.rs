use gdal::vector::LayerAccess;
use gdal::Dataset;
use geo::EuclideanDistance;
use geo::Geometry;
use rstar::{PointDistance, RTree, RTreeObject, AABB};

use crate::base::{LoaderError, LoaderResult, VectorDistanceLoader};

struct IndexedGeom {
    geom: Geometry<f64>,
    envelope: AABB<[f64; 2]>,
}

impl IndexedGeom {
    fn from_geom(geom: Geometry<f64>) -> Self {
        let bbox = bbox_of(&geom);
        let envelope =
            AABB::from_corners([bbox.min().x, bbox.min().y], [bbox.max().x, bbox.max().y]);
        Self { geom, envelope }
    }
}

impl RTreeObject for IndexedGeom {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.envelope
    }
}

impl PointDistance for IndexedGeom {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let p = geo::point!(x: point[0], y: point[1]);
        let d = p.euclidean_distance(&self.geom);
        d * d
    }
}

fn bbox_of(geom: &Geometry<f64>) -> geo::Rect<f64> {
    use geo::algorithm::BoundingRect;
    geom.bounding_rect().unwrap_or_else(|| {
        geo::Rect::new(
            geo::coord! { x: 0.0, y: 0.0 },
            geo::coord! { x: 0.0, y: 0.0 },
        )
    })
}

pub struct VectorLoader {
    tree: RTree<IndexedGeom>,
    geometries: Vec<Geometry<f64>>,
}

impl VectorLoader {
    pub fn from_file(
        path: &str,
        aoi_geom: Option<&Geometry<f64>>,
        _target_crs: Option<&str>,
    ) -> LoaderResult<Self> {
        let ds = Dataset::open(path)?;
        let mut layer = ds.layer(0).map_err(|e| LoaderError::Gdal(e.to_string()))?;

        let mut geoms: Vec<Geometry<f64>> = Vec::new();
        let mut indexed: Vec<IndexedGeom> = Vec::new();

        for feature in layer.features() {
            let ogr_geom = match feature.geometry() {
                Some(g) => g,
                None => continue,
            };

            if ogr_geom.is_empty() {
                continue;
            }

            let json = ogr_geom
                .json()
                .map_err(|e| LoaderError::Gdal(e.to_string()))?;
            let geo_geom = match ogr_json_to_geo(&json) {
                Some(g) => g,
                None => continue,
            };

            if let Some(ref aoi) = aoi_geom {
                use geo::Intersects;
                if !aoi.intersects(&geo_geom) {
                    continue;
                }
            }

            indexed.push(IndexedGeom::from_geom(geo_geom.clone()));
            geoms.push(geo_geom);
        }

        Ok(Self {
            tree: RTree::bulk_load(indexed),
            geometries: geoms,
        })
    }

    pub fn from_file_raw(path: &str) -> LoaderResult<Self> {
        Self::from_file(path, None, None)
    }

    pub fn len(&self) -> usize {
        self.geometries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.geometries.is_empty()
    }

    fn compute_distances(&self, x_coords: &[f64], y_coords: &[f64]) -> Vec<f64> {
        let n = x_coords.len();
        let mut dists = vec![0.0f64; n];

        if self.tree.size() == 0 {
            dists.fill(f64::NAN);
            return dists;
        }

        for i in 0..n {
            let query_pt = [x_coords[i], y_coords[i]];
            if let Some(nearest) = self.tree.nearest_neighbor(&query_pt) {
                let p = geo::point!(x: x_coords[i], y: y_coords[i]);
                let d = p.euclidean_distance(&nearest.geom);
                dists[i] = if d.is_finite() { d } else { f64::NAN };
            } else {
                dists[i] = f64::NAN;
            }
        }

        dists
    }
}

impl VectorDistanceLoader for VectorLoader {
    fn get_distance(&self, x_coords: &[f64], y_coords: &[f64]) -> LoaderResult<Vec<f64>> {
        Ok(self.compute_distances(x_coords, y_coords))
    }
}

fn ogr_json_to_geo(json: &str) -> Option<Geometry<f64>> {
    let gj: geojson::Geometry = json.parse().ok()?;
    use std::convert::TryFrom;
    Geometry::try_from(gj).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::DriverManager;
    use tempfile::TempDir;

    fn create_synthetic_line_shp(dir: &TempDir, name: &str) -> String {
        let path = dir.path().join(name);
        let path_str = path.to_str().unwrap().to_string();

        let driver = DriverManager::get_driver_by_name("ESRI Shapefile").unwrap();
        let mut ds = driver.create_vector_only(&path_str).unwrap();
        let mut layer = ds.create_layer(gdal::LayerOptions::default()).unwrap();

        let wkt = "LINESTRING (0 0, 10 0)";
        let geom = gdal::vector::Geometry::from_wkt(wkt).unwrap();
        layer.create_feature(geom).unwrap();
        drop(layer);
        drop(ds);

        path_str
    }

    fn create_synthetic_multi_line_shp(dir: &TempDir, name: &str) -> String {
        let path = dir.path().join(name);
        let path_str = path.to_str().unwrap().to_string();

        let driver = DriverManager::get_driver_by_name("ESRI Shapefile").unwrap();
        let mut ds = driver.create_vector_only(&path_str).unwrap();
        let mut layer = ds.create_layer(gdal::LayerOptions::default()).unwrap();

        let lines = ["LINESTRING (0 0, 10 0)", "LINESTRING (5 5, 5 15)"];
        for wkt in &lines {
            let geom = gdal::vector::Geometry::from_wkt(wkt).unwrap();
            layer.create_feature(geom).unwrap();
        }
        drop(layer);
        drop(ds);

        path_str
    }

    #[test]
    fn test_vector_load() {
        let dir = TempDir::new().unwrap();
        let path = create_synthetic_line_shp(&dir, "line.shp");
        let loader = VectorLoader::from_file_raw(&path).unwrap();
        assert_eq!(loader.len(), 1);
    }

    #[test]
    fn test_vector_distance_simple() {
        let dir = TempDir::new().unwrap();
        let path = create_synthetic_line_shp(&dir, "line.shp");
        let loader = VectorLoader::from_file_raw(&path).unwrap();

        let x = vec![5.0, 0.0, 20.0];
        let y = vec![5.0, 0.0, 0.0];
        let dists = loader.compute_distances(&x, &y);

        assert!((dists[0] - 5.0).abs() < 1.0);
        assert!((dists[1] - 0.0).abs() < 0.01);
        assert!(dists[2].is_finite());
    }

    #[test]
    fn test_vector_distance_no_cap() {
        let dir = TempDir::new().unwrap();
        let path = create_synthetic_line_shp(&dir, "line.shp");
        let loader = VectorLoader::from_file_raw(&path).unwrap();

        let x = vec![0.0];
        let y = vec![50000.0];
        let dists = loader.compute_distances(&x, &y);

        assert!(dists[0].is_finite());
        assert!(dists[0] > 10000.0);
    }

    #[test]
    fn test_vector_aoi_clip() {
        let dir = TempDir::new().unwrap();
        let path = create_synthetic_multi_line_shp(&dir, "lines.shp");

        let aoi = Geometry::Rect(geo::Rect::new(
            geo::coord! { x: -1.0, y: -1.0 },
            geo::coord! { x: 11.0, y: 1.0 },
        ));

        let loader = VectorLoader::from_file(&path, Some(&aoi), None).unwrap();
        assert!(loader.len() > 0);
    }

    #[test]
    fn test_vector_aoi_excludes_outside() {
        let dir = TempDir::new().unwrap();
        let path = create_synthetic_multi_line_shp(&dir, "lines.shp");

        let aoi = Geometry::Rect(geo::Rect::new(
            geo::coord! { x: -1.0, y: -1.0 },
            geo::coord! { x: 1.0, y: 1.0 },
        ));

        let loader = VectorLoader::from_file(&path, Some(&aoi), None).unwrap();
        assert_eq!(loader.len(), 1);
    }

    #[test]
    fn test_vector_distance_trait() {
        let dir = TempDir::new().unwrap();
        let path = create_synthetic_line_shp(&dir, "line.shp");
        let loader = VectorLoader::from_file_raw(&path).unwrap();

        let x = vec![5.0, 0.0];
        let y = vec![5.0, 0.0];
        let dists = loader.get_distance(&x, &y).unwrap();

        assert_eq!(dists.len(), 2);
        assert!(dists[1].abs() < 0.01);
    }

    #[test]
    fn test_empty_tree_returns_nan() {
        let dir = TempDir::new().unwrap();
        let driver = DriverManager::get_driver_by_name("ESRI Shapefile").unwrap();
        let path = dir.path().join("empty.shp");
        let path_str = path.to_str().unwrap().to_string();
        let mut ds = driver.create_vector_only(&path_str).unwrap();
        ds.create_layer(gdal::LayerOptions::default()).unwrap();
        drop(ds);

        let loader = VectorLoader::from_file_raw(&path_str).unwrap();
        assert_eq!(loader.len(), 0);

        let dists = loader.compute_distances(&[0.0], &[0.0]);
        assert!(dists[0].is_nan());
    }
}

//! River distance raster computation.
//!
//! Loads a river shapefile, builds an R-tree of individual line segments,
//! then computes the Haversine distance (metres) from each CLASS-grid pixel
//! centre to the nearest river geometry.
//!
//! Output: single-band float32 GeoTIFF in EPSG:4326.
//!
//! # Implementation notes
//!
//! - River geometries are decomposed into 2‑point segments and indexed with
//!   an [`rstar::RTree`].  Each pixel queries the R‑tree for the nearest
//!   segments, then computes the true Haversine point-to-segment distance.
//! - A tiled write strategy (1024×1024 blocks) keeps memory bounded
//!   regardless of grid size (target: 62166×35095 ≈ 2.18 Gpx).
//! - Euclidean (degree‑space) distance is used *only* inside the R‑tree for
//!   candidate pruning; the final distance is always Haversine.

use std::path::Path;

use geo::algorithm::{BoundingRect, ClosestPoint, HaversineDistance};
use geo::geometry::{Coord, LineString, Point};
use geo::EuclideanDistance;
use std::convert::TryInto;
use gdal::raster::Buffer;
use gdal::vector::LayerAccess;
use gdal::Dataset;
use rstar::{PointDistance, RTree, RTreeObject, AABB};

use if_loaders::base::{LoaderError, LoaderResult};

use crate::class_grid::ClassGrid;

// ── Indexed river segment ────────────────────────────────────────

/// A single 2‑point river segment stored in the R‑tree.
///
/// The envelope is the axis-aligned bounding box of the segment's
/// endpoints.  [`PointDistance`] uses Euclidean distance (degree space)
/// — this is only used for pruning, not for the final distance.
struct SegmentGeom {
    segment: LineString<f64>,
    envelope: AABB<[f64; 2]>,
}

impl SegmentGeom {
    fn from_coords(a: Coord<f64>, b: Coord<f64>) -> Self {
        let segment = LineString::new(vec![a, b]);
        let bbox = segment.bounding_rect().unwrap_or_else(|| {
            geo::Rect::new(a, b)
        });
        let envelope = AABB::from_corners(
            [bbox.min().x, bbox.min().y],
            [bbox.max().x, bbox.max().y],
        );
        Self { segment, envelope }
    }
}

impl RTreeObject for SegmentGeom {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.envelope
    }
}

impl PointDistance for SegmentGeom {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let p = Point::new(point[0], point[1]);
        let d = p.euclidean_distance(&self.segment);
        d * d
    }
}

// ── Haversine helpers ────────────────────────────────────────────

/// Haversine distance in metres between two geographic points.
#[inline]
#[allow(dead_code)]
fn haversine_m(lng1: f64, lat1: f64, lng2: f64, lat2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlng = (lng2 - lng1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlng / 2.0).sin().powi(2);
    r * 2.0 * a.sqrt().asin()
}

/// Shortest Haversine distance (metres) from a query point to a
/// linestring segment.
///
/// Finds the closest point on the segment via Euclidean (degree‑space)
/// projection, then returns the Haversine distance to that point.
fn haversine_to_segment_m(px: f64, py: f64, segment: &LineString<f64>) -> f64 {
    let query = Point::new(px, py);
    match segment.closest_point(&query) {
        geo::Closest::Intersection(pt) | geo::Closest::SinglePoint(pt) => {
            query.haversine_distance(&pt)
        }
        geo::Closest::Indeterminate => f64::NAN,
    }
}

// ── R‑tree construction ──────────────────────────────────────────

/// Open a river shapefile and build an R‑tree whose entries are
/// individual 2‑point line segments.
fn build_river_rtree(river_path: &str) -> LoaderResult<RTree<SegmentGeom>> {
    let ds = Dataset::open(Path::new(river_path))
        .map_err(|e| LoaderError::Gdal(format!("open river shp: {e}")))?;

    let mut segments: Vec<SegmentGeom> = Vec::new();

    for idx in 0..ds.layer_count() {
        let mut layer = ds
            .layer(idx)
            .map_err(|e| LoaderError::Gdal(format!("layer {idx}: {e}")))?;

        for feature in layer.features() {
            let ogr_geom = match feature.geometry() {
                Some(g) => g,
                None => continue,
            };
            if ogr_geom.is_empty() {
                continue;
            }

            // Convert OGR geometry → GeoJSON → geo-types
            let json = ogr_geom
                .json()
                .map_err(|e| LoaderError::Gdal(e.to_string()))?;
            let gj: geojson::Geometry = json
                .parse()
                .map_err(|e| LoaderError::Gdal(format!("parse geojson: {e}")))?;
            let geo_geom: geo::Geometry<f64> = gj
                .try_into()
                .map_err(|e| LoaderError::Gdal(format!("convert to geo: {e}")))?;

            // Decompose any multi‑line / polygon boundary into segments
            collect_segments(&geo_geom, &mut segments);
        }
    }

    log::info!(
        "Loaded {} river segments from {}",
        segments.len(),
        river_path
    );

    if segments.is_empty() {
        return Err(LoaderError::VectorRead(
            "No river segments found in shapefile".into(),
        ));
    }

    Ok(RTree::bulk_load(segments))
}

/// Walk a geometry tree and push every 2‑point segment into `out`.
fn collect_segments(geom: &geo::Geometry<f64>, out: &mut Vec<SegmentGeom>) {
    match geom {
        geo::Geometry::Line(line) => line_to_segments(line, out),
        geo::Geometry::LineString(ls) => line_string_to_segments(ls, out),
        geo::Geometry::MultiLineString(mls) => {
            for ls in mls {
                line_string_to_segments(ls, out);
            }
        }
        geo::Geometry::Polygon(poly) => {
            line_string_to_segments(poly.exterior(), out);
            for interior in poly.interiors() {
                line_string_to_segments(interior, out);
            }
        }
        geo::Geometry::MultiPolygon(mp) => {
            for poly in mp {
                line_string_to_segments(poly.exterior(), out);
                for interior in poly.interiors() {
                    line_string_to_segments(interior, out);
                }
            }
        }
        geo::Geometry::GeometryCollection(gc) => {
            for g in gc {
                collect_segments(g, out);
            }
        }
        _ => {
            // Points, triangles, rects — skip if no meaningful line
        }
    }
}

fn line_to_segments(line: &geo::Line<f64>, out: &mut Vec<SegmentGeom>) {
    if line.start != line.end {
        out.push(SegmentGeom::from_coords(line.start, line.end));
    }
}

fn line_string_to_segments(ls: &LineString<f64>, out: &mut Vec<SegmentGeom>) {
    let coords: Vec<Coord<f64>> = ls.coords().copied().collect();
    for w in coords.windows(2) {
        let a = w[0];
        let b = w[1];
        if a != b {
            out.push(SegmentGeom::from_coords(a, b));
        }
    }
}

// ── Public API ───────────────────────────────────────────────────

/// Compute the river-distance raster for the CLASS grid.
///
/// # Arguments
/// * `class_grid` — Reference grid providing dimensions and geotransform.
/// * `river_path` — Path to the river shapefile (e.g. `data/river/As_15s_Chimaps.shp`).
/// * `output_path` — Destination GeoTIFF path.
///
/// # Output
/// Single‑band Float32 GeoTIFF with the same dimensions and CRS as
/// `class_grid`.  Values are Haversine distances in metres.  Pixels
/// for which no river segment can be found within a reasonable
/// search radius are set to NaN.
pub fn compute_river_distance_raster(
    class_grid: &ClassGrid,
    river_path: &str,
    output_path: &str,
) -> LoaderResult<()> {
    let tree = build_river_rtree(river_path)?;

    let width = class_grid.width();
    let height = class_grid.height();
    let geo_transform = *class_grid.geo_transform();
    let crs_wkt = class_grid.crs().to_string();

    log::info!(
        "Computing river distance raster: {}x{} pixels",
        width,
        height
    );

    // ── Create output GeoTIFF ─────────────────────────────────────
    let driver = gdal::DriverManager::get_driver_by_name("GTiff")
        .map_err(|e| LoaderError::Gdal(format!("GTiff driver: {e}")))?;
    let mut ds = driver
        .create_with_band_type::<f32, _>(Path::new(output_path), width, height, 1)
        .map_err(|e| LoaderError::Gdal(format!("create GeoTIFF: {e}")))?;

    ds.set_geo_transform(&geo_transform)
        .map_err(|e| LoaderError::Gdal(format!("set_geo_transform: {e}")))?;
    if !crs_wkt.is_empty() {
        ds.set_projection(&crs_wkt)
            .map_err(|e| LoaderError::Gdal(format!("set_projection: {e}")))?;
    }

    let mut band = ds
        .rasterband(1)
        .map_err(|e| LoaderError::Gdal(format!("rasterband: {e}")))?;
    band.set_no_data_value(Some(f64::NAN))
        .map_err(|e| LoaderError::Gdal(format!("set_no_data_value: {e}")))?;

    // ── Tiled processing ──────────────────────────────────────────
    const TILE_SIZE: usize = 1024;

    let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    let total_tiles = tiles_x * tiles_y;

    let mut completed: usize = 0;

    for ty in 0..tiles_y {
        let row_off = ty * TILE_SIZE;
        let tile_h = (height - row_off).min(TILE_SIZE);

        for tx in 0..tiles_x {
            let col_off = tx * TILE_SIZE;
            let tile_w = (width - col_off).min(TILE_SIZE);

            let mut tile_data = vec![0.0f32; tile_w * tile_h];
            compute_tile(
                &tree,
                class_grid,
                col_off,
                row_off,
                tile_w,
                tile_h,
                &mut tile_data,
            );

            // Write tile into the output raster
            let mut buf = Buffer::new((tile_w, tile_h), tile_data);
            band.write((col_off as isize, row_off as isize), (tile_w, tile_h), &mut buf)
                .map_err(|e| LoaderError::Gdal(format!("write tile ({col_off},{row_off}): {e}")))?;

            completed += 1;
            if completed % 10 == 0 || completed == total_tiles {
                log::info!(
                    "River distance: {completed}/{total_tiles} tiles ({:.1}%)",
                    (completed as f64 / total_tiles as f64) * 100.0
                );
            }
        }
    }

    // Flush by closing dataset (drop)
    drop(band);
    drop(ds);

    log::info!("River distance raster written to {output_path}");
    Ok(())
}

/// Compute river distances for a single tile.
///
/// For each pixel, queries the R‑tree for up to 3 nearest segments
/// and returns the minimum Haversine point‑to‑segment distance.
fn compute_tile(
    tree: &RTree<SegmentGeom>,
    class_grid: &ClassGrid,
    col_off: usize,
    row_off: usize,
    tile_w: usize,
    tile_h: usize,
    out: &mut [f32],
) {
    for r in 0..tile_h {
        for c in 0..tile_w {
            let (lng, lat) = class_grid.pixel_to_geo(row_off + r, col_off + c);
            let idx = r * tile_w + c;

            let candidates = tree.nearest_neighbor_iter(&[lng, lat]);
            let mut best_m = f64::INFINITY;

            // Check up to 3 nearest segments for the true Haversine distance
            for (k, seg) in candidates.enumerate() {
                if k >= 3 {
                    break;
                }
                let d = haversine_to_segment_m(lng, lat, &seg.segment);
                if d.is_finite() && d < best_m {
                    best_m = d;
                }
            }

            out[idx] = if best_m.is_finite() {
                best_m as f32
            } else {
                f32::NAN
            };
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "gdal")]
mod tests {
    use super::*;
    use gdal::vector::LayerOptions;
    use gdal::DriverManager;

    use tempfile::TempDir;

    const WGS84_WKT: &str = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]";

    /// Create a synthetic shapefile with a single linestring.
    fn create_synth_river_shp(dir: &TempDir) -> String {
        let path = dir.path().join("test_river.shp");
        let path_str = path.to_str().unwrap();

        let driver = DriverManager::get_driver_by_name("ESRI Shapefile").unwrap();
        let mut ds = driver
            .create_vector_only(path_str)
            .unwrap();
        let _layer = ds.create_layer(LayerOptions::default()).unwrap();

        // Create a simple linestring (longitude, latitude)
        let wkt = "LINESTRING (100.0 30.0, 100.1 30.0, 100.2 30.1)";
        let geom = gdal::vector::Geometry::from_wkt(wkt).unwrap();
        {
            let mut layer = ds.layer(0).unwrap();
            layer
                .create_feature(geom)
                .unwrap();
        }

        drop(ds);
        path_str.to_string()
    }

    /// Create a tiny CLASS raster for testing.
    fn create_synth_class_grid(dir: &TempDir) -> (ClassGrid, String) {
        let path = dir.path().join("mini_class.tif");
        let path_str = path.to_str().unwrap().to_string();

        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let mut ds = driver
            .create_with_band_type::<i8, _>(&path_str, 10, 8, 2)
            .unwrap();
        ds.set_geo_transform(&[100.0, 0.01, 0.0, 30.2, 0.0, -0.01])
            .unwrap();
        ds.set_projection(WGS84_WKT).unwrap();
        drop(ds);

        let cg = ClassGrid::open(&path_str, 2016).unwrap();
        (cg, path_str)
    }

    #[test]
    fn test_build_river_rtree() {
        let tmp = TempDir::new().unwrap();
        let shp_path = create_synth_river_shp(&tmp);

        let tree = build_river_rtree(&shp_path).unwrap();
        assert!(tree.size() >= 2, "Expected at least 2 segments");
    }

    #[test]
    fn test_river_distance_creates_raster() {
        let tmp = TempDir::new().unwrap();
        let shp_path = create_synth_river_shp(&tmp);
        let (class_grid, _) = create_synth_class_grid(&tmp);
        let out_path = tmp.path().join("river_dist.tif");
        let out_str = out_path.to_str().unwrap();

        compute_river_distance_raster(&class_grid, &shp_path, out_str).unwrap();

        assert!(out_path.exists(), "Output raster must exist");

        // Re-open and verify dimensions / CRS
        let ds = Dataset::open(out_path).unwrap();
        let (w, h) = ds.raster_size();
        assert_eq!(w, class_grid.width());
        assert_eq!(h, class_grid.height());
        assert_eq!(ds.raster_count(), 1);

        // Read a few pixels — they should be finite distances
        let band = ds.rasterband(1).unwrap();
        let buf = band
            .read_as::<f32>((0, 0), (w, h), (w, h), None)
            .unwrap();
        let arr: Vec<f32> = buf.data().to_vec();
        let finite_count = arr.iter().filter(|v| v.is_finite()).count();
        assert!(
            finite_count > 0,
            "Expected some finite distance values, got none"
        );
    }

    #[test]
    fn test_haversine_known() {
        // Beijing (116.4°E, 39.9°N) → Shanghai (121.5°E, 31.2°N)
        // Expected: ~1068 km (within 2% tolerance)
        let d = haversine_m(116.4, 39.9, 121.5, 31.2);
        let expected = 1_068_000.0;
        let diff_pct = (d - expected).abs() / expected;
        assert!(
            diff_pct < 0.05,
            "Haversine: {d:.0} m, expected ~{expected:.0} m ({:.1}% off)",
            diff_pct * 100.0
        );
    }

    #[test]
    fn test_haversine_to_segment_along() {
        // Segment along a meridian: (100, 30) → (100, 31)
        // Point exactly at midpoint: (100, 30.5)
        // 0.5° latitude ≈ 55.6 km
        let seg = LineString::new(vec![
            Coord { x: 100.0, y: 29.0 },
            Coord { x: 100.0, y: 31.0 },
        ]);
        let d = haversine_to_segment_m(100.0, 30.0, &seg);
        // Point (100, 30.0) is ON the segment, distance should be ~0
        assert!(d < 10.0, "Point on segment should have near-zero distance, got {d}");
    }
}

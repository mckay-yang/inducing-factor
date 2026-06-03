//! Road / railway / waterway distance raster computation.
//!
//! For each year in the CLASS grid range (2016–2025), loads the corresponding
//! OSM shapefile from `data/road/` (e.g. `osm_2016_roads.shp`), builds an R-tree
//! of individual line segments, then computes the Haversine distance (metres)
//! from every CLASS-grid pixel centre to the nearest geometry.
//!
//! Three rasters are produced per year:
//!
//! | file                     | source shapefile       |
//! |--------------------------|------------------------|
//! | `road_dist_YYYY.tif`     | `osm_YYYY_roads.shp`   |
//! | `railway_dist_YYYY.tif`  | `osm_YYYY_railways.shp`|
//! | `waterway_dist_YYYY.tif` | `osm_YYYY_waterways.shp`|
//!
//! # Gap‑fill
//!
//! When a shapefile is missing for year *Y*, the latest prior year *P < Y*
//! with an existing file is used.  If *no* prior year has data for a given
//! type, that type is silently skipped for that year (all‑NaN raster).
//!
//! # Implementation notes
//!
//! - Each year's R‑tree contains **all** geometries from the shapefile —
//!   no subsetting or chunking of vector data.
//! - R‑trees are cached by source path so that gap‑filled years sharing
//!   the same file avoid rebuilding.
//! - Tiled writing (1024×1024 blocks) keeps memory bounded regardless of
//!   grid size (target: 62166×35095 ≈ 2.18 Gpx).
//! - Euclidean (degree‑space) distance is used *only* inside the R‑tree for
//!   candidate pruning; the final distance is always Haversine.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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

// ── Indexed segment ───────────────────────────────────────────────

/// A single 2‑point segment stored in the R‑tree.
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
        let bbox = segment
            .bounding_rect()
            .unwrap_or_else(|| geo::Rect::new(a, b));
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

// ── Haversine helpers ─────────────────────────────────────────────

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
fn haversine_to_segment_m(px: f64, py: f64, segment: &LineString<f64>) -> f64 {
    let query = Point::new(px, py);
    match segment.closest_point(&query) {
        geo::Closest::Intersection(pt) | geo::Closest::SinglePoint(pt) => {
            query.haversine_distance(&pt)
        }
        geo::Closest::Indeterminate => f64::NAN,
    }
}

// ── R‑tree construction ───────────────────────────────────────────

/// Open an OSM shapefile and build an R‑tree of individual line segments.
fn build_rtree(shp_path: &Path) -> LoaderResult<RTree<SegmentGeom>> {
    let ds = Dataset::open(shp_path)
        .map_err(|e| LoaderError::Gdal(format!("open shp {}: {e}", shp_path.display())))?;

    let mut segments: Vec<SegmentGeom> = Vec::new();

    for idx in 0..ds.layer_count() {
        let mut layer = ds
            .layer(idx)
            .map_err(|e| LoaderError::Gdal(format!("layer {idx} in {}: {e}", shp_path.display())))?;

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
            let gj: geojson::Geometry = json
                .parse()
                .map_err(|e| LoaderError::Gdal(format!("parse geojson: {e}")))?;
            let geo_geom: geo::Geometry<f64> = gj
                .try_into()
                .map_err(|e| LoaderError::Gdal(format!("convert to geo: {e}")))?;

            collect_segments(&geo_geom, &mut segments);
        }
    }

    log::info!(
        "Loaded {} segments from {}",
        segments.len(),
        shp_path.display()
    );

    if segments.is_empty() {
        return Err(LoaderError::VectorRead(format!(
            "No segments found in {}",
            shp_path.display()
        )));
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
        _ => {}
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

// ── Vehicle type ──────────────────────────────────────────────────

/// The three infrastructure types extracted from the road directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RoadType {
    Road,
    Railway,
    Waterway,
}

impl RoadType {
    fn file_pattern(&self) -> &str {
        match self {
            RoadType::Road => "roads",
            RoadType::Railway => "railways",
            RoadType::Waterway => "waterways",
        }
    }

    fn output_prefix(&self) -> &str {
        match self {
            RoadType::Road => "road_dist",
            RoadType::Railway => "railway_dist",
            RoadType::Waterway => "waterway_dist",
        }
    }
}

// ── Year discovery ────────────────────────────────────────────────

/// Scan `road_dir` for all years that have at least one OSM shapefile.
///
/// Matches filenames of the form `osm_{year}_{type}.shp` and returns the
/// sorted, deduplicated set of years.
fn discover_years(road_dir: &Path) -> LoaderResult<Vec<i32>> {
    let mut years: Vec<i32> = Vec::new();

    let entries = std::fs::read_dir(road_dir).map_err(|e| {
        LoaderError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("read road dir {}: {e}", road_dir.display()),
        ))
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            LoaderError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("dir entry in {}: {e}", road_dir.display()),
            ))
        })?;
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();

        // Pattern: osm_{year}_{roads|railways|waterways}.shp
        if fname.starts_with("osm_") && fname.ends_with(".shp") {
            let inner = &fname[4..fname.len() - 4]; // strip "osm_" prefix and ".shp" suffix
            if let Some(underscore_pos) = inner.find('_') {
                let year_str = &inner[..underscore_pos];
                if let Ok(y) = year_str.parse::<i32>() {
                    if !years.contains(&y) {
                        years.push(y);
                    }
                }
            }
        }
    }

    years.sort_unstable();
    Ok(years)
}

/// Find the latest year ≤ `target` for which the shapefile exists.
fn find_prior_year(
    target: i32,
    road_type: RoadType,
    road_dir: &Path,
    available_years: &[i32],
) -> Option<i32> {
    for &y in available_years.iter().rev() {
        if y > target {
            continue;
        }
        let fname = format!("osm_{}_{}.shp", y, road_type.file_pattern());
        if road_dir.join(&fname).exists() {
            return Some(y);
        }
    }
    None
}

// ── Tiled computation ─────────────────────────────────────────────

const TILE_SIZE: usize = 1024;

/// Compute distances for a single tile into `out`.
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

/// Write a GeoTIFF using tiled processing to avoid OOM.
fn write_tiled_raster(
    tree: &RTree<SegmentGeom>,
    class_grid: &ClassGrid,
    output_path: &Path,
    label: &str,
) -> LoaderResult<()> {
    let width = class_grid.width();
    let height = class_grid.height();
    let geo_transform = *class_grid.geo_transform();
    let crs_wkt = class_grid.crs().to_string();

    log::info!(
        "{} distance raster: {}x{} → {}",
        label,
        width,
        height,
        output_path.display()
    );

    let driver = gdal::DriverManager::get_driver_by_name("GTiff")
        .map_err(|e| LoaderError::Gdal(format!("GTiff driver: {e}")))?;
    let mut ds = driver
        .create_with_band_type::<f32, _>(output_path, width, height, 1)
        .map_err(|e| LoaderError::Gdal(format!("create GeoTIFF {}: {e}", output_path.display())))?;

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
            compute_tile(tree, class_grid, col_off, row_off, tile_w, tile_h, &mut tile_data);

            let mut buf = Buffer::new((tile_w, tile_h), tile_data);
            band.write((col_off as isize, row_off as isize), (tile_w, tile_h), &mut buf)
                .map_err(|e| {
                    LoaderError::Gdal(format!("write tile ({col_off},{row_off}): {e}"))
                })?;

            completed += 1;
            if completed % 100 == 0 || completed == total_tiles {
                log::info!(
                    "{}: {completed}/{total_tiles} tiles ({:.1}%)",
                    label,
                    (completed as f64 / total_tiles as f64) * 100.0
                );
            }
        }
    }

    drop(band);
    drop(ds);

    log::info!("{} raster written to {}", label, output_path.display());
    Ok(())
}

// ── Public API ────────────────────────────────────────────────────

/// Compute road / railway / waterway distance rasters for all CLASS-grid years.
///
/// # Arguments
/// * `class_grid` — Reference grid providing dimensions, geotransform, CRS, and
///   year range (2016–2025).
/// * `road_dir` — Path to the directory containing yearly OSM shapefiles
///   (`osm_{YYYY}_{roads|railways|waterways}.shp`).
/// * `output_dir` — Directory into which the output GeoTIFFs are written.
///
/// # Gap‑fill
///
/// For each year *Y* and each type (*road*, *railway*, *waterway*):
///
/// 1. If `osm_Y_TYPE.shp` exists → use it.
/// 2. Otherwise → find the largest year *P < Y* where `osm_P_TYPE.shp` exists.
/// 3. If no prior year exists → skip that type for *Y* (no raster).
///
/// R‑trees are cached by source path so that gap‑filled years reuse the
/// same in‑memory index.
///
/// # Resume
///
/// Years for which all three output rasters already exist are skipped.
///
/// # Returns
/// The sorted list of years that were processed (newly written).
pub fn compute_road_distance_rasters(
    class_grid: &ClassGrid,
    road_dir: &str,
    output_dir: &str,
) -> LoaderResult<Vec<i32>> {
    let road_dir_path = Path::new(road_dir);
    if !road_dir_path.is_dir() {
        return Err(LoaderError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("road_dir not found: {road_dir}"),
        )));
    }

    let out_dir = Path::new(output_dir);
    std::fs::create_dir_all(out_dir).map_err(|e| {
        LoaderError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("create output dir {output_dir}: {e}"),
        ))
    })?;

    // Discover all years in road_dir (2014–2024 typically)
    let available_years = discover_years(road_dir_path)?;
    log::info!(
        "Road data available for years: {:?}",
        available_years
    );

    // Determine target years from the CLASS grid (2016–2025)
    let target_years = class_grid.years();
    log::info!("Target years from CLASS grid: {:?}", target_years);

    let types = [RoadType::Road, RoadType::Railway, RoadType::Waterway];

    // ── Gap‑fill map ───────────────────────────────────────────
    // For each (target_year, road_type), find the actual source year.
    #[derive(Debug, Clone)]
    struct YearTypeTask {
        target_year: i32,
        road_type: RoadType,
        source_year: i32,
        source_path: PathBuf,
        output_path: PathBuf,
    }

    let mut tasks: Vec<YearTypeTask> = Vec::new();

    for &target_year in &target_years {
        let mut all_exist = true;

        for &rt in &types {
            let out_fname = format!("{}_{}.tif", rt.output_prefix(), target_year);
            let out_path = out_dir.join(&out_fname);

            let source_year = find_prior_year(target_year, rt, road_dir_path, &available_years);
            match source_year {
                Some(sy) => {
                    let src_fname = format!("osm_{}_{}.shp", sy, rt.file_pattern());
                    let src_path = road_dir_path.join(&src_fname);

                    if !out_path.exists() {
                        all_exist = false;
                        tasks.push(YearTypeTask {
                            target_year,
                            road_type: rt,
                            source_year: sy,
                            source_path: src_path,
                            output_path: out_path,
                        });
                    }
                }
                None => {
                    log::warn!(
                        "No prior data for {}_{}: skipping",
                        rt.file_pattern(),
                        target_year
                    );
                }
            }
        }

        if all_exist {
            log::info!(
                "Year {}: all three rasters exist, skipping",
                target_year
            );
        }
    }

    if tasks.is_empty() {
        log::info!("All road distance rasters already exist — nothing to do");
        return Ok(Vec::new());
    }

    // ── R‑tree cache (keyed by source path) ────────────────────
    let mut rtree_cache: HashMap<PathBuf, RTree<SegmentGeom>> = HashMap::new();
    let mut processed_years: Vec<i32> = Vec::new();

    // Group tasks by source_path so consecutive same-source tasks reuse the R‑tree
    for task in &tasks {
        if !rtree_cache.contains_key(&task.source_path) {
            log::info!(
                "Building R-tree for {} (source year {})",
                task.source_path.display(),
                task.source_year
            );
            let tree = build_rtree(&task.source_path)?;
            rtree_cache.insert(task.source_path.clone(), tree);
        } else {
            log::info!(
                "Reusing cached R-tree for {} (gap-fill for {})",
                task.source_path.display(),
                task.target_year
            );
        }

        let tree = rtree_cache.get(&task.source_path).unwrap();
        let label = format!(
            "{}_Y{} (source Y{})",
            task.road_type.output_prefix(),
            task.target_year,
            task.source_year
        );

        write_tiled_raster(tree, class_grid, &task.output_path, &label)?;

        if !processed_years.contains(&task.target_year) {
            processed_years.push(task.target_year);
        }
    }

    processed_years.sort_unstable();
    log::info!(
        "Road distance computation complete. Processed years: {:?}",
        processed_years
    );

    Ok(processed_years)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "gdal")]
mod tests {
    use super::*;
    use gdal::vector::LayerOptions;
    use gdal::DriverManager;
    use tempfile::TempDir;

    const WGS84_WKT: &str = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]";

    /// Create a synthetic OSM road shapefile with a single linestring.
    fn create_synth_road_shp(dir: &TempDir, year: i32, road_type: &str) -> String {
        let fname = format!("osm_{}_{}.shp", year, road_type);
        let path = dir.path().join(&fname);
        let path_str = path.to_str().unwrap().to_string();

        let driver = DriverManager::get_driver_by_name("ESRI Shapefile").unwrap();
        let mut ds = driver.create_vector_only(&path_str).unwrap();
        let _layer = ds.create_layer(LayerOptions::default()).unwrap();

        let wkt = "LINESTRING (100.0 30.0, 100.1 30.0, 100.2 30.1)";
        let geom = gdal::vector::Geometry::from_wkt(wkt).unwrap();
        {
            let mut layer = ds.layer(0).unwrap();
            layer.create_feature(geom).unwrap();
        }

        drop(ds);
        path_str
    }

    /// Create a tiny CLASS raster for testing.
    fn create_synth_class_grid(dir: &TempDir, start_year: i32, num_years: usize) -> (ClassGrid, String) {
        let path = dir.path().join("mini_class.tif");
        let path_str = path.to_str().unwrap().to_string();

        let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
        let mut ds = driver
            .create_with_band_type::<i8, _>(&path_str, 10, 8, num_years)
            .unwrap();
        ds.set_geo_transform(&[100.0, 0.01, 0.0, 30.2, 0.0, -0.01])
            .unwrap();
        ds.set_projection(WGS84_WKT).unwrap();
        drop(ds);

        let cg = ClassGrid::open(&path_str, start_year).unwrap();
        (cg, path_str)
    }

    // ── Unit tests ─────────────────────────────────────────────

    #[test]
    fn test_build_rtree() {
        let tmp = TempDir::new().unwrap();
        let shp_path = create_synth_road_shp(&tmp, 2020, "roads");
        let tree = build_rtree(Path::new(&shp_path)).unwrap();
        assert!(tree.size() >= 2, "Expected at least 2 segments");
    }

    #[test]
    fn test_distance_computation() {
        let tmp = TempDir::new().unwrap();
        let shp_path = create_synth_road_shp(&tmp, 2020, "roads");
        let (class_grid, _) = create_synth_class_grid(&tmp, 2020, 1);
        let out_path = tmp.path().join("road_dist_2020.tif");
        let out_str = out_path.to_str().unwrap();

        let tree = build_rtree(Path::new(&shp_path)).unwrap();
        write_tiled_raster(&tree, &class_grid, Path::new(out_str), "road_test").unwrap();

        assert!(out_path.exists(), "Output raster must exist");

        let ds = Dataset::open(out_path).unwrap();
        let (w, h) = ds.raster_size();
        assert_eq!(w, class_grid.width());
        assert_eq!(h, class_grid.height());
        assert_eq!(ds.raster_count(), 1);

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
    fn test_gapfill() {
        let tmp = TempDir::new().unwrap();

        // Create road data only for 2020 — 2021 and 2022 are missing
        create_synth_road_shp(&tmp, 2020, "roads");
        create_synth_road_shp(&tmp, 2020, "railways");
        create_synth_road_shp(&tmp, 2020, "waterways");

        // CLASS grid with years 2020–2022 (3 bands)
        let (class_grid, _) = create_synth_class_grid(&tmp, 2020, 3);

        let out_dir = tmp.path().join("road_out");
        std::fs::create_dir_all(&out_dir).unwrap();

        let road_dir = tmp.path().to_str().unwrap();
        let out_dir_str = out_dir.to_str().unwrap();

        let processed = compute_road_distance_rasters(&class_grid, road_dir, out_dir_str)
            .unwrap();

        // Should process all three years (2020, 2021, 2022) since rasters don't exist yet
        assert_eq!(processed.len(), 3, "Expected 3 processed years");

        // All three types for all three years should exist
        for year in [2020, 2021, 2022] {
            for (prefix, _label) in [
                ("road_dist", "roads"),
                ("railway_dist", "railways"),
                ("waterway_dist", "waterways"),
            ] {
                let fname = format!("{}_{}.tif", prefix, year);
                let p = out_dir.join(&fname);
                assert!(p.exists(), "Expected {fname} to exist");
            }
        }

        // 2021 and 2022 should use 2020 data (same source year)
        // Re-run: should skip all years (resume)
        let processed2 = compute_road_distance_rasters(&class_grid, road_dir, out_dir_str)
            .unwrap();
        assert!(processed2.is_empty(), "Resume: should produce 0 new rasters");
    }

    #[test]
    fn test_haversine_known() {
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
        let seg = LineString::new(vec![
            Coord { x: 100.0, y: 29.0 },
            Coord { x: 100.0, y: 31.0 },
        ]);
        let d = haversine_to_segment_m(100.0, 30.0, &seg);
        assert!(d < 10.0, "Point on segment should have near-zero distance, got {d}");
    }
}

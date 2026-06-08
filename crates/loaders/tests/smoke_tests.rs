#[cfg(all(feature = "real-data", feature = "gdal"))]
mod real_data_smoke {
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        std::env::var("DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data"))
    }

    fn should_run_real_tests() -> bool {
        std::env::var("RUN_REAL_DATA_TESTS")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    }

    fn check_file(path: &std::path::Path) -> bool {
        if !path.exists() {
            eprintln!("SKIP: file not found: {}", path.display());
            false
        } else {
            true
        }
    }

    #[test]
    fn dem_loader_smoke_open_resolution() {
        if !should_run_real_tests() {
            eprintln!("SKIP: set RUN_REAL_DATA_TESTS=1 to enable");
            return;
        }
        let dem_path = data_dir().join("dem").join("tibet_dem.tif");
        if !check_file(&dem_path) {
            return;
        }
        let loader = loaders::dem::DEMLoader::open(&dem_path.to_string_lossy())
            .expect("Failed to open DEM file");
        let (res_x, res_y) = loader.resolution();
        assert!(res_x > 0.0, "resolution_x = {res_x}");
        assert!(res_y > 0.0, "resolution_y = {res_y}");
    }

    #[test]
    fn dem_loader_smoke_read_window() {
        if !should_run_real_tests() {
            eprintln!("SKIP: set RUN_REAL_DATA_TESTS=1 to enable");
            return;
        }
        let dem_path = data_dir().join("dem").join("tibet_dem.tif");
        if !check_file(&dem_path) {
            return;
        }
        let loader = loaders::dem::DEMLoader::open(&dem_path.to_string_lossy())
            .expect("Failed to open DEM file");
        let window = loaders::base::Window::new(0, 0, 128, 128);
        let elev = loader
            .read_elevation(&window)
            .expect("Failed to read window");
        assert_eq!(elev.dim(), (128, 128));
    }

    #[test]
    fn ccdc_loader_smoke_discover_years() {
        if !should_run_real_tests() {
            eprintln!("SKIP: set RUN_REAL_DATA_TESTS=1 to enable");
            return;
        }
        let ccdc_dir = data_dir().join("ccdc_res");
        if !ccdc_dir.exists() {
            eprintln!("SKIP: CCDC directory not found: {}", ccdc_dir.display());
            return;
        }
        let loader = loaders::ccdc::CcdcLoader::new_default(&ccdc_dir.to_string_lossy());
        let years = loader.discover_years();
        assert!(!years.is_empty(), "CCDC should discover at least one year");
    }

    #[test]
    fn vector_loader_smoke_open() {
        if !should_run_real_tests() {
            eprintln!("SKIP: set RUN_REAL_DATA_TESTS=1 to enable");
            return;
        }
        let road_dir = data_dir().join("road");
        if !road_dir.exists() {
            eprintln!("SKIP: road directory not found: {}", road_dir.display());
            return;
        }
        let road_path = road_dir.join("osm_2016_roads.shp");
        if !check_file(&road_path) {
            return;
        }
        let loader = loaders::vector::VectorLoader::from_file_raw(&road_path.to_string_lossy())
            .expect("Failed to open road shapefile");
        assert!(loader.len() > 0, "road shapefile should contain features");
    }

    #[test]
    fn landuse_loader_smoke_construct() {
        if !should_run_real_tests() {
            eprintln!("SKIP: set RUN_REAL_DATA_TESTS=1 to enable");
            return;
        }
        let landuse_dir = data_dir().join("landuse");
        if !landuse_dir.exists() {
            eprintln!(
                "SKIP: landuse directory not found: {}",
                landuse_dir.display()
            );
            return;
        }
        let _loader = loaders::landuse::LandUseLoader::new_temporal(
            &landuse_dir.to_string_lossy(),
            "CLCD_v01_{year}_albert.tif",
        );
    }
}

#[cfg(not(feature = "real-data"))]
#[test]
fn real_data_tests_require_feature() {
    eprintln!("SKIP: compile with --features real-data,gdal to enable loader smoke tests");
}

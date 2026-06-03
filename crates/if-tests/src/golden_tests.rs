use if_io::artifacts::*;
use polars::prelude::*;
use serde::de::DeserializeOwned;

use crate::fixtures::golden_path;

pub fn load_golden_json<T: DeserializeOwned>(name: &str) -> T {
    let path = golden_path(name);
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read golden fixture {}: {}", path.display(), e));
    let sanitized = sanitize_json_infinities(&content);
    serde_json::from_str(&sanitized)
        .unwrap_or_else(|e| panic!("Failed to parse golden fixture {}: {}", path.display(), e))
}

fn sanitize_json_infinities(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if chars[i] == '-' && i + 8 < n && &raw[i..i + 9] == "-Infinity" {
            out.push_str("\"-Infinity\"");
            i += 9;
        } else if i + 8 < n && &raw[i..i + 8] == "Infinity" {
            out.push_str("\"Infinity\"");
            i += 8;
        } else if i + 3 < n && &raw[i..i + 3] == "NaN" {
            out.push_str("\"NaN\"");
            i += 3;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub fn load_golden_parquet(name: &str) -> DataFrame {
    let path = golden_path(name);
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open parquet file {}: {}", path.display(), e));
    ParquetReader::new(file)
        .finish()
        .unwrap_or_else(|e| panic!("Failed to read parquet file {}: {}", path.display(), e))
}

pub fn manifest() -> Manifest {
    load_golden_json("MANIFEST.json")
}

pub fn load_factor_library_raw() -> FactorLibraryRaw {
    load_golden_json("factor_library.json")
}

pub fn load_model_metadata_raw() -> ModelMetadataRaw {
    load_golden_json("model_metadata.json")
}

pub fn load_bayesian_network_edges_raw() -> BayesianNetworkEdgesRaw {
    load_golden_json("bayesian_network_edges.json")
}

pub fn load_discretization_labels_raw() -> DiscretizationLabelsRaw {
    load_golden_json("discretization_labels.json")
}

pub fn load_selected_factors_raw() -> SelectedFactorsRaw {
    load_golden_json("selected_factors.json")
}

pub fn load_model_predictions_raw() -> ModelPredictionsRaw {
    let df = load_golden_parquet("model_predictions.parquet");
    parquet_to_predictions(&df)
}

fn parquet_to_predictions(df: &DataFrame) -> ModelPredictionsRaw {
    let n = df.height();
    let mut rows = Vec::with_capacity(n);

    let point_id = df.column("point_id").ok();
    let x = df.column("x").ok();
    let y = df.column("y").ok();
    let year = df.column("year").ok();
    let dist_prob = df.column("disturbance_probability").ok();
    let cp1 = df.column("class_probability_1").ok();
    let cp3 = df.column("class_probability_3").ok();
    let cp4 = df.column("class_probability_4").ok();
    let dom_class = df.column("dominant_disturbance_class").ok();
    let pfa = df.column("prob_fire_abs").ok();
    let pgha = df.column("prob_geo_hazard_abs").ok();
    let pra = df.column("prob_road_abs").ok();

    fn col_i64(col: Option<&Column>, i: usize) -> i64 {
        col.and_then(|c| c.i64().ok())
            .and_then(|s| s.get(i))
            .unwrap_or(0)
    }
    fn col_f64(col: Option<&Column>, i: usize) -> f64 {
        col.and_then(|c| c.f64().ok())
            .and_then(|s| s.get(i))
            .unwrap_or(f64::NAN)
    }

    for i in 0..n {
        rows.push(ModelPredictionRow {
            point_id: col_i64(point_id, i),
            x: col_f64(x, i),
            y: col_f64(y, i),
            year: col_i64(year, i),
            disturbance_probability: col_f64(dist_prob, i),
            class_probability_1: col_f64(cp1, i),
            class_probability_3: col_f64(cp3, i),
            class_probability_4: col_f64(cp4, i),
            dominant_disturbance_class: col_i64(dom_class, i),
            prob_fire_abs: col_f64(pfa, i),
            prob_geo_hazard_abs: col_f64(pgha, i),
            prob_road_abs: col_f64(pra, i),
        });
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_path_resolves_correctly() {
        let p = golden_path("MANIFEST.json");
        assert!(p.exists(), "Expected {} to exist", p.display());
        assert!(p.ends_with("tests/golden/MANIFEST.json"));
    }

    #[test]
    fn load_golden_json_manifest() {
        let m = manifest();
        assert_eq!(m.random_seed, 42);
        assert!(!m.description.is_empty());
        assert!(!m.pipeline_parameters.discretizer_method.is_empty());
    }

    #[test]
    fn load_golden_parquet_predictions() {
        let df = load_golden_parquet("model_predictions.parquet");
        assert!(df.height() > 0, "Expected non-empty dataframe");
        let expected_cols = vec![
            "point_id",
            "x",
            "y",
            "year",
            "disturbance_probability",
            "class_probability_1",
            "class_probability_3",
            "class_probability_4",
            "dominant_disturbance_class",
            "prob_fire_abs",
            "prob_geo_hazard_abs",
            "prob_road_abs",
        ];
        for col in &expected_cols {
            assert!(
                df.column(col).is_ok(),
                "Expected column '{}' in model_predictions.parquet",
                col
            );
        }
    }

    #[test]
    fn load_factor_library_from_golden() {
        let factors = load_factor_library_raw();
        assert!(!factors.is_empty());
        assert_eq!(&factors[0].factor_id, "IF_001");
    }

    #[test]
    fn load_discretization_labels_from_golden() {
        let labels = load_discretization_labels_raw();
        assert!(labels.contains_key("slope"));
        assert!(labels.contains_key("dist_to_river"));
        assert!(labels.contains_key("rainfall_prev_2m"));
    }

    #[test]
    fn load_bayesian_network_edges_from_golden() {
        let edges = load_bayesian_network_edges_raw();
        assert!(!edges.is_empty());
        assert!(edges.iter().any(|(_, to)| to == "disturbance_flag"));
    }

    #[test]
    fn load_model_predictions_raw_from_golden() {
        let preds = load_model_predictions_raw();
        assert!(!preds.is_empty());
        for p in &preds {
            assert!(
                p.disturbance_probability >= 0.0 && p.disturbance_probability <= 1.0,
                "probability out of range: {}",
                p.disturbance_probability
            );
        }
    }

    #[test]
    fn manifest_tolerances_are_valid() {
        let m = manifest();
        assert_eq!(m.tolerances.discretization_labels, "exact");
        let occ = m.tolerances.occurrence_probabilities.unwrap();
        assert!(occ.mae_max > 0.0);
        assert!(occ.max_error > 0.0);
        let rankings = m.tolerances.factor_rankings.unwrap();
        assert!(rankings.spearman_min > 0.0);
    }
}

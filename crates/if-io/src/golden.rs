use crate::artifacts::*;
use serde::Serialize;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GoldenError {
    #[error("discretization mismatch for feature '{feature}': field '{field}' expected '{expected}', got '{actual}'")]
    DiscretizationMismatch {
        feature: String,
        field: String,
        expected: String,
        actual: String,
    },

    #[error("statistics tolerance exceeded for metric '{metric}': value {actual} vs expected {expected} (abs_diff={abs_diff}, tol={tolerance})")]
    StatisticsTolerance {
        metric: String,
        actual: f64,
        expected: f64,
        abs_diff: f64,
        tolerance: f64,
    },

    #[error("probability tolerance exceeded for sample {point_id}: {detail}")]
    ProbabilityTolerance { point_id: i64, detail: String },

    #[error("raster metadata mismatch: {detail}")]
    RasterMismatch { detail: String },
}

pub type TolError = GoldenError;
pub type DiffError = GoldenError;

/// A single comparison result (used for per-metric reporting).
#[derive(Debug, Clone)]
pub enum MetricResult {
    Pass {
        name: String,
        actual: f64,
        expected: f64,
        tolerance: f64,
    },
    Fail {
        name: String,
        actual: f64,
        expected: f64,
        tolerance: f64,
        abs_diff: f64,
    },
}

impl MetricResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, MetricResult::Pass { .. })
    }

    pub fn name(&self) -> &str {
        match self {
            MetricResult::Pass { name, .. } => name,
            MetricResult::Fail { name, .. } => name,
        }
    }
}

/// Detailed report from a golden comparison run.
#[derive(Debug, Clone)]
pub struct GoldenReport {
    pub overall_pass: bool,
    pub results: Vec<MetricResult>,
    pub spearman_rho: Option<f64>,
    pub spearman_p_value: Option<f64>,
}

impl GoldenReport {
    pub fn print(&self) {
        println!("=== Golden Comparison Report ===");
        println!(
            "Overall: {}",
            if self.overall_pass { "PASS" } else { "FAIL" }
        );
        for r in &self.results {
            match r {
                MetricResult::Pass {
                    name,
                    actual,
                    expected,
                    tolerance,
                } => {
                    println!(
                        "  [PASS] {name}: actual={actual:.6e}, expected={expected:.6e}, tol={tolerance:.1e}",
                    );
                }
                MetricResult::Fail {
                    name,
                    actual,
                    expected,
                    tolerance,
                    abs_diff,
                } => {
                    println!(
                        "  [FAIL] {name}: actual={actual:.6e}, expected={expected:.6e}, tol={tolerance:.1e}, diff={abs_diff:.1e}",
                    );
                }
            }
        }
        if let Some(rho) = self.spearman_rho {
            println!(
                "  Spearman ρ = {:.6}, p = {:.6e}",
                rho,
                self.spearman_p_value.unwrap_or(f64::NAN)
            );
        }
    }
}

// ── public comparison functions ────────────────────────────────────────────

/// Exact comparison of discretization labels. Every feature, bin edge, and
/// label is checked byte-for-byte.
pub fn compare_discretization_labels(
    actual: &HashMap<String, FeatureDiscretization>,
    golden: &HashMap<String, FeatureDiscretization>,
) -> Result<(), Vec<DiffError>> {
    let mut errors: Vec<DiffError> = Vec::new();

    // Check for missing/extra features
    let actual_keys: std::collections::BTreeSet<_> = actual.keys().collect();
    let golden_keys: std::collections::BTreeSet<_> = golden.keys().collect();

    for k in golden_keys.difference(&actual_keys) {
        errors.push(GoldenError::DiscretizationMismatch {
            feature: k.to_string(),
            field: "existence".into(),
            expected: "present".into(),
            actual: "missing".into(),
        });
    }
    for k in actual_keys.difference(&golden_keys) {
        errors.push(GoldenError::DiscretizationMismatch {
            feature: k.to_string(),
            field: "existence".into(),
            expected: "absent".into(),
            actual: "present".into(),
        });
    }

    // Per-feature comparison
    for (feat, golden_fd) in golden {
        if let Some(actual_fd) = actual.get(feat) {
            if actual_fd.kind != golden_fd.kind {
                errors.push(GoldenError::DiscretizationMismatch {
                    feature: feat.clone(),
                    field: "kind".into(),
                    expected: golden_fd.kind.clone(),
                    actual: actual_fd.kind.clone(),
                });
            }
            if actual_fd.categories != golden_fd.categories {
                errors.push(GoldenError::DiscretizationMismatch {
                    feature: feat.clone(),
                    field: "categories".into(),
                    expected: format!("{:?}", &golden_fd.categories),
                    actual: format!("{:?}", &actual_fd.categories),
                });
            }
            if actual_fd.labels != golden_fd.labels {
                errors.push(GoldenError::DiscretizationMismatch {
                    feature: feat.clone(),
                    field: "labels".into(),
                    expected: format!("{:?}", &golden_fd.labels),
                    actual: format!("{:?}", &actual_fd.labels),
                });
            }
            match (&actual_fd.bins, &golden_fd.bins) {
                (Some(a_bins), Some(g_bins)) => {
                    if a_bins.len() != g_bins.len() {
                        errors.push(GoldenError::DiscretizationMismatch {
                            feature: feat.clone(),
                            field: "bins.len".into(),
                            expected: g_bins.len().to_string(),
                            actual: a_bins.len().to_string(),
                        });
                    } else {
                        for (i, (a, g)) in a_bins.iter().zip(g_bins.iter()).enumerate() {
                            let both_inf = a.is_infinite()
                                && g.is_infinite()
                                && a.is_sign_positive() == g.is_sign_positive();
                            let both_nan = a.is_nan() && g.is_nan();
                            if !both_inf && !both_nan && (a - g).abs() > 1e-15 {
                                errors.push(GoldenError::DiscretizationMismatch {
                                    feature: feat.clone(),
                                    field: format!("bins[{i}]"),
                                    expected: format!("{:.15}", g),
                                    actual: format!("{:.15}", a),
                                });
                            }
                        }
                    }
                }
                (None, None) => {}
                (a, g) => {
                    errors.push(GoldenError::DiscretizationMismatch {
                        feature: feat.clone(),
                        field: "bins".into(),
                        expected: format!("{:?}", g),
                        actual: format!("{:?}", a),
                    });
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Compare statistical metrics (relative risk, confidence intervals, etc.)
/// with tolerance checks. Tolerance is an absolute threshold.
pub fn compare_statistics(
    actual: &[FactorEntry],
    golden: &[FactorEntry],
    tol: f64,
) -> Result<(), Vec<TolError>> {
    let mut errors: Vec<TolError> = Vec::new();

    if actual.len() != golden.len() {
        errors.push(GoldenError::StatisticsTolerance {
            metric: "factor_count".into(),
            actual: actual.len() as f64,
            expected: golden.len() as f64,
            abs_diff: (actual.len() as f64 - golden.len() as f64).abs(),
            tolerance: 0.0,
        });
        return Err(errors);
    }

    for (i, (a, g)) in actual.iter().zip(golden.iter()).enumerate() {
        let metrics: Vec<(&str, f64, f64)> = vec![
            ("support_event", a.support_event, g.support_event),
            (
                "support_background",
                a.support_background,
                g.support_background,
            ),
            ("relative_risk", a.relative_risk, g.relative_risk),
            ("stability", a.stability, g.stability),
            ("score", a.score, g.score),
            ("rr_ci_lower", a.rr_ci_lower, g.rr_ci_lower),
            ("rr_ci_upper", a.rr_ci_upper, g.rr_ci_upper),
            ("p_value", a.p_value, g.p_value),
        ];
        for (name, av, gv) in &metrics {
            let abs_diff = (av - gv).abs();
            if abs_diff > tol {
                errors.push(GoldenError::StatisticsTolerance {
                    metric: format!("[{i}].{name}"),
                    actual: *av,
                    expected: *gv,
                    abs_diff,
                    tolerance: tol,
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Compare occurrence probabilities: checks both MAE and maximum error
/// against permissive thresholds.
pub fn compare_probabilities(
    actual: &[ModelPredictionRow],
    golden: &[ModelPredictionRow],
    mae_max: f64,
    max_error: f64,
) -> Result<(), Vec<TolError>> {
    let mut errors: Vec<TolError> = Vec::new();

    if actual.len() != golden.len() {
        errors.push(GoldenError::ProbabilityTolerance {
            point_id: 0i64,
            detail: format!(
                "row count mismatch: actual={}, golden={}",
                actual.len(),
                golden.len()
            ),
        });
        return Err(errors);
    }

    let n = actual.len() as f64;
    let mut sum_abs_err = 0.0f64;
    let mut max_err = 0.0f64;

    for (_i, (a, g)) in actual.iter().zip(golden.iter()).enumerate() {
        let abs_err = (a.disturbance_probability - g.disturbance_probability).abs();
        sum_abs_err += abs_err;
        if abs_err > max_err {
            max_err = abs_err;
        }
        if abs_err > max_error {
            errors.push(GoldenError::ProbabilityTolerance {
                point_id: a.point_id,
                detail: format!(
                    "max error exceeded: actual={:.8}, golden={:.8}, diff={:.2e} > {:.2e}",
                    a.disturbance_probability, g.disturbance_probability, abs_err, max_error
                ),
            });
        }
    }

    let mae = sum_abs_err / n;
    if mae > mae_max {
        errors.push(GoldenError::ProbabilityTolerance {
            point_id: 0i64,
            detail: format!(
                "MAE exceeded: mae={:.6e} > {:.6e} (max_error={:.6e})",
                mae, mae_max, max_err
            ),
        });
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Compare factor rankings using Spearman rank correlation.
/// Returns (rho, p_value). Rho near 1.0 means strong agreement.
pub fn compare_factor_rankings(actual: &[FactorEntry], golden: &[FactorEntry]) -> (f64, f64) {
    let n = actual.len();
    if n == 0 {
        return (1.0, 1.0);
    }

    // Map factor_id to its position (rank) in each list
    let rank_actual: HashMap<&str, usize> = actual
        .iter()
        .enumerate()
        .map(|(i, f)| (f.factor_id.as_str(), i))
        .collect();
    let rank_golden: HashMap<&str, usize> = golden
        .iter()
        .enumerate()
        .map(|(i, f)| (f.factor_id.as_str(), i))
        .collect();

    // Build paired ranks for factors present in both
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for fid in rank_actual.keys() {
        if let Some(g_rank) = rank_golden.get(fid) {
            pairs.push((rank_actual[fid], *g_rank));
        }
    }

    if pairs.len() < 2 {
        return (0.0, 1.0);
    }

    let m = pairs.len() as f64;
    let mut d2_sum = 0.0f64;
    for (a, g) in &pairs {
        let d = (*a as f64) - (*g as f64);
        d2_sum += d * d;
    }

    let rho = 1.0 - (6.0 * d2_sum) / (m * (m * m - 1.0));
    let p = spearman_p_value(rho, m);

    (rho, p.clamp(0.0, 1.0))
}

/// Simple t-distribution CDF approximation for Spearman p-value.
/// Uses the relationship: rho * sqrt((n-2)/(1-rho^2)) ~ t(n-2).
fn spearman_p_value(rho: f64, n: f64) -> f64 {
    if n < 3.0 || rho.abs() >= 1.0 {
        return if rho.abs() >= 1.0 { 0.0 } else { 1.0 };
    }
    let t = rho * ((n - 2.0) / (1.0 - rho * rho + 1e-15)).sqrt();
    // Use normal approximation
    normal_sf(t.abs()) * 2.0
}

/// Standard normal survival function (1 - CDF).
fn normal_sf(x: f64) -> f64 {
    // Abramowitz & Stegun approximation (max error < 7.5e-8)
    let p = 0.2316419;
    let b1 = 0.319381530;
    let b2 = -0.356563782;
    let b3 = 1.781477937;
    let b4 = -1.821255978;
    let b5 = 1.330274429;
    let t = 1.0 / (1.0 + p * x);
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * (b1 * t + b2 * t * t + b3 * t * t * t + b4 * t * t * t * t + b5 * t * t * t * t * t);
    // sf = 1 - cdf ≈ upper tail
    (1.0 - cdf).max(0.0)
}

/// Exact comparison for raster metadata (CRS, dimensions, transform).
pub fn compare_raster_metadata(
    actual: &RasterMetadata,
    golden: &RasterMetadata,
) -> Result<(), Vec<DiffError>> {
    let mut errors: Vec<DiffError> = Vec::new();

    if actual.crs != golden.crs {
        errors.push(GoldenError::RasterMismatch {
            detail: format!(
                "CRS mismatch: expected '{}', got '{}'",
                golden.crs, actual.crs
            ),
        });
    }
    if actual.width != golden.width {
        errors.push(GoldenError::RasterMismatch {
            detail: format!(
                "width mismatch: expected {}, got {}",
                golden.width, actual.width
            ),
        });
    }
    if actual.height != golden.height {
        errors.push(GoldenError::RasterMismatch {
            detail: format!(
                "height mismatch: expected {}, got {}",
                golden.height, actual.height
            ),
        });
    }
    if actual.nodata != golden.nodata {
        errors.push(GoldenError::RasterMismatch {
            detail: format!(
                "nodata mismatch: expected {:?}, got {:?}",
                golden.nodata, actual.nodata
            ),
        });
    }
    // Compare transforms element-wise
    if let (Some(a_t), Some(g_t)) = (&actual.transform, &golden.transform) {
        if a_t.len() == g_t.len() {
            for (i, (a, g)) in a_t.iter().zip(g_t.iter()).enumerate() {
                if (a - g).abs() > 1e-15 {
                    errors.push(GoldenError::RasterMismatch {
                        detail: format!("transform[{i}] mismatch: expected {g}, got {a}"),
                    });
                }
            }
        } else {
            errors.push(GoldenError::RasterMismatch {
                detail: format!(
                    "transform length mismatch: expected {}, got {}",
                    g_t.len(),
                    a_t.len()
                ),
            });
        }
    } else {
        let actual_has = actual.transform.is_some();
        let golden_has = golden.transform.is_some();
        if actual_has != golden_has {
            errors.push(GoldenError::RasterMismatch {
                detail: format!(
                    "transform presence mismatch: actual={actual_has}, golden={golden_has}"
                ),
            });
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ── Raster metadata type ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RasterMetadata {
    pub crs: String,
    pub width: u32,
    pub height: u32,
    pub nodata: Option<f64>,
    pub transform: Option<Vec<f64>>,
}

// ── Golden fixture abstraction ─────────────────────────────────────────────

/// A loaded golden fixture ready for comparison.
#[derive(Debug)]
pub struct GoldenFixture {
    pub name: String,
    pub meta: Option<Manifest>,
}

/// Orchestrates comparisons between Rust outputs and golden fixtures.
#[derive(Debug, Default)]
pub struct GoldenComparator {
    results: Vec<MetricResult>,
    spearman_rho: Option<f64>,
    spearman_p_value: Option<f64>,
}

impl GoldenComparator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_f64(&mut self, name: &str, actual: f64, expected: f64, tolerance: f64) {
        let abs_diff = (actual - expected).abs();
        if abs_diff <= tolerance {
            self.results.push(MetricResult::Pass {
                name: name.to_string(),
                actual,
                expected,
                tolerance,
            });
        } else {
            self.results.push(MetricResult::Fail {
                name: name.to_string(),
                actual,
                expected,
                tolerance,
                abs_diff,
            });
        }
    }

    pub fn record_spearman(&mut self, rho: f64, p_value: f64) {
        self.spearman_rho = Some(rho);
        self.spearman_p_value = Some(p_value);
    }

    pub fn report(&self) -> GoldenReport {
        GoldenReport {
            overall_pass: self.results.iter().all(|r| r.is_pass()),
            results: self.results.clone(),
            spearman_rho: self.spearman_rho,
            spearman_p_value: self.spearman_p_value,
        }
    }

    pub fn report_and_check(&self) -> Result<(), Vec<String>> {
        let report = self.report();
        report.print();
        if report.overall_pass {
            Ok(())
        } else {
            let failures: Vec<String> = report
                .results
                .iter()
                .filter(|r| !r.is_pass())
                .map(|r| r.name().to_string())
                .collect();
            Err(failures)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_fd(kind: &str, bins: Vec<f64>, labels: Vec<&str>) -> FeatureDiscretization {
        FeatureDiscretization {
            kind: kind.to_string(),
            bins: Some(bins),
            labels: Some(labels.into_iter().map(String::from).collect()),
            categories: None,
        }
    }

    #[test]
    fn discretization_exact_match() {
        let a = HashMap::from([(
            "slope".into(),
            make_fd(
                "continuous",
                bins(7.83, 22.45),
                vec!["low", "medium", "high"],
            ),
        )]);
        let g = a.clone();
        assert!(compare_discretization_labels(&a, &g).is_ok());
    }

    #[test]
    fn discretization_detects_drift() {
        let a = HashMap::from([(
            "slope".into(),
            make_fd(
                "continuous",
                bins(7.83, 22.45),
                vec!["low", "medium", "high"],
            ),
        )]);
        let g = HashMap::from([(
            "slope".into(),
            make_fd(
                "continuous",
                bins(7.84, 22.45),
                vec!["low", "medium", "high"],
            ),
        )]);
        let err = compare_discretization_labels(&a, &g).unwrap_err();
        assert!(!err.is_empty());
        assert!(err[0].to_string().contains("slope"));
    }

    #[test]
    fn discretization_detects_missing_feature() {
        let a: HashMap<String, FeatureDiscretization> = HashMap::new();
        let g = HashMap::from([(
            "slope".into(),
            make_fd(
                "continuous",
                bins(7.83, 22.45),
                vec!["low", "medium", "high"],
            ),
        )]);
        let err = compare_discretization_labels(&a, &g).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn statistics_exact_match() {
        let a = vec![sample_factor("IF_001")];
        let g = vec![sample_factor("IF_001")];
        assert!(compare_statistics(&a, &g, 1e-6).is_ok());
    }

    #[test]
    fn statistics_detects_drift() {
        let mut g = vec![sample_factor("IF_001")];
        g[0].relative_risk += 0.01;
        let a = vec![sample_factor("IF_001")];
        let err = compare_statistics(&a, &g, 1e-6).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn probabilities_pass_and_fail() {
        let a = vec![pred_row(0, 0.5)];
        let g = vec![pred_row(0, 0.5)];
        assert!(compare_probabilities(&a, &g, 0.03, 0.1).is_ok());

        let g2 = vec![pred_row(0, 0.7)];
        let err = compare_probabilities(&a, &g2, 0.03, 0.1).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn probabilities_mae_threshold() {
        let a: Vec<_> = (0..10).map(|i| pred_row(i, 0.5)).collect();
        let g: Vec<_> = (0..10).map(|i| pred_row(i, 0.55)).collect();
        // MAE = 0.05 > 0.03, should fail
        let err = compare_probabilities(&a, &g, 0.03, 0.1).unwrap_err();
        assert!(!err.is_empty());
        assert!(err.iter().any(|e| e.to_string().contains("MAE")));
    }

    #[test]
    fn factor_rankings_perfect_correlation() {
        let a = vec![sample_factor("A"), sample_factor("B"), sample_factor("C")];
        let g = vec![sample_factor("A"), sample_factor("B"), sample_factor("C")];
        let (rho, _p) = compare_factor_rankings(&a, &g);
        assert!((rho - 1.0).abs() < 1e-10, "rho should be 1.0, got {rho}");
    }

    #[test]
    fn factor_rankings_reversed() {
        let a = vec![sample_factor("A"), sample_factor("B"), sample_factor("C")];
        let g = vec![sample_factor("C"), sample_factor("B"), sample_factor("A")];
        let (rho, _) = compare_factor_rankings(&a, &g);
        assert!((rho + 1.0).abs() < 1e-10, "rho should be -1.0, got {rho}");
    }

    #[test]
    fn raster_metadata_exact_match() {
        let a = raster_meta();
        let g = raster_meta();
        assert!(compare_raster_metadata(&a, &g).is_ok());
    }

    #[test]
    fn raster_metadata_detects_crs_mismatch() {
        let a = raster_meta();
        let mut g = raster_meta();
        g.crs = "EPSG:4327".into();
        let err = compare_raster_metadata(&a, &g).unwrap_err();
        assert!(!err.is_empty());
        assert!(err[0].to_string().contains("CRS"));
    }

    #[test]
    fn golden_comparator_records_and_reports() {
        let mut cmp = GoldenComparator::new();
        cmp.record_f64("mae", 0.01, 0.0, 0.03);
        cmp.record_f64("max_err", 0.06, 0.0, 0.05);
        let report = cmp.report();
        assert_eq!(report.results.len(), 2);
        assert!(report.results[0].is_pass());
        assert!(!report.results[1].is_pass());
        assert!(!report.overall_pass);
    }

    fn bins(lo: f64, hi: f64) -> Vec<f64> {
        vec![f64::NEG_INFINITY, lo, hi, f64::INFINITY]
    }

    fn sample_factor(id: &str) -> FactorEntry {
        FactorEntry {
            factor_id: id.to_string(),
            name: format!("factor_{id}"),
            semantics: format!("semantics_{id}"),
            items: vec![],
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 0.95,
            score: 3.0,
            count_event: 84,
            count_background: 5,
            n_event: 168,
            n_background: 48,
            rr_ci_lower: 2.0,
            rr_ci_upper: 12.0,
            p_value: 0.001,
            fdr_significant: true,
        }
    }

    fn pred_row(id: u64, prob: f64) -> ModelPredictionRow {
        ModelPredictionRow {
            point_id: id as i64,
            x: 0.0,
            y: 0.0,
            year: 2020,
            disturbance_probability: prob,
            class_probability_1: 0.0,
            class_probability_3: 0.0,
            class_probability_4: 0.0,
            dominant_disturbance_class: 0,
            prob_fire_abs: 0.0,
            prob_geo_hazard_abs: 0.0,
            prob_road_abs: 0.0,
        }
    }

    fn raster_meta() -> RasterMetadata {
        RasterMetadata {
            crs: "EPSG:4326".into(),
            width: 1024,
            height: 1024,
            nodata: Some(-9999.0),
            transform: Some(vec![0.0, 30.0, 0.0, 0.0, 0.0, -30.0]),
        }
    }
}

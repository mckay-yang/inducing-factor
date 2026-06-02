use std::collections::HashMap;

use polars::prelude::*;

use if_core::discretize::AdaptiveDiscretizer;
use if_core::model::{DisturbanceClassEngine, OccurrenceProbabilityEngine};
use if_core::types::InducingFactor;

use crate::feature_builder::FeatureBuilder;

/// Run 12-month ensemble prediction and aggregate into mean/max/peak_month.
///
/// For each month 1-12:
/// 1. Build features with `predict_month = Some(month)` (leakage-free).
/// 2. Filter to the target `year`.
/// 3. Discretize and run occurrence prediction.
/// 4. Collect per-point `(point_id, x, y, raster_row, raster_col, prob, month)`.
///
/// Then group by point_id/coordinates and compute mean, max, peak_month of
/// `disturbance_probability`.
pub fn predict_ensemble(
    df: &DataFrame,
    year: i32,
    feature_builder: &FeatureBuilder,
    discretizer: &Option<AdaptiveDiscretizer>,
    occurrence_engine: &Option<OccurrenceProbabilityEngine>,
    class_engine: &Option<DisturbanceClassEngine>,
    selected_factors: &[InducingFactor],
    feature_columns: &[String],
) -> PolarsResult<DataFrame> {
    let id_cols: Vec<&str> = ["point_id", "x", "y", "raster_row", "raster_col"]
        .iter()
        .filter(|c| df.column(**c).is_ok())
        .copied()
        .collect();
    if id_cols.is_empty() {
        let empty: DataFrame = DataFrame::new(vec![Column::new(
            "disturbance_probability_mean".into(),
            Vec::<f64>::new(),
        )])?;
        return Ok(empty);
    }

    let mut month_rows: Vec<MonthPrediction> = Vec::new();

    for month in 1u32..=12 {
        let mut fb = feature_builder.clone();
        let (features_df, _engineered_cols) =
            fb.build_chunk_features(df, Some(month))?;

        // Filter to target year
        let year_series = features_df.column("year")?;
        let year_i64 = year_series.i64()?;
        let year_mask: Vec<bool> = year_i64
            .into_iter()
            .map(|v| v.is_some_and(|y| y == year as i64))
            .collect();
        let year_df = features_df.filter(&BooleanChunked::new("yr_mask".into(), &year_mask))?;
        if year_df.height() == 0 {
            continue;
        }

        let n_rows = year_df.height();
        let mut feature_data: Vec<Vec<f64>> =
            vec![vec![f64::NAN; feature_columns.len()]; n_rows];
        for (fi, feat) in feature_columns.iter().enumerate() {
            if let Ok(col) = year_df.column(feat) {
                if let Ok(series) = col.as_materialized_series().f64() {
                    for (ri, val) in series.into_iter().enumerate() {
                        feature_data[ri][fi] = val.unwrap_or(f64::NAN);
                    }
                }
            }
        }

        let labels: Vec<HashMap<String, String>> = if let Some(ref disc) = discretizer {
            let transformed = disc.transform(feature_columns, &feature_data, true);
            transformed
                .into_iter()
                .map(|row_labels| {
                    let mut map = HashMap::new();
                    for (fi, feat) in feature_columns.iter().enumerate() {
                        map.insert(feat.clone(), row_labels[fi].clone());
                    }
                    map
                })
                .collect()
        } else {
            feature_data
                .iter()
                .map(|row| {
                    let mut map = HashMap::new();
                    for (fi, feat) in feature_columns.iter().enumerate() {
                        let val = row[fi];
                        map.insert(
                            feat.clone(),
                            if val.is_nan() {
                                "missing".to_string()
                            } else {
                                format!("{:.4}", val)
                            },
                        );
                    }
                    map
                })
                .collect()
        };

        let occ_probs: Vec<f64> = if let Some(ref eng) = occurrence_engine {
            eng.predict_proba(&labels, selected_factors)
        } else {
            vec![0.0; n_rows]
        };

        let class_probs_all: Vec<Vec<(i64, f64)>> = if let Some(ref eng) = class_engine {
            eng.predict_proba(&labels, selected_factors)
        } else {
            vec![vec![]; n_rows]
        };

        for (i, &prob) in occ_probs.iter().enumerate() {
            let mut row = MonthPrediction {
                point_id: None,
                x: None,
                y: None,
                raster_row: None,
                raster_col: None,
                prob,
                month: month as i64,
                class_probs: class_probs_all.get(i).cloned().unwrap_or_default(),
            };
            for &col in &id_cols {
                match col {
                    "point_id" => {
                        if let Ok(c) = year_df.column("point_id") {
                            if let Ok(s) = c.i64() {
                                row.point_id = s.get(i);
                            }
                        }
                    }
                    "x" => {
                        if let Ok(c) = year_df.column("x") {
                            if let Ok(s) = c.f64() {
                                row.x = s.get(i);
                            }
                        }
                    }
                    "y" => {
                        if let Ok(c) = year_df.column("y") {
                            if let Ok(s) = c.f64() {
                                row.y = s.get(i);
                            }
                        }
                    }
                    "raster_row" => {
                        if let Ok(c) = year_df.column("raster_row") {
                            if let Ok(s) = c.i64() {
                                row.raster_row = s.get(i);
                            }
                        }
                    }
                    "raster_col" => {
                        if let Ok(c) = year_df.column("raster_col") {
                            if let Ok(s) = c.i64() {
                                row.raster_col = s.get(i);
                            }
                        }
                    }
                    _ => {}
                }
            }
            month_rows.push(row);
        }
    }

    if month_rows.is_empty() {
        let empty: DataFrame = DataFrame::new(vec![Column::new(
            "disturbance_probability_mean".into(),
            Vec::<f64>::new(),
        )])?;
        return Ok(empty);
    }

    // Group by id_cols and aggregate
    aggregate_ensemble(&month_rows, &id_cols)
}

#[derive(Debug, Clone)]
struct MonthPrediction {
    point_id: Option<i64>,
    x: Option<f64>,
    y: Option<f64>,
    raster_row: Option<i64>,
    raster_col: Option<i64>,
    prob: f64,
    month: i64,
    class_probs: Vec<(i64, f64)>,
}

fn aggregate_ensemble(
    rows: &[MonthPrediction],
    id_cols: &[&str],
) -> PolarsResult<DataFrame> {
    use std::collections::HashMap;

    // Group by point_id (primary key; x/y/raster_row/raster_col are consistent per point_id)
    let mut groups: HashMap<i64, Vec<(f64, i64, Vec<(i64, f64)>)>> = HashMap::new();
    let mut point_meta: HashMap<i64, (Option<f64>, Option<f64>, Option<i64>, Option<i64>)> =
        HashMap::new();

    for row in rows {
        if let Some(pid) = row.point_id {
            groups
                .entry(pid)
                .or_default()
                .push((row.prob, row.month, row.class_probs.clone()));
            point_meta
                .entry(pid)
                .or_insert_with(|| (row.x, row.y, row.raster_row, row.raster_col));
        }
    }

    let mut out_point_ids: Vec<Option<i64>> = Vec::with_capacity(groups.len());
    let mut out_xs: Vec<Option<f64>> = Vec::with_capacity(groups.len());
    let mut out_ys: Vec<Option<f64>> = Vec::with_capacity(groups.len());
    let mut out_rr: Vec<Option<i64>> = Vec::with_capacity(groups.len());
    let mut out_rc: Vec<Option<i64>> = Vec::with_capacity(groups.len());
    let mut out_mean: Vec<f64> = Vec::with_capacity(groups.len());
    let mut out_max: Vec<f64> = Vec::with_capacity(groups.len());
    let mut out_peak: Vec<i64> = Vec::with_capacity(groups.len());
    let mut out_class_fire: Vec<f64> = Vec::with_capacity(groups.len());
    let mut out_class_logging: Vec<f64> = Vec::with_capacity(groups.len());
    let mut out_class_geo: Vec<f64> = Vec::with_capacity(groups.len());
    let mut out_class_road: Vec<f64> = Vec::with_capacity(groups.len());

    for (pid, entries) in &groups {
        let (x, y, rr, rc) = point_meta.get(pid).copied().unwrap_or_default();
        out_point_ids.push(Some(*pid));
        out_xs.push(x);
        out_ys.push(y);
        out_rr.push(rr);
        out_rc.push(rc);

        let probs: Vec<f64> = entries.iter().map(|e| e.0).collect();
        let mean = probs.iter().sum::<f64>() / probs.len() as f64;
        let (max, peak_idx) = probs
            .iter()
            .enumerate()
            .fold((f64::NEG_INFINITY, 0usize), |(mx, mi), (i, &v)| {
                if v > mx { (v, i) } else { (mx, mi) }
            });
        let peak_month = entries[peak_idx].1;

        out_mean.push(mean);
        out_max.push(max);
        out_peak.push(peak_month);

        // Average class probs across months
        let n_months = entries.len() as f64;
        let class_avg = |cls: i64| -> f64 {
            entries
                .iter()
                .map(|e| {
                    e.2.iter()
                        .find(|(c, _)| *c == cls)
                        .map(|(_, p)| *p)
                        .unwrap_or(0.0)
                })
                .sum::<f64>()
                / n_months
        };
        out_class_fire.push(class_avg(1));
        out_class_logging.push(class_avg(2));
        out_class_geo.push(class_avg(3));
        out_class_road.push(class_avg(4));
    }

    let mut columns: Vec<Column> = Vec::new();
    if id_cols.contains(&"point_id") {
        columns.push(Column::new("point_id".into(), out_point_ids));
    }
    if id_cols.contains(&"x") {
        columns.push(Column::new("x".into(), out_xs));
    }
    if id_cols.contains(&"y") {
        columns.push(Column::new("y".into(), out_ys));
    }
    if id_cols.contains(&"raster_row") {
        columns.push(Column::new("raster_row".into(), out_rr));
    }
    if id_cols.contains(&"raster_col") {
        columns.push(Column::new("raster_col".into(), out_rc));
    }
    columns.push(Column::new(
        "disturbance_probability_mean".into(),
        out_mean,
    ));
    columns.push(Column::new(
        "disturbance_probability_max".into(),
        out_max,
    ));
    columns.push(Column::new(
        "disturbance_probability_peak_month".into(),
        out_peak,
    ));
    columns.push(Column::new(
        "disturbance_probability".into(),
        columns[columns.len() - 3]
            .as_materialized_series()
            .clone(),
    ));
    columns.push(Column::new("class_probability_1".into(), out_class_fire));
    columns.push(Column::new("class_probability_2".into(), out_class_logging));
    columns.push(Column::new("class_probability_3".into(), out_class_geo));
    columns.push(Column::new("class_probability_4".into(), out_class_road));

    DataFrame::new(columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_rows() -> Vec<MonthPrediction> {
        let mut rows = Vec::new();
        for month in 1..=12 {
            let prob = month as f64 * 0.05; // linear: 0.05, 0.10, ..., 0.60
            rows.push(MonthPrediction {
                point_id: Some(1),
                x: Some(100.0),
                y: Some(29.0),
                raster_row: Some(10),
                raster_col: Some(20),
                prob,
                month: month as i64,
                class_probs: vec![(1, prob * 0.5), (2, prob * 0.3), (3, prob * 0.1), (4, prob * 0.1)],
            });
        }
        // Add a second point
        for month in 1..=12 {
            let prob = (13 - month) as f64 * 0.05; // descending: 0.60, 0.55, ..., 0.05
            rows.push(MonthPrediction {
                point_id: Some(2),
                x: Some(200.0),
                y: Some(29.0),
                raster_row: Some(11),
                raster_col: Some(21),
                prob,
                month: month as i64,
                class_probs: vec![(1, prob * 0.5), (2, prob * 0.3), (3, prob * 0.1), (4, prob * 0.1)],
            });
        }
        rows
    }

    #[test]
    fn test_ensemble_mean_max_peak() {
        let rows = make_test_rows();
        let id_cols: &[&str] = &["point_id", "x", "y", "raster_row", "raster_col"];
        let result = aggregate_ensemble(&rows, id_cols).unwrap();

        assert_eq!(result.height(), 2);

        let pids: Vec<i64> = result
            .column("point_id")
            .unwrap()
            .i64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap())
            .collect();

        // Find point_id=1
        let idx1 = pids.iter().position(|&p| p == 1).unwrap();
        let idx2 = pids.iter().position(|&p| p == 2).unwrap();

        let mean_col = result.column("disturbance_probability_mean").unwrap();
        let mean_vals: Vec<f64> = mean_col
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap())
            .collect();

        let max_col = result.column("disturbance_probability_max").unwrap();
        let max_vals: Vec<f64> = max_col
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap())
            .collect();

        let peak_col = result
            .column("disturbance_probability_peak_month")
            .unwrap();
        let peak_vals: Vec<i64> = peak_col
            .i64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap())
            .collect();

        // Point 1: probs 0.05..0.60 (ascending), mean = (0.05*(1+..+12))/12 = 0.05*78/12 = 0.325
        let expected_mean1 = 0.05 * 78.0 / 12.0;
        assert!((mean_vals[idx1] - expected_mean1).abs() < 1e-9);
        // Max = 0.60 (month 12)
        assert!((max_vals[idx1] - 0.60).abs() < 1e-9);
        assert_eq!(peak_vals[idx1], 12);

        // Point 2: probs 0.60..0.05 (descending), max = 0.60 (month 1)
        assert!((max_vals[idx2] - 0.60).abs() < 1e-9);
        assert_eq!(peak_vals[idx2], 1);

        // Verify legacy column equals mean
        let legacy = result.column("disturbance_probability").unwrap();
        let legacy_vals: Vec<f64> = legacy
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap())
            .collect();
        assert!((legacy_vals[idx1] - expected_mean1).abs() < 1e-9);
    }
}

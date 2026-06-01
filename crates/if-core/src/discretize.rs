use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::stats::entropy;
use crate::types::{DiscretizerConfig, FeatureMeta};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveDiscretizer {
    pub config: DiscretizerConfig,
    pub feature_meta: HashMap<String, FeatureMeta>,
    pub features: Vec<String>,
}

impl AdaptiveDiscretizer {
    pub fn new(config: DiscretizerConfig) -> Self {
        Self {
            config,
            feature_meta: HashMap::new(),
            features: Vec::new(),
        }
    }

    pub fn fit(&mut self, feature_names: &[String], data: &[Vec<f64>], y: &[f64]) {
        let n_features = feature_names.len();
        self.features = feature_names.to_vec();
        self.feature_meta.clear();

        for fi in 0..n_features {
            let name = &feature_names[fi];
            let col: Vec<f64> = data.iter().map(|row| row[fi]).collect();
            let meta = self.fit_feature(name, &col, y);
            self.feature_meta.insert(name.clone(), meta);
        }
    }

    fn fit_feature(&self, name: &str, series: &[f64], y: &[f64]) -> FeatureMeta {
        let non_null: Vec<usize> = (0..series.len())
            .filter(|&i| !series[i].is_nan())
            .collect();

        if non_null.is_empty() {
            return FeatureMeta {
                kind: "categorical".to_string(),
                bins: None,
                labels: None,
                categories: Some(vec!["missing".to_string()]),
            };
        }

        let non_null_values: Vec<f64> = non_null.iter().map(|&i| series[i]).collect();
        let unique_count = count_unique_f64(&non_null_values);

        if unique_count <= self.config.categorical_unique_threshold {
            let mut categories: Vec<String> = unique_sorted_str(&non_null_values);
            categories.sort();
            return FeatureMeta {
                kind: "categorical".to_string(),
                bins: None,
                labels: None,
                categories: Some(categories),
            };
        }

        self.fit_continuous_feature(name, series, y)
    }

    fn fit_continuous_feature(&self, name: &str, series: &[f64], y: &[f64]) -> FeatureMeta {
        let mask: Vec<usize> = (0..series.len())
            .filter(|&i| !series[i].is_nan() && !y[i].is_nan())
            .collect();

        if mask.len() <= 2 {
            return self.fallback_quantile(name, series);
        }

        let labels: Vec<i64> = mask.iter().map(|&i| y[i] as i64).collect();
        let unique_labels: std::collections::HashSet<i64> = labels.iter().copied().collect();
        if unique_labels.len() <= 1 {
            return self.fallback_quantile(name, series);
        }

        let min_leaf = {
            let m = self.config.min_samples_leaf;
            if m > 0.0 && m < 1.0 {
                (mask.len() as f64 * m).ceil() as usize
            } else {
                m.ceil() as usize
            }
        }
        .max(2);

        let method = &self.config.method;
        let thresholds = if method == "mdlp" {
            self.fit_mdlp(series, y, &mask, min_leaf)
        } else {
            self.fit_supervised_tree(series, y, &mask, min_leaf)
        };

        if thresholds.is_empty() {
            return self.fallback_quantile(name, series);
        }

        let n_states = thresholds.len() + 1;
        let labels = default_labels(name, n_states);

        let mut bins = vec![f64::NEG_INFINITY];
        bins.extend(thresholds);
        bins.push(f64::INFINITY);

        FeatureMeta {
            kind: "continuous".to_string(),
            bins: Some(bins),
            labels: Some(labels),
            categories: None,
        }
    }

    fn fit_supervised_tree(
        &self,
        series: &[f64],
        y: &[f64],
        mask: &[usize],
        min_leaf: usize,
    ) -> Vec<f64> {
        let vals: Vec<f64> = mask.iter().map(|&i| series[i]).collect();
        let lbls: Vec<i64> = mask.iter().map(|&i| y[i] as i64).collect();
        let max_leaf_nodes = self.config.n_bins;

        tree_splits(&vals, &lbls, max_leaf_nodes, min_leaf)
    }

    fn fit_mdlp(
        &self,
        series: &[f64],
        y: &[f64],
        mask: &[usize],
        min_leaf: usize,
    ) -> Vec<f64> {
        let vals: Vec<f64> = mask.iter().map(|&i| series[i]).collect();
        let lbls: Vec<i64> = mask.iter().map(|&i| y[i] as i64).collect();
        if vals.len() < 4 {
            return vec![];
        }

        let mut cuts = Vec::new();
        mdlp_find_cuts(&vals, &lbls, 0, 6, min_leaf, &mut cuts);

        let mut thresholds: Vec<f64> = cuts.iter().copied().collect();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        thresholds
    }

    fn fallback_quantile(&self, name: &str, series: &[f64]) -> FeatureMeta {
        let non_null: Vec<f64> = series.iter().filter(|v| !v.is_nan()).copied().collect();
        let n_states = usize::max(1, self.config.n_bins);
        let thresholds = if non_null.is_empty() || non_null.len() < 2 {
            vec![]
        } else {
            let mut sorted = non_null.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n_bins = self.config.n_bins;
            let mut thresholds: Vec<f64> = Vec::new();
            for k in 1..n_bins {
                let idx = (sorted.len() as f64 * k as f64 / n_bins as f64).floor() as usize;
                if idx < sorted.len() {
                    thresholds.push(sorted[idx]);
                }
            }
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            thresholds
        };

        let labels = default_labels(name, n_states);
        let mut bins = vec![f64::NEG_INFINITY];
        bins.extend(thresholds);
        bins.push(f64::INFINITY);

        FeatureMeta {
            kind: "continuous".to_string(),
            bins: Some(bins),
            labels: Some(labels),
            categories: None,
        }
    }

    pub fn transform(
        &self,
        feature_names: &[String],
        data: &[Vec<f64>],
        as_labels: bool,
    ) -> Vec<Vec<String>> {
        let n_rows = data.len();
        let n_cols = feature_names.len();
        let mut result: Vec<Vec<String>> = vec![vec!["missing".to_string(); n_cols]; n_rows];

        for (fi, name) in feature_names.iter().enumerate() {
            let meta = match self.feature_meta.get(name) {
                Some(m) => m,
                None => continue,
            };

            for ri in 0..n_rows {
                let val = data[ri][fi];
                if val.is_nan() {
                    result[ri][fi] = "missing".to_string();
                    continue;
                }

                match meta.kind.as_str() {
                    "continuous" => {
                        if let Some(ref bins) = meta.bins {
                            let bin_idx = find_bin(val, bins);
                            if as_labels {
                                if let Some(ref labels) = meta.labels {
                                    result[ri][fi] = if bin_idx < labels.len() {
                                        labels[bin_idx].clone()
                                    } else {
                                        "missing".to_string()
                                    };
                                } else {
                                    result[ri][fi] = "missing".to_string();
                                }
                            } else {
                                result[ri][fi] = bin_idx.to_string();
                            }
                        }
                    }
                    "categorical" => {
                        if as_labels {
                            if let Some(ref cats) = meta.categories {
                                let idx = category_index(val, cats);
                                result[ri][fi] = if idx < cats.len() {
                                    cats[idx].clone()
                                } else {
                                    "missing".to_string()
                                };
                            }
                        } else {
                            if let Some(ref cats) = meta.categories {
                                let idx = category_index(val, cats);
                                result[ri][fi] = idx.to_string();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        result
    }
}

fn tree_splits(vals: &[f64], lbls: &[i64], max_leaf_nodes: usize, min_leaf: usize) -> Vec<f64> {
    let n = vals.len();
    if n < min_leaf * 2 || max_leaf_nodes <= 1 {
        return vec![];
    }

    let parent_h = entropy(lbls);

    let best_split = find_best_split(vals, lbls, min_leaf, parent_h);
    if best_split.is_none() {
        return vec![];
    }
    let (t1, _imp1) = best_split.unwrap();

    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for i in 0..n {
        if vals[i] <= t1 {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }

    if max_leaf_nodes <= 2 {
        return vec![t1];
    }

    let left_vals: Vec<f64> = left_idx.iter().map(|&i| vals[i]).collect();
    let left_lbls: Vec<i64> = left_idx.iter().map(|&i| lbls[i]).collect();
    let left_h = entropy(&left_lbls);
    let right_vals: Vec<f64> = right_idx.iter().map(|&i| vals[i]).collect();
    let right_lbls: Vec<i64> = right_idx.iter().map(|&i| lbls[i]).collect();
    let right_h = entropy(&right_lbls);

    let left_split = find_best_split(&left_vals, &left_lbls, min_leaf, left_h);
    let right_split = find_best_split(&right_vals, &right_lbls, min_leaf, right_h);

    let imp_left = match &left_split {
        Some((_, imp)) => *imp,
        None => -1.0,
    };
    let imp_right = match &right_split {
        Some((_, imp)) => *imp,
        None => -1.0,
    };

    let mut thresholds = vec![t1];

    if imp_left >= imp_right && imp_left > 0.0 {
        thresholds.push(left_split.unwrap().0);
    } else if imp_right > 0.0 {
        thresholds.push(right_split.unwrap().0);
    }

    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    thresholds
}

fn find_best_split(vals: &[f64], lbls: &[i64], min_leaf: usize, parent_h: f64) -> Option<(f64, f64)> {
    let n = vals.len();
    if n < min_leaf * 2 {
        return None;
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_vals: Vec<f64> = idx.iter().map(|&i| vals[i]).collect();
    let sorted_lbls: Vec<i64> = idx.iter().map(|&i| lbls[i]).collect();

    let mut midpoints: Vec<f64> = Vec::new();
    for i in 0..sorted_vals.len() - 1 {
        if sorted_vals[i + 1] - sorted_vals[i] > 1e-15 {
            let mid = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0;
            if midpoints.is_empty() || (mid - midpoints.last().unwrap()).abs() > 1e-15 {
                midpoints.push(mid);
            }
        }
    }

    if midpoints.is_empty() {
        return None;
    }

    let sample_mids = if midpoints.len() > 50 {
        let step = midpoints.len() / 50;
        midpoints
            .iter()
            .enumerate()
            .filter(|(i, _)| i % step == 0 || *i == midpoints.len() - 1)
            .map(|(_, &v)| v)
            .collect()
    } else {
        midpoints
    };

    let mut best_threshold: Option<f64> = None;
    let mut best_weighted_entropy = f64::INFINITY;
    let mut best_impurity_decrease = f64::NEG_INFINITY;

    for &mid in &sample_mids {
        let mut n_left = 0;
        let mut n_right = 0;
        for &v in &sorted_vals {
            if v <= mid {
                n_left += 1;
            } else {
                n_right += 1;
            }
        }

        if n_left < min_leaf || n_right < min_leaf {
            continue;
        }

        let left_lbls: Vec<i64> = sorted_lbls[..n_left].to_vec();
        let right_lbls: Vec<i64> = sorted_lbls[n_left..].to_vec();

        let h_left = entropy(&left_lbls);
        let h_right = entropy(&right_lbls);
        let weighted_ent = (n_left as f64 * h_left + n_right as f64 * h_right) / n as f64;
        let impurity_decrease = n as f64 * parent_h - n_left as f64 * h_left - n_right as f64 * h_right;

        if impurity_decrease > best_impurity_decrease + 1e-12 {
            best_impurity_decrease = impurity_decrease;
            best_weighted_entropy = weighted_ent;
            best_threshold = Some(mid);
        } else if (impurity_decrease - best_impurity_decrease).abs() < 1e-12 && weighted_ent < best_weighted_entropy {
            best_weighted_entropy = weighted_ent;
            best_threshold = Some(mid);
        }
    }

    best_threshold.map(|t| (t, best_impurity_decrease))
}

fn mdlp_find_cuts(
    vals: &[f64],
    lbls: &[i64],
    depth: usize,
    max_depth: usize,
    min_leaf: usize,
    cuts: &mut Vec<f64>,
) {
    if depth >= max_depth || vals.len() < 4 {
        return;
    }

    let unique_labels: std::collections::HashSet<i64> = lbls.iter().copied().collect();
    if unique_labels.len() < 2 {
        return;
    }

    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_vals: Vec<f64> = idx.iter().map(|&i| vals[i]).collect();
    let sorted_lbls: Vec<i64> = idx.iter().map(|&i| lbls[i]).collect();

    let mut midpoints: Vec<f64> = Vec::new();
    for i in 0..sorted_vals.len() - 1 {
        if sorted_vals[i + 1] - sorted_vals[i] > 1e-15 {
            let mid = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0;
            if midpoints.is_empty() || (mid - midpoints.last().unwrap()).abs() > 1e-15 {
                midpoints.push(mid);
            }
        }
    }

    let sample_mids = if midpoints.len() > 50 {
        let step = midpoints.len() / 50;
        midpoints
            .iter()
            .enumerate()
            .step_by(step)
            .map(|(_, &v)| v)
            .collect()
    } else {
        midpoints
    };

    let parent_h = entropy(lbls);
    let mut best_mid: Option<f64> = None;
    let mut best_gain = -1.0_f64;

    for &mid in &sample_mids {
        if !mdlp_accept(&sorted_vals, &sorted_lbls, mid, min_leaf) {
            continue;
        }

        let mut n_left = 0;
        for &v in &sorted_vals {
            if v <= mid {
                n_left += 1;
            } else {
                break;
            }
        }

        let n_right = sorted_vals.len() - n_left;
        let left_lbls: Vec<i64> = sorted_lbls[..n_left].to_vec();
        let right_lbls: Vec<i64> = sorted_lbls[n_left..].to_vec();
        let h_left = entropy(&left_lbls);
        let h_right = entropy(&right_lbls);

        let gain = parent_h
            - (n_left as f64 / sorted_vals.len() as f64) * h_left
            - (n_right as f64 / sorted_vals.len() as f64) * h_right;

        if gain > best_gain {
            best_gain = gain;
            best_mid = Some(mid);
        }
    }

    if let Some(mid) = best_mid {
        cuts.push(mid);

        let left_mask: Vec<usize> = (0..vals.len()).filter(|&i| vals[i] <= mid).collect();
        let right_mask: Vec<usize> = (0..vals.len()).filter(|&i| vals[i] > mid).collect();

        let left_vals: Vec<f64> = left_mask.iter().map(|&i| vals[i]).collect();
        let left_lbls: Vec<i64> = left_mask.iter().map(|&i| lbls[i]).collect();
        let right_vals: Vec<f64> = right_mask.iter().map(|&i| vals[i]).collect();
        let right_lbls: Vec<i64> = right_mask.iter().map(|&i| lbls[i]).collect();

        mdlp_find_cuts(&left_vals, &left_lbls, depth + 1, max_depth, min_leaf, cuts);
        mdlp_find_cuts(&right_vals, &right_lbls, depth + 1, max_depth, min_leaf, cuts);
    }
}

fn mdlp_accept(vals: &[f64], lbls: &[i64], threshold: f64, _min_leaf: usize) -> bool {
    let mut n_left = 0;
    for &v in vals {
        if v <= threshold {
            n_left += 1;
        } else {
            break;
        }
    }
    let n_right = vals.len() - n_left;
    if n_left < 2 || n_right < 2 {
        return false;
    }

    let parent_h = entropy(lbls);
    let left_lbls: Vec<i64> = lbls[..n_left].to_vec();
    let right_lbls: Vec<i64> = lbls[n_left..].to_vec();
    let h_left = entropy(&left_lbls);
    let h_right = entropy(&right_lbls);

    let n = vals.len() as f64;
    let info_gain = parent_h
        - (n_left as f64 / n) * h_left
        - (n_right as f64 / n) * h_right;

    let k = count_unique_i64(lbls) as f64;
    let k_left = count_unique_i64(&left_lbls) as f64;
    let k_right = count_unique_i64(&right_lbls) as f64;

    let delta = libm::log2(3.0_f64.powf(k) - 2.0)
        - k * parent_h
        + k_left * h_left
        + k_right * h_right;

    info_gain > (libm::log2(n - 1.0) / n + delta / n)
}

fn find_bin(val: f64, bins: &[f64]) -> usize {
    if val.is_nan() {
        return bins.len();
    }
    for i in 1..bins.len() {
        if val <= bins[i] {
            return i - 1;
        }
    }
    bins.len() - 2
}

fn category_index(val: f64, categories: &[String]) -> usize {
    let val_str = if val.fract().abs() < 1e-12 {
        format!("{:.1}", val)
    } else {
        format!("{}", val)
    };
    categories.iter().position(|c| c == &val_str).unwrap_or(categories.len())
}

fn count_unique_f64(vals: &[f64]) -> usize {
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut count = 0;
    for i in 0..sorted.len() {
        if i == 0 || (sorted[i] - sorted[i - 1]).abs() > 1e-12 {
            count += 1;
        }
    }
    count
}

fn count_unique_i64(vals: &[i64]) -> usize {
    let mut sorted = vals.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted.len()
}

fn unique_sorted_str(vals: &[f64]) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for v in &sorted {
        let s = if v.fract().abs() < 1e-12 {
            format!("{:.1}", v)
        } else {
            format!("{}", v)
        };
        if result.last() != Some(&s) {
            result.push(s);
        }
    }
    result
}

pub fn default_labels(feature: &str, n_states: usize) -> Vec<String> {
    if n_states <= 1 {
        return vec!["state_0".to_string()];
    }

    if feature.contains("dist_") {
        match n_states {
            2 => vec!["near".to_string(), "far".to_string()],
            3 => vec!["near".to_string(), "mid".to_string(), "far".to_string()],
            4 => vec![
                "very_near".to_string(),
                "near".to_string(),
                "far".to_string(),
                "very_far".to_string(),
            ],
            5 => vec![
                "very_near".to_string(),
                "near".to_string(),
                "mid".to_string(),
                "far".to_string(),
                "very_far".to_string(),
            ],
            _ => (0..n_states).map(|i| format!("state_{}", i)).collect(),
        }
    } else {
        match n_states {
            2 => vec!["low".to_string(), "high".to_string()],
            3 => vec!["low".to_string(), "medium".to_string(), "high".to_string()],
            4 => vec![
                "very_low".to_string(),
                "low".to_string(),
                "high".to_string(),
                "very_high".to_string(),
            ],
            5 => vec![
                "very_low".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "very_high".to_string(),
            ],
            _ => (0..n_states).map(|i| format!("state_{}", i)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_unique_f64() {
        assert_eq!(count_unique_f64(&[1.0, 2.0, 2.0, 3.0]), 3);
        assert_eq!(count_unique_f64(&[]), 0);
    }

    #[test]
    fn test_default_labels_dist() {
        let labels = default_labels("dist_to_road", 3);
        assert_eq!(labels, vec!["near", "mid", "far"]);
    }

    #[test]
    fn test_default_labels_non_dist() {
        let labels = default_labels("slope", 3);
        assert_eq!(labels, vec!["low", "medium", "high"]);
    }

    #[test]
    fn test_find_bin_basic() {
        let bins = vec![f64::NEG_INFINITY, 5.0, 10.0, f64::INFINITY];
        assert_eq!(find_bin(3.0, &bins), 0);
        assert_eq!(find_bin(5.0, &bins), 0);
        assert_eq!(find_bin(7.0, &bins), 1);
        assert_eq!(find_bin(15.0, &bins), 2);
    }

    #[test]
    fn test_discretizer_empty_data() {
        let mut disc = AdaptiveDiscretizer::new(DiscretizerConfig::default());
        let features: Vec<String> = vec!["f1".to_string()];
        let data: Vec<Vec<f64>> = vec![vec![f64::NAN]];
        let y: Vec<f64> = vec![1.0];
        disc.fit(&features, &data, &y);
        let meta = disc.feature_meta.get("f1").unwrap();
        assert_eq!(meta.kind, "categorical");
        assert_eq!(meta.categories.as_ref().unwrap(), &vec!["missing"]);
    }

    #[test]
    fn test_discretizer_single_value() {
        let mut config = DiscretizerConfig::default();
        config.categorical_unique_threshold = 12;
        let mut disc = AdaptiveDiscretizer::new(config);
        let features: Vec<String> = vec!["f1".to_string()];
        let data: Vec<Vec<f64>> = (0..10).map(|_| vec![5.0]).collect();
        let y: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        disc.fit(&features, &data, &y);
        let meta = disc.feature_meta.get("f1").unwrap();
        assert!(meta.kind == "continuous" || meta.kind == "categorical");
    }

    #[test]
    fn test_discretizer_categorical_threshold() {
        let mut config = DiscretizerConfig::default();
        config.categorical_unique_threshold = 12;
        let mut disc = AdaptiveDiscretizer::new(config);
        let features: Vec<String> = vec!["landuse".to_string()];
        let data: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![1.0], vec![2.0]];
        let y: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
        disc.fit(&features, &data, &y);
        let meta = disc.feature_meta.get("landuse").unwrap();
        assert_eq!(meta.kind, "categorical");
    }

    #[test]
    fn test_discretizer_tree_splits_two_groups() {
        let vals = vec![1.0, 2.0, 3.0, 8.0, 9.0, 10.0];
        let lbls = vec![0, 0, 0, 1, 1, 1];
        let thresholds = tree_splits(&vals, &lbls, 3, 2);
        assert_eq!(thresholds.len(), 1);
        assert!(thresholds[0] > 3.0 && thresholds[0] < 8.0);
    }
}

use std::collections::{BTreeMap, BTreeSet, HashMap};

use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::stats::{benjamini_hochberg, relative_risk_ci, true_relative_risk};
use crate::types::{FactorMiningConfig, InducingFactor};

pub struct FactorMiner {
    config: FactorMiningConfig,
}

impl FactorMiner {
    pub fn new(config: FactorMiningConfig) -> Self {
        Self { config }
    }

    pub fn transactions_from_labels(
        &self,
        label_rows: &[Vec<String>],
        feature_names: &[String],
    ) -> Vec<Vec<String>> {
        label_rows
            .iter()
            .map(|row| {
                let mut items: Vec<String> = feature_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| format!("{}={}", name, row[i]))
                    .collect();
                items.sort();
                items
            })
            .collect()
    }

    pub fn count_patterns(
        &self,
        transactions: &[Vec<String>],
    ) -> HashMap<Vec<String>, usize> {
        let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
        for items in transactions {
            let max_len = self.config.max_pattern_size.min(items.len());
            for size in 1..=max_len {
                for combo in items.iter().combinations(size) {
                    let pattern: Vec<String> = combo.iter().map(|&&ref s| s.clone()).collect();
                    *counts.entry(pattern).or_insert(0) += 1;
                }
            }
        }
        counts
    }

    pub fn mine_pattern_candidates(
        &self,
        event_transactions: &[Vec<String>],
        background_transactions: &[Vec<String>],
    ) -> HashMap<Vec<String>, PatternStats> {
        if event_transactions.is_empty() {
            return HashMap::new();
        }

        let event_counts = self.count_patterns(event_transactions);
        let background_counts = self.count_patterns(background_transactions);
        let n_event = event_transactions.len();
        let n_bg = background_transactions.len().max(1);

        let min_event_count = ((self.config.min_support * n_event as f64).ceil() as usize).max(1);
        let epsilon = 1e-6_f64;

        let mut candidates: HashMap<Vec<String>, PatternStats> = HashMap::new();

        for (pattern, count_event) in &event_counts {
            if *count_event < min_event_count {
                continue;
            }
            let count_bg = background_counts.get(pattern).copied().unwrap_or(0);
            let support_event = *count_event as f64 / n_event as f64;
            let support_background = count_bg as f64 / n_bg as f64;
            let relative_risk =
                (support_event + epsilon) / (support_background + epsilon);

            if relative_risk < self.config.min_relative_risk {
                continue;
            }

            let n_bg_safe = n_bg.max(1) as i64;
            let (_rr, ci_lo, ci_hi, p_val) = relative_risk_ci(
                *count_event as i64,
                n_event as i64,
                count_bg as i64,
                n_bg_safe,
                0.05,
                0.5,
            );
            let true_rr = true_relative_risk(
                *count_event as i64,
                count_bg as i64,
                n_event as i64,
                n_bg_safe,
                0.5,
            );

            candidates.insert(
                pattern.clone(),
                PatternStats {
                    count_event: *count_event,
                    count_background: count_bg,
                    support_event,
                    support_background,
                    relative_risk,
                    p_value: p_val,
                    rr_ci_lower: ci_lo,
                    rr_ci_upper: ci_hi,
                    true_relative_risk: true_rr,
                    stability: 0.0,
                },
            );
        }

        candidates
    }

    pub fn estimate_pattern_stability(
        &self,
        event_transactions: &[Vec<String>],
        background_transactions: &[Vec<String>],
    ) -> HashMap<Vec<String>, f64> {
        if event_transactions.is_empty() {
            return HashMap::new();
        }

        let mut rng = StdRng::seed_from_u64(self.config.random_state);
        let mut stability_counter: HashMap<Vec<String>, usize> = HashMap::new();
        let n_event = event_transactions.len();
        let n_bg = background_transactions.len();

        for _ in 0..self.config.bootstrap_iterations {
            let event_size = self.config.max_bootstrap_samples.min(n_event);
            let bg_size = self.config.max_bootstrap_samples.min(n_bg);

            let event_sample: Vec<Vec<String>> = (0..event_size)
                .map(|_| {
                    let idx = rng.gen_range(0..n_event);
                    event_transactions[idx].clone()
                })
                .collect();

            let bg_sample: Vec<Vec<String>> = if n_bg > 0 {
                (0..bg_size)
                    .map(|_| {
                        let idx = rng.gen_range(0..n_bg);
                        background_transactions[idx].clone()
                    })
                    .collect()
            } else {
                vec![]
            };

            let mined = self.mine_pattern_candidates(&event_sample, &bg_sample);
            for pattern in mined.keys() {
                *stability_counter.entry(pattern.clone()).or_insert(0) += 1;
            }
        }

        let mut stability: HashMap<Vec<String>, f64> = HashMap::new();
        let div = self.config.bootstrap_iterations.max(1) as f64;
        for (pattern, count) in stability_counter {
            stability.insert(pattern, count as f64 / div);
        }
        stability
    }

    pub fn reduce_redundant_patterns(
        &self,
        pattern_candidates: &HashMap<Vec<String>, PatternStats>,
        pattern_stability: &HashMap<Vec<String>, f64>,
        redundancy_threshold: f64,
    ) -> HashMap<Vec<String>, PatternStats> {
        let mut stable_patterns: HashMap<Vec<String>, PatternStats> = HashMap::new();
        for (pattern, stats) in pattern_candidates {
            let stability = pattern_stability.get(pattern).copied().unwrap_or(0.0);
            if stability >= self.config.min_stability {
                let mut s = stats.clone();
                s.stability = stability;
                stable_patterns.insert(pattern.clone(), s);
            }
        }

        if stable_patterns.is_empty() {
            return stable_patterns;
        }

        let mut patterns_by_size: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();
        for pattern in stable_patterns.keys() {
            patterns_by_size
                .entry(pattern.len())
                .or_default()
                .push(pattern.clone());
        }

        let mut to_remove: BTreeSet<Vec<String>> = BTreeSet::new();

        let sizes: Vec<usize> = patterns_by_size.keys().copied().sorted().rev().collect();
        for &size in &sizes {
            if size == 1 {
                continue;
            }
            if let Some(patterns) = patterns_by_size.get(&size) {
                for pattern in patterns {
                    if to_remove.contains(pattern) {
                        continue;
                    }
                    let rr_pattern = stable_patterns[pattern].relative_risk;
                    for drop_idx in 0..pattern.len() {
                        let subset: Vec<String> = pattern
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != drop_idx)
                            .map(|(_, s)| s.clone())
                            .collect();
                        if stable_patterns.contains_key(&subset)
                            && !to_remove.contains(&subset)
                        {
                            let rr_subset = stable_patterns[&subset].relative_risk;
                            if (rr_pattern - rr_subset).abs()
                                < redundancy_threshold * rr_subset
                            {
                                to_remove.insert(pattern.clone());
                                break;
                            }
                        }
                    }
                }
            }
        }

        stable_patterns.retain(|k, _| !to_remove.contains(k));
        stable_patterns
    }

    pub fn build_factor_library(
        &self,
        reduced_patterns: &HashMap<Vec<String>, PatternStats>,
        n_event: i64,
        n_background: i64,
    ) -> Vec<InducingFactor> {
        let patterns_list: Vec<(Vec<String>, PatternStats)> = reduced_patterns
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let p_values: Vec<f64> = patterns_list.iter().map(|(_, s)| s.p_value).collect();
        let fdr_results = if !p_values.is_empty() {
            benjamini_hochberg(&p_values, 0.05)
        } else {
            vec![]
        };

        let mut library: Vec<InducingFactor> = Vec::new();
        for (idx, (pattern, stats)) in patterns_list.iter().enumerate() {
            let stability = stats.stability;
            let score = if self.config.score_method == "log_rr" {
                stats.support_event * libm::log(stats.relative_risk.max(1.0)) * stability
            } else {
                stats.support_event * stats.relative_risk * stability
            };

            let name = pattern_name(pattern);
            let semantics = pattern_semantics(pattern);
            let count_event = stats.count_event as i64;
            let count_background = stats.count_background as i64;

            let fdr_sig = if idx < fdr_results.len() {
                fdr_results[idx]
            } else {
                false
            };

            library.push(InducingFactor {
                factor_id: String::new(),
                name,
                semantics,
                items: pattern.clone(),
                support_event: stats.support_event,
                support_background: stats.support_background,
                relative_risk: stats.relative_risk,
                stability,
                score,
                count_event,
                count_background,
                n_event,
                n_background,
                rr_ci_lower: stats.rr_ci_lower,
                rr_ci_upper: stats.rr_ci_upper,
                p_value: stats.p_value,
                fdr_significant: fdr_sig,
                likelihood_ratio: stats.relative_risk,
                true_relative_risk: stats.true_relative_risk,
            });
        }

        library.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, factor) in library.iter_mut().enumerate() {
            factor.factor_id = format!("IF_{:03}", idx + 1);
        }

        library
    }
}

#[derive(Debug, Clone)]
pub struct PatternStats {
    pub count_event: usize,
    pub count_background: usize,
    pub support_event: f64,
    pub support_background: f64,
    pub relative_risk: f64,
    pub p_value: f64,
    pub rr_ci_lower: f64,
    pub rr_ci_upper: f64,
    pub true_relative_risk: f64,
    pub stability: f64,
}

pub fn pattern_name(pattern: &[String]) -> String {
    pattern
        .iter()
        .map(|item| item.replace('=', "_"))
        .join("__")
}

pub fn pattern_semantics(pattern: &[String]) -> String {
    let readable: Vec<String> = pattern
        .iter()
        .map(|item| {
            let parts: Vec<&str> = item.splitn(2, '=').collect();
            if parts.len() == 2 {
                let feature = parts[0];
                let state = parts[1].replace('_', " ");
                let display_feature = if feature.ends_with("_prev_mean") {
                    format!("mean {}", &feature[..feature.len() - 10])
                } else if feature.ends_with("_prev_max") {
                    format!("max {}", &feature[..feature.len() - 9])
                } else if feature.ends_with("_prev_last") {
                    format!("last {}", &feature[..feature.len() - 10])
                } else if feature.ends_with("_prev_sum") {
                    format!("total {}", &feature[..feature.len() - 9])
                } else if feature.ends_with("_prev_min") {
                    format!("min {}", &feature[..feature.len() - 9])
                } else {
                    feature.to_string()
                };
                format!("{} {}", state, display_feature)
            } else {
                item.clone()
            }
        })
        .collect();
    readable.join(" + ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_name_single() {
        let name = pattern_name(&["rainfall_prev_2m=high".to_string()]);
        assert_eq!(name, "rainfall_prev_2m_high");
    }

    #[test]
    fn test_pattern_name_multi() {
        let name = pattern_name(&[
            "dist_to_road=near".to_string(),
            "rainfall_prev_1m=high".to_string(),
        ]);
        assert_eq!(name, "dist_to_road_near__rainfall_prev_1m_high");
    }

    #[test]
    fn test_pattern_semantics_single() {
        let sem = pattern_semantics(&["rainfall_prev_2m=high".to_string()]);
        assert_eq!(sem, "high rainfall_prev_2m");
    }

    #[test]
    fn test_pattern_semantics_prev_last() {
        let sem = pattern_semantics(&[
            "rainfall_prev_1m=high".to_string(),
            "resident_prev_last=low".to_string(),
        ]);
        assert!(
            sem.contains("low last resident"),
            "Expected 'low last resident', got: {}",
            sem
        );
    }

    #[test]
    fn test_factor_mining_empty_input() {
        let config = FactorMiningConfig::default();
        let miner = FactorMiner::new(config);
        let candidates = miner.mine_pattern_candidates(&[], &[]);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_factor_mining_low_support() {
        let mut config = FactorMiningConfig::default();
        config.min_support = 0.5;
        config.min_relative_risk = 1.0;
        let miner = FactorMiner::new(config);

        let event_tx: Vec<Vec<String>> = (0..10)
            .map(|i| vec![format!("f1=val_{}", i % 3)])
            .collect();
        let bg_tx: Vec<Vec<String>> = (0..10)
            .map(|i| vec![format!("f1=val_{}", i % 3)])
            .collect();

        let candidates = miner.mine_pattern_candidates(&event_tx, &bg_tx);
        assert!(candidates.is_empty(), "Expected no pattern with high min_support");
    }

    #[test]
    fn test_reduce_redundant_removes_larger_similar() {
        let mut candidates: HashMap<Vec<String>, PatternStats> = HashMap::new();
        candidates.insert(
            vec!["A=high".to_string()],
            PatternStats {
                count_event: 50,
                count_background: 10,
                support_event: 0.5,
                support_background: 0.1,
                relative_risk: 5.0,
                p_value: 0.001,
                rr_ci_lower: 2.0,
                rr_ci_upper: 10.0,
                true_relative_risk: 5.0,
                stability: 0.0,
            },
        );
        candidates.insert(
            vec!["A=high".to_string(), "B=low".to_string()],
            PatternStats {
                count_event: 48,
                count_background: 9,
                support_event: 0.48,
                support_background: 0.09,
                relative_risk: 5.01,
                p_value: 0.001,
                rr_ci_lower: 2.0,
                rr_ci_upper: 10.0,
                true_relative_risk: 5.01,
                stability: 0.0,
            },
        );

        let stability: HashMap<Vec<String>, f64> = vec![
            (vec!["A=high".to_string()], 1.0),
            (vec!["A=high".to_string(), "B=low".to_string()], 1.0),
        ]
        .into_iter()
        .collect();

        let config = FactorMiningConfig::default();
        let miner = FactorMiner::new(config);
        let reduced = miner.reduce_redundant_patterns(&candidates, &stability, 0.1);

        assert!(
            reduced.contains_key(&vec!["A=high".to_string()]),
            "Single-item pattern should be kept"
        );
        assert!(
            !reduced.contains_key(&vec!["A=high".to_string(), "B=low".to_string()]),
            "Redundant larger pattern should be removed"
        );
    }

    #[test]
    fn test_combinations_empty_transaction() {
        let config = FactorMiningConfig::default();
        let miner = FactorMiner::new(config);
        let counts = miner.count_patterns(&[vec![]]);
        assert!(counts.is_empty());
    }
}

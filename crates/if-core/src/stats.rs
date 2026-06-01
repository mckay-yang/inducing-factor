use statrs::distribution::{ContinuousCDF, DiscreteCDF, Hypergeometric, Normal};

pub fn entropy(labels: &[i64]) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let n = labels.len() as f64;
    let mut counts: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
    for &v in labels {
        *counts.entry(v).or_insert(0.0) += 1.0;
    }
    let mut h = 0.0;
    let epsilon = 1e-12;
    for &c in counts.values() {
        if c > 0.0 {
            let p = c / n;
            h -= p * libm::log2(p + epsilon);
        }
    }
    h
}

pub fn relative_risk_ci(
    count_event: i64,
    n_event: i64,
    count_bg: i64,
    n_bg: i64,
    alpha: f64,
    epsilon: f64,
) -> (f64, f64, f64, f64) {
    let count_event_adj = count_event as f64 + epsilon;
    let count_bg_adj = (count_bg.max(0) as f64) + epsilon;
    let p_event = count_event_adj / (n_event as f64 + epsilon);
    let p_bg = count_bg_adj / (n_bg as f64 + epsilon);
    let rr = p_event / p_bg;

    let se_ln_rr = libm::sqrt(
        1.0 / count_event_adj + 1.0 / count_bg_adj
            - 1.0 / (n_event as f64 + epsilon)
            - 1.0 / (n_bg as f64 + epsilon),
    )
    .max(1e-10);

    let z = norm_ppf(1.0 - alpha / 2.0);
    let ci_lo = libm::exp(libm::log(rr.max(1e-300)) - z * se_ln_rr);
    let ci_hi = libm::exp(libm::log(rr.max(1e-300)) + z * se_ln_rr);

    let (_or, p_value) = fisher_exact_test(count_event, n_event, count_bg, n_bg);
    (rr, ci_lo, ci_hi, p_value)
}

pub fn fisher_exact_test(a: i64, n_a: i64, c: i64, n_b: i64) -> (f64, f64) {
    let b = n_a - a;
    let d = n_b - c;
    let total = a + b + c + d;

    if total <= 0 {
        return (1.0, 1.0);
    }

    let odds_ratio = if b > 0 && c > 0 {
        (a * d) as f64 / (b * c) as f64
    } else {
        f64::INFINITY
    };

    if a <= 0 {
        return (odds_ratio, 1.0);
    }

    if a + c == 0 {
        return (odds_ratio, 1.0);
    }

    let population = total as u64;
    let successes = n_a as u64;
    let draws = (a + c) as u64;

    if draws == 0 || successes == 0 || draws > population {
        return (odds_ratio, 1.0);
    }

    match Hypergeometric::new(population, successes, draws) {
        Ok(hyper) => {
            let p_value = if a as u64 > 0 {
                1.0 - hyper.cdf((a - 1) as u64)
            } else {
                1.0
            };
            (odds_ratio, p_value.clamp(0.0, 1.0))
        }
        Err(_) => (odds_ratio, 1.0),
    }
}

pub fn benjamini_hochberg(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let n = p_values.len();
    if n == 0 {
        return vec![];
    }

    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut threshold: f64 = 0.0;
    for (rank_i, &(_orig_idx, p)) in indexed.iter().enumerate() {
        let bh_threshold = alpha * (rank_i + 1) as f64 / n as f64;
        if p <= bh_threshold {
            threshold = bh_threshold;
        }
    }

    let mut rejected = vec![false; n];
    for &(orig_idx, p) in &indexed {
        if p <= threshold {
            rejected[orig_idx] = true;
        }
    }
    rejected
}

pub fn true_relative_risk(
    count_pattern_event: i64,
    count_pattern_bg: i64,
    n_event: i64,
    n_bg: i64,
    epsilon: f64,
) -> f64 {
    let count_pattern_bg_adj = (count_pattern_bg.max(0) as f64) + epsilon;
    let n_pattern = count_pattern_event as f64 + count_pattern_bg_adj;
    let n_not_pattern = n_event as f64 + n_bg as f64 - count_pattern_event as f64
        - count_pattern_bg.max(0) as f64;
    let p_event_given_pattern =
        (count_pattern_event as f64 + epsilon) / (n_pattern + epsilon);
    let p_event_given_not_pattern =
        ((n_event - count_pattern_event).max(0) as f64 + epsilon)
            / (n_not_pattern.max(1.0) + epsilon);
    p_event_given_pattern / p_event_given_not_pattern
}

pub fn morans_i(values: &[f64], weights: &[Vec<f64>]) -> (f64, f64) {
    let n = values.len();
    if n < 3 {
        return (0.0, 1.0);
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let deviations: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let s0: f64 = weights.iter().flat_map(|row| row.iter()).sum();
    if s0 == 0.0 {
        return (0.0, 1.0);
    }

    let mut numerator = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            numerator += weights[i][j] * deviations[i] * deviations[j];
        }
    }
    let denominator: f64 = deviations.iter().map(|d| d * d).sum();
    if denominator == 0.0 {
        return (0.0, 1.0);
    }
    let i_stat = (n as f64 / s0) * (numerator / denominator);

    let e_i = -1.0 / (n as f64 - 1.0);
    let s1: f64 = weights
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter().enumerate().map(move |(j, &w)| {
                let wij = if i == j { 0.0 } else { w };
                (wij + w) * (wij + w) / 2.0
            })
        })
        .sum();
    let s2: f64 = weights
        .iter()
        .map(|row| {
            let ri: f64 = row.iter().sum();
            ri * ri
        })
        .sum();
    let n_f = n as f64;
    let var_i = (n_f * (n_f * n_f - 3.0 * n_f + 3.0) * s1
        - n_f * s2
        + 3.0 * s0 * s0)
        / ((n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * s0 * s0)
        - e_i * e_i;

    if var_i <= 0.0 {
        return (i_stat, 1.0);
    }
    let z = (i_stat - e_i) / libm::sqrt(var_i);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value_two_sided = 2.0 * (1.0 - normal.cdf(z.abs()));
    (i_stat, p_value_two_sided)
}

fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    Normal::new(0.0, 1.0).unwrap().inverse_cdf(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        let labels = vec![0, 1, 0, 1];
        let h = entropy(&labels);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_pure() {
        let labels = vec![1, 1, 1, 1];
        let h = entropy(&labels);
        assert!(h.abs() < 2e-12, "entropy of pure set should be near zero, got {}", h);
    }

    #[test]
    fn test_entropy_empty() {
        let h = entropy(&[]);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_fisher_exact_basic() {
        let (or, pv) = fisher_exact_test(10, 100, 2, 100);
        assert!(or > 1.0);
        assert!(pv < 0.05);
    }

    #[test]
    fn test_fisher_exact_no_association() {
        let (or, pv) = fisher_exact_test(50, 100, 50, 100);
        assert!((or - 1.0).abs() < 0.1);
        assert!(pv > 0.05);
    }

    #[test]
    fn test_relative_risk_ci_basic() {
        let (rr, _ci_lo, _ci_hi, pv) = relative_risk_ci(50, 100, 10, 100, 0.05, 0.5);
        assert!(rr > 1.0);
        assert!(pv < 0.001);
    }

    #[test]
    fn test_benjamini_hochberg_all_significant() {
        let pvals = vec![0.001, 0.002, 0.003];
        let rejected = benjamini_hochberg(&pvals, 0.05);
        assert_eq!(rejected, vec![true, true, true]);
    }

    #[test]
    fn test_benjamini_hochberg_none_significant() {
        let pvals = vec![0.5, 0.6, 0.7];
        let rejected = benjamini_hochberg(&pvals, 0.05);
        assert_eq!(rejected, vec![false, false, false]);
    }

    #[test]
    fn test_benjamini_hochberg_empty() {
        let rejected = benjamini_hochberg(&[], 0.05);
        assert!(rejected.is_empty());
    }

    #[test]
    fn test_true_relative_risk_basic() {
        let trr = true_relative_risk(50, 10, 100, 100, 0.5);
        assert!(trr > 1.0);
    }

    #[test]
    fn test_morans_i_no_spatial() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = values.len();
        let weights: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 0.0 } else { 1.0 }).collect())
            .collect();
        let (i_stat, _pv) = morans_i(&values, &weights);
        assert!(!i_stat.is_nan());
    }

    #[test]
    fn test_norm_ppf_median() {
        let z = norm_ppf(0.5);
        assert!(z.abs() < 1e-6);
    }

    #[test]
    fn test_norm_ppf_975() {
        let z = norm_ppf(0.975);
        assert!((z - 1.95996).abs() < 0.001);
    }
}

use std::collections::HashMap;

use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::types::InducingFactor;

fn default_activation_gamma() -> f64 {
    1.5
}

// ── Matrix helpers ──────────────────────────────────────────────

#[allow(non_snake_case)]
/// Solve Ax = b using Gaussian elimination with partial pivoting.
/// A is n×n, b is n×1. Returns x.
fn solve_linear(A: &Array2<f64>, b: &ArrayView1<f64>) -> Vec<f64> {
    let n = A.nrows();
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = A[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            continue;
        }
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = if aug[[i, i]].abs() > 1e-14 {
            sum / aug[[i, i]]
        } else {
            0.0
        };
    }
    x
}

/// Sigmoid function.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let neg = (-x).exp();
        1.0 / (1.0 + neg)
    } else {
        let pos = x.exp();
        pos / (1.0 + pos)
    }
}

/// Softmax along last axis. Returns probabilities summing to 1 per row.
fn softmax(logits: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = logits.dim();
    let mut probs = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        let max_val = logits.row(r).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for c in 0..cols {
            let v = (logits[[r, c]] - max_val).exp();
            probs[[r, c]] = v;
            sum += v;
        }
        if sum > 0.0 {
            for c in 0..cols {
                probs[[r, c]] /= sum;
            }
        }
    }
    probs
}

fn softmax_vec(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut values = Vec::with_capacity(logits.len());
    for &logit in logits {
        let value = (logit - max).exp();
        sum += value;
        values.push(value);
    }
    if sum <= 0.0 || !sum.is_finite() {
        return vec![0.0; logits.len()];
    }
    values.into_iter().map(|value| value / sum).collect()
}

// ── Factor activation ──────────────────────────────────────────

/// Build factor activation matrix from discretized label data.
/// Each row in `label_df` is a HashMap<String, String> (feature_name → bin_label).
/// Output: rows×factors binary matrix.
pub fn build_factor_activation(
    label_df: &[HashMap<String, String>],
    factors: &[InducingFactor],
) -> Array2<f64> {
    let n_rows = label_df.len();
    let n_factors = factors.len();
    if n_rows == 0 || n_factors == 0 {
        return Array2::zeros((n_rows, n_factors));
    }

    let mut activation = Array2::<f64>::zeros((n_rows, n_factors));
    for (fi, factor) in factors.iter().enumerate() {
        for (ri, row) in label_df.iter().enumerate() {
            let mut active = true;
            for item in &factor.items {
                // item is "feature=state"
                if let Some((feature, state)) = item.split_once('=') {
                    let val = row.get(feature).map(|s| s.as_str()).unwrap_or("");
                    if val != state {
                        active = false;
                        break;
                    }
                }
            }
            if active {
                activation[[ri, fi]] = 1.0;
            }
        }
    }
    activation
}

/// Build factor activation for a single sample row.
pub fn factor_activation_for_row(
    row: &HashMap<String, String>,
    factors: &[InducingFactor],
) -> Vec<f64> {
    factors
        .iter()
        .map(|factor| {
            let mut active = true;
            for item in &factor.items {
                if let Some((feature, state)) = item.split_once('=') {
                    let val = row.get(feature).map(|s| s.as_str()).unwrap_or("");
                    if val != state {
                        active = false;
                        break;
                    }
                }
            }
            if active {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Build factor activation for a single sample row, returning Vec<i32> (used in Bayesian engine).
pub fn factor_activation_for_row_int(
    row: &HashMap<String, String>,
    factors: &[InducingFactor],
) -> Vec<i32> {
    factors
        .iter()
        .map(|factor| {
            let mut active = true;
            for item in &factor.items {
                if let Some((feature, state)) = item.split_once('=') {
                    let val = row.get(feature).map(|s| s.as_str()).unwrap_or("");
                    if val != state {
                        active = false;
                        break;
                    }
                }
            }
            if active {
                1
            } else {
                0
            }
        })
        .collect()
}

// ── IRLS Logistic Regression ───────────────────────────────────

#[allow(non_snake_case)]
/// Binary logistic regression via IRLS with L2 regularization.
/// Returns (weights, intercept).
pub fn fit_logistic_binary_internal(
    #[allow(non_snake_case)] X: &Array2<f64>,
    y: &[f64],
    class_weight: Option<&[f64]>,
    max_iter: usize,
    l2_penalty: f64,
) -> (Vec<f64>, f64) {
    fit_logistic_binary_internal_impl(X, y, class_weight, max_iter, l2_penalty, None)
}

#[allow(non_snake_case)]
fn fit_logistic_binary_internal_with_progress(
    #[allow(non_snake_case)] X: &Array2<f64>,
    y: &[f64],
    class_weight: Option<&[f64]>,
    max_iter: usize,
    l2_penalty: f64,
    progress: &(dyn Fn(&str) + Sync),
) -> (Vec<f64>, f64) {
    fit_logistic_binary_internal_impl(X, y, class_weight, max_iter, l2_penalty, Some(progress))
}

#[allow(non_snake_case)]
fn fit_logistic_binary_internal_impl(
    #[allow(non_snake_case)] X: &Array2<f64>,
    y: &[f64],
    class_weight: Option<&[f64]>,
    max_iter: usize,
    l2_penalty: f64,
    progress: Option<&(dyn Fn(&str) + Sync)>,
) -> (Vec<f64>, f64) {
    let (n_samples, n_features) = X.dim();
    if n_samples == 0 {
        return (vec![0.0; n_features], 0.0);
    }

    // Include intercept: augment X with column of 1s
    let mut X_aug = Array2::<f64>::ones((n_samples, n_features + 1));
    for i in 0..n_samples {
        for j in 0..n_features {
            X_aug[[i, j]] = X[[i, j]];
        }
    }

    let n_params = n_features + 1;

    // Sample weights
    let has_weights = class_weight.is_some();
    let w: Vec<f64> = if let Some(cw) = class_weight {
        y.iter()
            .map(|yi| if *yi > 0.5 { cw[1] } else { cw[0] })
            .collect()
    } else {
        vec![1.0; n_samples]
    };

    // Precompute balanced total
    let w_sum: f64 = w.iter().sum();
    let default_weight = w_sum / n_samples as f64;

    // Initial beta = zeros
    let mut beta = vec![0.0; n_params];

    // Compute regularization matrix: diagonal with L2 penalty
    // Don't regularize intercept (last parameter)
    let mut reg = Array2::<f64>::zeros((n_params, n_params));
    for i in 0..n_features {
        reg[[i, i]] = l2_penalty;
    }

    for iter in 0..max_iter {
        progress_log(
            progress,
            format!(
                "[numeric-logistic] iteration {}/{} start, rows={}, dimensions={}",
                iter + 1,
                max_iter,
                n_samples,
                n_features
            ),
        );
        // Compute p_i = sigmoid(X_i · beta)
        let mut p = vec![0.0; n_samples];
        let mut has_non_finite = false;
        let mut pred_bucket = usize::MAX;
        for (i, pi) in p.iter_mut().enumerate() {
            let lin: f64 = X_aug
                .row(i)
                .iter()
                .zip(beta.iter())
                .map(|(&x, &b)| x * b)
                .sum();
            // Clamp to avoid overflow
            let clamped = lin.clamp(-30.0, 30.0);
            *pi = sigmoid(clamped);
            if !(*pi).is_finite() {
                has_non_finite = true;
            }
            progress_rows(
                progress,
                "numeric-logistic",
                iter + 1,
                max_iter,
                "prediction pass",
                i + 1,
                n_samples,
                &mut pred_bucket,
            );
        }

        if has_non_finite {
            break;
        }

        // Weighted Hessian: H = X^T W X + reg, where W_ii = w_i * p_i * (1-p_i)
        let mut H = reg.clone();
        let mut hessian_bucket = usize::MAX;
        for i in 0..n_samples {
            let wi = if has_weights { w[i] } else { default_weight };
            let diag = wi * p[i] * (1.0 - p[i]).max(1e-10);
            for j in 0..n_params {
                for k in 0..n_params {
                    H[[j, k]] += diag * X_aug[[i, j]] * X_aug[[i, k]];
                }
            }
            progress_rows(
                progress,
                "numeric-logistic",
                iter + 1,
                max_iter,
                "hessian pass",
                i + 1,
                n_samples,
                &mut hessian_bucket,
            );
        }

        // Gradient: g = X^T W (y - p) - reg · beta
        let mut grad = vec![0.0; n_params];
        let mut gradient_bucket = usize::MAX;
        for i in 0..n_samples {
            let wi = if has_weights { w[i] } else { default_weight };
            let residual = wi * (y[i] - p[i]);
            for j in 0..n_params {
                grad[j] += residual * X_aug[[i, j]];
            }
            progress_rows(
                progress,
                "numeric-logistic",
                iter + 1,
                max_iter,
                "gradient pass",
                i + 1,
                n_samples,
                &mut gradient_bucket,
            );
        }
        // Subtract regularization gradient (not for intercept)
        for j in 0..n_features {
            grad[j] -= l2_penalty * beta[j];
        }

        // Solve H · delta = g
        let delta = solve_linear(&H, &ArrayView1::from(&grad));

        // Update beta
        let mut max_delta: f64 = 0.0;
        for j in 0..n_params {
            beta[j] += delta[j];
            max_delta = max_delta.max(delta[j].abs());
        }

        progress_log(
            progress,
            format!(
                "[numeric-logistic] iteration {}/{} complete, rows={}, max_delta={:.6e}",
                iter + 1,
                max_iter,
                n_samples,
                max_delta
            ),
        );

        if max_delta < 1e-8 {
            progress_log(
                progress,
                format!(
                    "[numeric-logistic] converged at iteration {}/{} with max_delta={:.6e}",
                    iter + 1,
                    max_iter,
                    max_delta
                ),
            );
            break;
        }
    }

    let intercept = beta[n_features];
    let weights = beta[..n_features].to_vec();
    (weights, intercept)
}

struct NumericBatchFit<'a> {
    feature_data: &'a [Vec<f64>],
    target: &'a [f64],
    feature_names: &'a [String],
    means: &'a [f64],
    scales: &'a [f64],
    class_weight: &'a [f64],
    max_epochs: usize,
    learning_rate: f64,
    l2_penalty: f64,
    progress: Option<&'a (dyn Fn(&str) + Sync)>,
}

fn fit_numeric_logistic_parallel(cfg: NumericBatchFit<'_>) -> (Vec<f64>, f64) {
    let n_rows = cfg.feature_data.len();
    let n_cols = cfg.means.len().min(cfg.scales.len());
    if n_rows == 0 || n_cols == 0 {
        return (vec![0.0; n_cols], 0.0);
    }

    let pos = cfg.target.iter().filter(|&&v| v > 0.5).count() as f64;
    let base_rate = (pos / n_rows as f64).clamp(1e-6, 1.0 - 1e-6);
    let mut weights = vec![0.0; n_cols];
    let mut intercept = (base_rate / (1.0 - base_rate)).ln();
    let mut prev_loss = f64::INFINITY;

    for epoch in 0..cfg.max_epochs {
        progress_log(
            cfg.progress,
            format!(
                "[numeric-logistic] epoch {}/{} start, rows={}, dimensions={}",
                epoch + 1,
                cfg.max_epochs,
                n_rows,
                n_cols
            ),
        );

        let mut grad = vec![0.0; n_cols + 1];
        let mut weighted_loss = 0.0;
        let mut pass_bucket = usize::MAX;

        for segment in 0..4 {
            let start = n_rows * segment / 4;
            let end = n_rows * (segment + 1) / 4;
            if start >= end {
                continue;
            }
            let (segment_grad, segment_loss) = (start..end)
                .into_par_iter()
                .fold(
                    || (vec![0.0; n_cols + 1], 0.0f64),
                    |(mut local_grad, mut local_loss), row_idx| {
                        let row = &cfg.feature_data[row_idx];
                        let y = cfg.target[row_idx];
                        let sample_weight = if y > 0.5 {
                            cfg.class_weight.get(1).copied().unwrap_or(1.0)
                        } else {
                            cfg.class_weight.first().copied().unwrap_or(1.0)
                        };
                        let mut lin = intercept;
                        for col_idx in 0..n_cols {
                            let x = scaled_standardize_value(
                                row.get(col_idx).copied().unwrap_or(f64::NAN),
                                cfg.means[col_idx],
                                cfg.scales[col_idx],
                                cfg.feature_names.get(col_idx).map(String::as_str),
                                climate_influence_scale(),
                            );
                            lin += weights[col_idx] * x;
                        }
                        let p = sigmoid(lin.clamp(-30.0, 30.0));
                        let residual = sample_weight * (p - y);
                        for col_idx in 0..n_cols {
                            let x = scaled_standardize_value(
                                row.get(col_idx).copied().unwrap_or(f64::NAN),
                                cfg.means[col_idx],
                                cfg.scales[col_idx],
                                cfg.feature_names.get(col_idx).map(String::as_str),
                                climate_influence_scale(),
                            );
                            local_grad[col_idx] += residual * x;
                        }
                        local_grad[n_cols] += residual;

                        let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
                        local_loss += sample_weight
                            * (-(y * p_clamped.ln() + (1.0 - y) * (1.0 - p_clamped).ln()));
                        (local_grad, local_loss)
                    },
                )
                .reduce(
                    || (vec![0.0; n_cols + 1], 0.0f64),
                    |(mut grad_a, loss_a), (grad_b, loss_b)| {
                        for (a, b) in grad_a.iter_mut().zip(grad_b.iter()) {
                            *a += *b;
                        }
                        (grad_a, loss_a + loss_b)
                    },
                );

            for (dst, src) in grad.iter_mut().zip(segment_grad.iter()) {
                *dst += *src;
            }
            weighted_loss += segment_loss;
            progress_rows(
                cfg.progress,
                "numeric-logistic",
                epoch + 1,
                cfg.max_epochs,
                "gradient pass",
                end,
                n_rows,
                &mut pass_bucket,
            );
        }

        let n_inv = 1.0 / n_rows as f64;
        for col_idx in 0..n_cols {
            grad[col_idx] = grad[col_idx] * n_inv + cfg.l2_penalty * weights[col_idx] * n_inv;
        }
        grad[n_cols] *= n_inv;

        let step = cfg.learning_rate / (1.0 + 0.05 * epoch as f64);
        let mut max_delta: f64 = 0.0;
        for col_idx in 0..n_cols {
            let delta = -step * grad[col_idx];
            weights[col_idx] += delta;
            max_delta = max_delta.max(delta.abs());
        }
        let intercept_delta = -step * grad[n_cols];
        intercept += intercept_delta;
        max_delta = max_delta.max(intercept_delta.abs());

        let l2_loss = 0.5 * cfg.l2_penalty * weights.iter().map(|w| w * w).sum::<f64>();
        let mean_loss = (weighted_loss + l2_loss) * n_inv;
        progress_log(
            cfg.progress,
            format!(
                "[numeric-logistic] epoch {}/{} complete, rows={}, mean_loss={:.6}, max_delta={:.6e}",
                epoch + 1,
                cfg.max_epochs,
                n_rows,
                mean_loss,
                max_delta
            ),
        );

        if (prev_loss - mean_loss).abs() < 1e-7 || max_delta < 1e-7 {
            progress_log(
                cfg.progress,
                format!(
                    "[numeric-logistic] converged at epoch {}/{} with mean_loss={:.6}, max_delta={:.6e}",
                    epoch + 1,
                    cfg.max_epochs,
                    mean_loss,
                    max_delta
                ),
            );
            break;
        }
        prev_loss = mean_loss;
    }

    (weights, intercept)
}

struct NumericInteractionBatchFit<'a> {
    feature_data: &'a [Vec<f64>],
    target: &'a [f64],
    feature_names: &'a [String],
    means: &'a [f64],
    scales: &'a [f64],
    class_weight: &'a [f64],
    initial_weights: &'a [f64],
    initial_intercept: f64,
    baseline_weights: &'a [f64],
    interaction_terms: &'a [NumericInteractionTerm],
    activation_gamma: f64,
    max_epochs: usize,
    learning_rate: f64,
    l2_penalty: f64,
    progress: Option<&'a (dyn Fn(&str) + Sync)>,
}

fn fit_numeric_interaction_logistic_parallel(
    cfg: NumericInteractionBatchFit<'_>,
) -> (Vec<f64>, f64, Vec<f64>) {
    let n_rows = cfg.feature_data.len();
    let n_cols = cfg
        .means
        .len()
        .min(cfg.scales.len())
        .min(cfg.initial_weights.len())
        .min(cfg.baseline_weights.len());
    let n_interactions = cfg.interaction_terms.len();
    if n_rows == 0 || n_cols == 0 {
        return (vec![0.0; n_cols], 0.0, vec![0.0; n_interactions]);
    }

    let mut weights = cfg.initial_weights[..n_cols].to_vec();
    let mut interaction_weights = vec![0.0; n_interactions];
    let mut intercept = cfg.initial_intercept;
    let mut prev_loss = f64::INFINITY;

    for epoch in 0..cfg.max_epochs {
        progress_log(
            cfg.progress,
            format!(
                "[interaction-logistic] epoch {}/{} start, rows={}, main_dimensions={}, interaction_terms={}",
                epoch + 1,
                cfg.max_epochs,
                n_rows,
                n_cols,
                n_interactions
            ),
        );

        let grad_len = n_cols + n_interactions + 1;
        let mut grad = vec![0.0; grad_len];
        let mut weighted_loss = 0.0;
        let mut pass_bucket = usize::MAX;

        for segment in 0..4 {
            let start = n_rows * segment / 4;
            let end = n_rows * (segment + 1) / 4;
            if start >= end {
                continue;
            }

            let (segment_grad, segment_loss) = (start..end)
                .into_par_iter()
                .fold(
                    || (vec![0.0; grad_len], 0.0f64),
                    |(mut local_grad, mut local_loss), row_idx| {
                        let row = &cfg.feature_data[row_idx];
                        let y = cfg.target[row_idx];
                        let sample_weight = if y > 0.5 {
                            cfg.class_weight.get(1).copied().unwrap_or(1.0)
                        } else {
                            cfg.class_weight.first().copied().unwrap_or(1.0)
                        };

                        let mut standardized = vec![0.0; n_cols];
                        let mut lin = intercept;
                        for col_idx in 0..n_cols {
                            let z = scaled_standardize_value(
                                row.get(col_idx).copied().unwrap_or(f64::NAN),
                                cfg.means[col_idx],
                                cfg.scales[col_idx],
                                cfg.feature_names.get(col_idx).map(String::as_str),
                                climate_influence_scale(),
                            );
                            standardized[col_idx] = z;
                            lin += weights[col_idx] * z;
                        }

                        for (term_idx, term) in cfg.interaction_terms.iter().enumerate() {
                            let value = interaction_activation_product(
                                &standardized,
                                cfg.baseline_weights,
                                &term.feature_indices,
                                cfg.activation_gamma,
                            );
                            lin += interaction_weights[term_idx] * value;
                        }

                        let p = sigmoid(lin.clamp(-30.0, 30.0));
                        let residual = sample_weight * (p - y);
                        for col_idx in 0..n_cols {
                            local_grad[col_idx] += residual * standardized[col_idx];
                        }
                        for term_idx in 0..n_interactions {
                            let value = interaction_activation_product(
                                &standardized,
                                cfg.baseline_weights,
                                &cfg.interaction_terms[term_idx].feature_indices,
                                cfg.activation_gamma,
                            );
                            local_grad[n_cols + term_idx] += residual * value;
                        }
                        local_grad[grad_len - 1] += residual;

                        let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
                        local_loss += sample_weight
                            * (-(y * p_clamped.ln() + (1.0 - y) * (1.0 - p_clamped).ln()));
                        (local_grad, local_loss)
                    },
                )
                .reduce(
                    || (vec![0.0; grad_len], 0.0f64),
                    |(mut grad_a, loss_a), (grad_b, loss_b)| {
                        for (a, b) in grad_a.iter_mut().zip(grad_b.iter()) {
                            *a += *b;
                        }
                        (grad_a, loss_a + loss_b)
                    },
                );

            for (dst, src) in grad.iter_mut().zip(segment_grad.iter()) {
                *dst += *src;
            }
            weighted_loss += segment_loss;
            progress_rows(
                cfg.progress,
                "interaction-logistic",
                epoch + 1,
                cfg.max_epochs,
                "gradient pass",
                end,
                n_rows,
                &mut pass_bucket,
            );
        }

        let n_inv = 1.0 / n_rows as f64;
        for col_idx in 0..n_cols {
            grad[col_idx] = grad[col_idx] * n_inv + cfg.l2_penalty * weights[col_idx] * n_inv;
        }
        for term_idx in 0..n_interactions {
            let offset = n_cols + term_idx;
            grad[offset] =
                grad[offset] * n_inv + cfg.l2_penalty * interaction_weights[term_idx] * n_inv;
        }
        grad[grad_len - 1] *= n_inv;

        let step = cfg.learning_rate / (1.0 + 0.05 * epoch as f64);
        let mut max_delta: f64 = 0.0;
        for col_idx in 0..n_cols {
            let delta = step * grad[col_idx];
            weights[col_idx] -= delta;
            max_delta = max_delta.max(delta.abs());
        }
        for term_idx in 0..n_interactions {
            let offset = n_cols + term_idx;
            let delta = step * grad[offset];
            interaction_weights[term_idx] -= delta;
            max_delta = max_delta.max(delta.abs());
        }
        let intercept_delta = step * grad[grad_len - 1];
        intercept -= intercept_delta;
        max_delta = max_delta.max(intercept_delta.abs());

        let l2 = weights.iter().map(|w| w * w).sum::<f64>()
            + interaction_weights.iter().map(|w| w * w).sum::<f64>();
        let avg_loss = weighted_loss * n_inv + 0.5 * cfg.l2_penalty * l2 * n_inv;
        progress_log(
            cfg.progress,
            format!(
                "[interaction-logistic] epoch {}/{} complete: loss={:.6}, step={:.5}, max_delta={:.6e}",
                epoch + 1,
                cfg.max_epochs,
                avg_loss,
                step,
                max_delta
            ),
        );
        if (prev_loss - avg_loss).abs() < 1e-7 || max_delta < 1e-7 {
            progress_log(
                cfg.progress,
                format!(
                    "[interaction-logistic] converged at epoch {}/{}",
                    epoch + 1,
                    cfg.max_epochs
                ),
            );
            break;
        }
        prev_loss = avg_loss;
    }

    (weights, intercept, interaction_weights)
}

fn interaction_activation_product(
    standardized: &[f64],
    baseline_weights: &[f64],
    indices: &[usize],
    gamma: f64,
) -> f64 {
    let mut product = 1.0;
    for &idx in indices {
        let Some(&z) = standardized.get(idx) else {
            return 0.0;
        };
        let direction = if baseline_weights.get(idx).copied().unwrap_or(0.0) < 0.0 {
            -1.0
        } else {
            1.0
        };
        product *= sigmoid(gamma * direction * z);
    }
    product
}

/// Multinomial (softmax) logistic regression via gradient-based optimization.
/// One-vs-rest approach: fit K-1 binary classifiers, then softmax.
fn fit_logistic_multiclass(
    #[allow(non_snake_case)] X: &Array2<f64>,
    y: &[f64],
    classes: &[i64],
    class_weight: Option<&[f64]>,
    max_iter: usize,
    l2_penalty: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (n_samples, n_features) = X.dim();
    let n_classes = classes.len();

    if n_samples == 0 || n_classes <= 1 {
        let weights = vec![vec![0.0; n_features]; n_classes];
        let intercepts = vec![0.0; n_classes];
        return (weights, intercepts);
    }

    // For each class, fit a binary logistic regression (one-vs-rest)
    let mut all_weights = Vec::with_capacity(n_classes);
    let mut all_intercepts = Vec::with_capacity(n_classes);

    for target_class in classes.iter() {
        let target_class = *target_class as f64;
        let y_binary: Vec<f64> = y
            .iter()
            .map(|&val| {
                if (val - target_class).abs() < 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Check if all labels are the same
        let n_pos: f64 = y_binary.iter().sum();
        let n_neg = n_samples as f64 - n_pos;

        if n_pos < 1.0 || n_neg < 1.0 {
            // Single class: all weights = 0, intercept = logit of base rate
            let base_rate = n_pos / (n_samples as f64).max(1.0);
            let logit = (base_rate / (1.0 - base_rate).max(1e-10)).ln();
            all_weights.push(vec![0.0; n_features]);
            all_intercepts.push(logit);
        } else {
            let cw: Option<Vec<f64>> = class_weight.map(|_cw| {
                let pos_w = n_samples as f64 / (2.0 * n_pos.max(1.0));
                let neg_w = n_samples as f64 / (2.0 * n_neg.max(1.0));
                vec![neg_w, pos_w]
            });
            let cw_ref = cw.as_deref();
            let (w, intercept) =
                fit_logistic_binary_internal(X, &y_binary, cw_ref, max_iter, l2_penalty);
            all_weights.push(w);
            all_intercepts.push(intercept);
        }
    }

    // Calibrate intercepts so that softmax produces well-calibrated probabilities
    // For simplicity, we leave them as-is (sklearn's multinomial logistic does joint
    // optimization; our one-vs-rest approximation is close enough for golden parity)

    (all_weights, all_intercepts)
}

/// Predict class probabilities via softmax.
#[allow(non_snake_case)]
fn predict_softmax_proba(X: &Array2<f64>, weights: &[Vec<f64>], intercepts: &[f64]) -> Array2<f64> {
    let n_samples = X.nrows();
    let n_classes = weights.len();
    let n_features = X.ncols();

    let mut logits = Array2::<f64>::zeros((n_samples, n_classes));
    for i in 0..n_samples {
        for ci in 0..n_classes {
            let mut lin = intercepts[ci];
            for j in 0..n_features {
                lin += weights[ci][j] * X[[i, j]];
            }
            logits[[i, ci]] = lin;
        }
    }
    softmax(&logits)
}

// ── OccurrenceProbabilityEngine ────────────────────────────────

/// Continuous high-order interaction term used by the numeric occurrence model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericInteractionTerm {
    pub id: String,
    pub order: usize,
    pub feature_indices: Vec<usize>,
    pub feature_names: Vec<String>,
    pub weight: f64,
    pub support_pos: f64,
    pub support_neg: f64,
    pub lift: f64,
    pub log_lift: f64,
    pub stability: f64,
    pub score: f64,
    #[serde(default)]
    pub risk_thresholds: Vec<NumericRiskThreshold>,
    #[serde(default)]
    pub activation_midpoints: Vec<NumericRiskThreshold>,
    #[serde(default)]
    pub learned_thresholds: Vec<FeatureThreshold>,
    #[serde(default)]
    pub explanation_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericRiskThreshold {
    pub feature_index: usize,
    pub feature_name: String,
    pub direction: String,
    pub midpoint_value: f64,
    pub z_midpoint: f64,
    pub activation_midpoint: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedBin {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub pos_rate: f64,
    pub neg_rate: f64,
    pub lift: f64,
    pub count_pos: usize,
    pub count_neg: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureThreshold {
    pub feature_name: String,
    pub direction: String,
    pub activation_midpoint: f64,
    pub learned_threshold: Option<f64>,
    #[serde(default)]
    pub learned_bins: Vec<LearnedBin>,
    pub method: String,
    pub support_pos: f64,
    pub support_neg: f64,
    pub lift: f64,
    pub youden_j: f64,
    pub information_gain: f64,
    pub auc_1d: f64,
    pub confidence: String,
    #[serde(default)]
    pub warning: Option<String>,
    #[serde(default)]
    pub explanation_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct InteractionSearchConfig {
    pub algorithm: String,
    pub max_order: usize,
    pub min_support_pos: f64,
    pub min_lift: f64,
    pub pair_beam: usize,
    pub triple_beam: usize,
    pub quad_beam: usize,
    pub final_top: usize,
    pub order2_bonus: f64,
    pub order3_bonus: f64,
    pub order4_bonus: f64,
    #[serde(default = "default_climate_influence_scale")]
    pub climate_influence_scale: f64,
    #[serde(default = "default_climate_only_interaction_penalty")]
    pub climate_only_interaction_penalty: f64,
    #[serde(default = "default_mixed_climate_interaction_penalty")]
    pub mixed_climate_interaction_penalty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct InteractionSearchStats {
    pub order2_total: usize,
    pub strong_pairs: usize,
    pub generated_triples_from_beam: usize,
    pub strong_triples: usize,
    pub generated_quads_from_beam: usize,
    pub strong_quads: usize,
    pub final_interactions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TypeClassDiagnostic {
    pub class_value: i64,
    pub label: String,
    pub positive_count: usize,
    pub negative_count: usize,
    pub positive_rate: f64,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DisturbanceTypeDiagnostics {
    pub total_labeled: usize,
    #[serde(default)]
    pub classes: Vec<TypeClassDiagnostic>,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub probability_semantics: String,
}

/// Engine for predicting disturbance occurrence probability.
/// Trains a logistic regression on factor activations (binary matrix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccurrenceProbabilityEngine {
    /// Logistic regression coefficient for each factor
    pub weights: Vec<f64>,
    /// Intercept term
    pub intercept: f64,
    /// Factor IDs used in this model
    pub factor_ids: Vec<String>,
    /// Backend identifier
    pub backend: String,
    /// Base rate for fallback (when only one class present)
    pub base_rate: f64,
    /// Numeric feature columns used by vector-space logistic regression.
    #[serde(default)]
    pub numeric_feature_columns: Vec<String>,
    /// Per-feature training mean used for missing-value imputation.
    #[serde(default)]
    pub numeric_means: Vec<f64>,
    /// Per-feature training scale used for z-score normalization.
    #[serde(default)]
    pub numeric_scales: Vec<f64>,
    /// Baseline numeric weights used to orient continuous risk activations.
    #[serde(default)]
    pub baseline_weights: Vec<f64>,
    /// Continuous interaction terms included in the enhanced numeric model.
    #[serde(default)]
    pub interactions: Vec<NumericInteractionTerm>,
    /// Activation strength for risk-side fuzzy interactions.
    #[serde(default = "default_activation_gamma")]
    pub activation_gamma: f64,
    /// Multiplicative scale applied to standardized rainfall/temperature features.
    #[serde(default = "default_climate_influence_scale")]
    pub climate_influence_scale: f64,
    /// Mean-based activation midpoints. These are not learned risk thresholds.
    #[serde(default)]
    pub activation_midpoints: Vec<NumericRiskThreshold>,
    /// Sample-learned thresholds and bins used only for explanation.
    #[serde(default)]
    pub learned_thresholds: Vec<FeatureThreshold>,
    #[serde(default)]
    pub interaction_search_config: InteractionSearchConfig,
    #[serde(default)]
    pub interaction_search_stats: InteractionSearchStats,
}

impl OccurrenceProbabilityEngine {
    /// Create a new engine with fallback backend.
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            intercept: 0.0,
            factor_ids: Vec::new(),
            backend: "logistic_regression".to_string(),
            base_rate: 0.0,
            numeric_feature_columns: Vec::new(),
            numeric_means: Vec::new(),
            numeric_scales: Vec::new(),
            baseline_weights: Vec::new(),
            interactions: Vec::new(),
            activation_gamma: default_activation_gamma(),
            climate_influence_scale: default_climate_influence_scale(),
            activation_midpoints: Vec::new(),
            learned_thresholds: Vec::new(),
            interaction_search_config: InteractionSearchConfig::default(),
            interaction_search_stats: InteractionSearchStats::default(),
        }
    }

    /// Fit a logistic regression directly on numeric feature vectors.
    ///
    /// Missing and non-finite values are imputed with the training mean, then
    /// each feature is z-score normalized. Prediction must use the stored
    /// means/scales to stay in the same high-dimensional feature space.
    pub fn fit_numeric(
        &mut self,
        feature_columns: &[String],
        feature_data: &[Vec<f64>],
        target: &[f64],
        random_state: u64,
    ) {
        self.fit_numeric_impl(feature_columns, feature_data, target, random_state, None);
    }

    pub fn fit_numeric_with_progress(
        &mut self,
        feature_columns: &[String],
        feature_data: &[Vec<f64>],
        target: &[f64],
        random_state: u64,
        progress: &(dyn Fn(&str) + Sync),
    ) {
        self.fit_numeric_impl(
            feature_columns,
            feature_data,
            target,
            random_state,
            Some(progress),
        );
    }

    fn fit_numeric_impl(
        &mut self,
        feature_columns: &[String],
        feature_data: &[Vec<f64>],
        target: &[f64],
        random_state: u64,
        progress: Option<&(dyn Fn(&str) + Sync)>,
    ) {
        self.factor_ids.clear();
        self.numeric_feature_columns = feature_columns.to_vec();
        let n_rows = feature_data.len();
        let n_cols = feature_columns.len();
        self.numeric_means = numeric_means(feature_data, n_cols);
        self.numeric_scales = numeric_scales(feature_data, &self.numeric_means, n_cols);

        let n_pos: f64 = target.iter().filter(|&&v| v > 0.5).count() as f64;
        let n_neg = target.len() as f64 - n_pos;
        self.base_rate = n_pos / (target.len() as f64).max(1.0);

        if n_rows == 0 || n_cols == 0 || n_pos < 1.0 || n_neg < 1.0 {
            self.weights = vec![0.0; n_cols];
            self.intercept = if self.base_rate > 0.0 && self.base_rate < 1.0 {
                (self.base_rate / (1.0 - self.base_rate)).ln()
            } else if self.base_rate >= 1.0 {
                10.0
            } else {
                -10.0
            };
            self.backend = "numeric_logistic_fallback_single_class".to_string();
            return;
        }

        let pos_weight = target.len() as f64 / (2.0 * n_pos.max(1.0));
        let neg_weight = target.len() as f64 / (2.0 * n_neg.max(1.0));
        let class_weight = vec![neg_weight, pos_weight];
        let _seed = random_state;

        let (weights, intercept) = fit_numeric_logistic_parallel(NumericBatchFit {
            feature_data,
            target,
            feature_names: &self.numeric_feature_columns,
            means: &self.numeric_means,
            scales: &self.numeric_scales,
            class_weight: &class_weight,
            max_epochs: 40,
            learning_rate: 0.8,
            l2_penalty: 1.0,
            progress,
        });
        self.weights = weights;
        self.intercept = intercept;
        self.baseline_weights = self.weights.clone();
        self.interactions.clear();
        self.activation_gamma = default_activation_gamma();
        self.climate_influence_scale = climate_influence_scale();
        self.activation_midpoints.clear();
        self.learned_thresholds.clear();
        self.backend = "numeric_parallel_logistic_regression".to_string();
    }

    pub fn fit_numeric_interactions_with_progress(
        &mut self,
        feature_data: &[Vec<f64>],
        target: &[f64],
        interaction_terms: &[NumericInteractionTerm],
        activation_gamma: f64,
        progress: &(dyn Fn(&str) + Sync),
    ) {
        let n_cols = self
            .numeric_feature_columns
            .len()
            .min(self.weights.len())
            .min(self.numeric_means.len())
            .min(self.numeric_scales.len());
        let mut terms: Vec<NumericInteractionTerm> = interaction_terms
            .iter()
            .filter(|term| {
                !term.feature_indices.is_empty()
                    && term.feature_indices.iter().all(|&idx| idx < n_cols)
            })
            .cloned()
            .collect();
        if n_cols == 0 || feature_data.is_empty() || target.is_empty() || terms.is_empty() {
            self.baseline_weights = self.weights.clone();
            self.interactions.clear();
            self.activation_gamma = activation_gamma;
            self.climate_influence_scale = climate_influence_scale();
            return;
        }

        let baseline_weights = self.weights[..n_cols].to_vec();
        let n_pos: f64 = target.iter().filter(|&&v| v > 0.5).count() as f64;
        let n_neg = target.len() as f64 - n_pos;
        if n_pos < 1.0 || n_neg < 1.0 {
            self.baseline_weights = baseline_weights;
            self.interactions.clear();
            self.activation_gamma = activation_gamma;
            self.climate_influence_scale = climate_influence_scale();
            return;
        }

        let pos_weight = target.len() as f64 / (2.0 * n_pos.max(1.0));
        let neg_weight = target.len() as f64 / (2.0 * n_neg.max(1.0));
        let class_weight = vec![neg_weight, pos_weight];

        let max_epochs = std::env::var("INTERACTION_LOGISTIC_EPOCHS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(15);
        let (weights, intercept, interaction_weights) =
            fit_numeric_interaction_logistic_parallel(NumericInteractionBatchFit {
                feature_data,
                target,
                feature_names: &self.numeric_feature_columns[..n_cols],
                means: &self.numeric_means[..n_cols],
                scales: &self.numeric_scales[..n_cols],
                class_weight: &class_weight,
                initial_weights: &self.weights[..n_cols],
                initial_intercept: self.intercept,
                baseline_weights: &baseline_weights,
                interaction_terms: &terms,
                activation_gamma,
                max_epochs,
                learning_rate: 0.5,
                l2_penalty: 1.0,
                progress: Some(progress),
            });
        for (term, weight) in terms.iter_mut().zip(interaction_weights) {
            term.weight = weight;
        }
        self.weights = weights;
        self.intercept = intercept;
        self.baseline_weights = baseline_weights;
        self.interactions = terms;
        self.activation_gamma = activation_gamma;
        self.climate_influence_scale = climate_influence_scale();
        self.backend = "numeric_logistic_regression_with_interactions".to_string();
    }

    pub fn predict_numeric_rows(&self, feature_data: &[Vec<f64>]) -> Vec<f64> {
        if feature_data.is_empty() {
            return Vec::new();
        }
        let n_cols = self
            .numeric_feature_columns
            .len()
            .min(self.weights.len())
            .min(self.numeric_means.len())
            .min(self.numeric_scales.len());
        feature_data
            .par_iter()
            .map(|row| {
                let (lin, _) = self.numeric_logit_and_dominant(row, n_cols);
                sigmoid(lin.clamp(-30.0, 30.0))
            })
            .collect()
    }

    pub fn predict_numeric_rows_with_dominant(
        &self,
        feature_data: &[Vec<f64>],
    ) -> (Vec<f64>, Vec<i16>) {
        if feature_data.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let n_cols = self
            .numeric_feature_columns
            .len()
            .min(self.weights.len())
            .min(self.numeric_means.len())
            .min(self.numeric_scales.len());
        let pairs: Vec<(f64, i16)> = feature_data
            .par_iter()
            .map(|row| {
                let (lin, dominant) = self.numeric_logit_and_dominant(row, n_cols);
                (sigmoid(lin.clamp(-30.0, 30.0)), dominant)
            })
            .collect();
        pairs.into_iter().unzip()
    }

    fn numeric_logit_and_dominant(&self, row: &[f64], n_cols: usize) -> (f64, i16) {
        let mut standardized = vec![0.0; n_cols];
        let mut lin = self.intercept;
        let mut best_value = 0.0;
        let mut best_index = 0i16;
        let mut best_priority = 0usize;
        for col_idx in 0..n_cols {
            let z = scaled_standardize_value(
                row.get(col_idx).copied().unwrap_or(f64::NAN),
                self.numeric_means[col_idx],
                self.numeric_scales[col_idx],
                self.numeric_feature_columns.get(col_idx).map(String::as_str),
                self.climate_influence_scale,
            );
            standardized[col_idx] = z;
            let contribution = self.weights[col_idx] * z;
            lin += contribution;
            let is_climate = self
                .numeric_feature_columns
                .get(col_idx)
                .map(|name| is_climate_feature_name(name))
                .unwrap_or(false);
            if !is_climate
                && contribution > 0.0
                && (best_priority < 1 || (best_priority == 1 && contribution > best_value))
            {
                best_value = contribution;
                best_index = (col_idx + 1) as i16;
                best_priority = 1;
            }
        }

        let baseline_weights = if self.baseline_weights.len() >= n_cols {
            &self.baseline_weights[..n_cols]
        } else {
            &self.weights[..n_cols]
        };
        let dominant_interaction_limit = std::env::var("DOMINANT_INTERACTION_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(64);
        for (term_idx, term) in self
            .interactions
            .iter()
            .take(dominant_interaction_limit)
            .enumerate()
        {
            let activation = interaction_activation_product(
                &standardized,
                baseline_weights,
                &term.feature_indices,
                self.activation_gamma,
            );
            let contribution = term.weight * activation;
            lin += contribution;
            let priority = interaction_dominance_priority(term.order);
            if contribution > 0.0
                && priority > 0
                && (priority > best_priority
                    || (priority == best_priority && contribution > best_value))
            {
                best_value = contribution;
                best_index = (n_cols + term_idx + 1) as i16;
                best_priority = priority;
            }
        }
        (lin, best_index)
    }

    pub fn is_numeric_backend(&self) -> bool {
        self.backend.starts_with("numeric_") && !self.numeric_feature_columns.is_empty()
    }

    /// Fit the model on discretized label data.
    ///
    /// `label_df`: each row is `HashMap<feature_name, bin_label_string>`
    /// `target`: disturbance_flag (0 or 1)
    /// `factors`: selected inducing factors
    pub fn fit(
        &mut self,
        label_df: &[HashMap<String, String>],
        target: &[f64],
        factors: &[InducingFactor],
        random_state: u64,
    ) {
        self.factor_ids = factors.iter().map(|f| f.factor_id.clone()).collect();
        let x_matrix = build_factor_activation(label_df, factors);

        let n_pos: f64 = target.iter().filter(|&&v| v > 0.5).count() as f64;
        let n_neg = target.len() as f64 - n_pos;

        // Check for single-class training
        if n_pos < 1.0 || n_neg < 1.0 {
            self.base_rate = if target.is_empty() {
                0.0
            } else {
                target.iter().sum::<f64>() / (target.len() as f64)
            };
            self.weights = vec![0.0; factors.len()];
            self.intercept = if self.base_rate > 0.0 && self.base_rate < 1.0 {
                (self.base_rate / (1.0 - self.base_rate)).ln()
            } else if self.base_rate >= 1.0 {
                10.0 // all positive → high logit
            } else {
                -10.0 // all negative → low logit
            };
            self.backend = "logistic_fallback_single_class".to_string();
            return;
        }

        // Class weights for balanced learning
        let pos_weight = target.len() as f64 / (2.0 * n_pos.max(1.0));
        let neg_weight = target.len() as f64 / (2.0 * n_neg.max(1.0));
        let class_weight = vec![neg_weight, pos_weight];

        // Suppress unused variable warning
        let _seed = random_state;

        let (weights, intercept) =
            fit_logistic_binary_internal(&x_matrix, target, Some(&class_weight), 100, 1.0);

        self.weights = weights;
        self.intercept = intercept;
        self.base_rate = n_pos / (target.len() as f64).max(1.0);
        self.backend = "logistic_regression".to_string();
    }

    /// Predict probability of disturbance (class 1) for each sample.
    /// Returns Vec<f64> where each value is in [0, 1].
    pub fn predict_proba(
        &self,
        label_df: &[HashMap<String, String>],
        factors: &[InducingFactor],
    ) -> Vec<f64> {
        let n = label_df.len();
        if n == 0 {
            return Vec::new();
        }

        let x_matrix = build_factor_activation(label_df, factors);
        let mut probs = Vec::with_capacity(n);

        for i in 0..n {
            let mut lin = self.intercept;
            for j in 0..self.weights.len().min(x_matrix.ncols()) {
                lin += self.weights[j] * x_matrix[[i, j]];
            }
            probs.push(sigmoid(lin.clamp(-30.0, 30.0)));
        }
        probs
    }

    /// Predict probability for a single sample row.
    pub fn predict_single(&self, row: &HashMap<String, String>, factors: &[InducingFactor]) -> f64 {
        let activation = factor_activation_for_row(row, factors);
        let mut lin = self.intercept;
        for (j, &val) in activation.iter().enumerate() {
            if j < self.weights.len() {
                lin += self.weights[j] * val;
            }
        }
        sigmoid(lin.clamp(-30.0, 30.0))
    }
}

fn interaction_dominance_priority(order: usize) -> usize {
    match order {
        3 => 3,
        2 => 2,
        _ => 0,
    }
}

fn is_climate_feature_name(name: &str) -> bool {
    name.starts_with("rainfall_") || name.starts_with("temp_")
}

impl Default for OccurrenceProbabilityEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn numeric_means(feature_data: &[Vec<f64>], n_cols: usize) -> Vec<f64> {
    (0..n_cols)
        .map(|col_idx| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for row in feature_data {
                let value = row.get(col_idx).copied().unwrap_or(f64::NAN);
                if value.is_finite() {
                    sum += value;
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                sum / count as f64
            }
        })
        .collect()
}

fn numeric_scales(feature_data: &[Vec<f64>], means: &[f64], n_cols: usize) -> Vec<f64> {
    (0..n_cols)
        .map(|col_idx| {
            let mean = means.get(col_idx).copied().unwrap_or(0.0);
            let mut sum_sq = 0.0;
            let mut count = 0usize;
            for row in feature_data {
                let value = row.get(col_idx).copied().unwrap_or(f64::NAN);
                if value.is_finite() {
                    let centered = value - mean;
                    sum_sq += centered * centered;
                    count += 1;
                }
            }
            if count <= 1 {
                1.0
            } else {
                let std = (sum_sq / count as f64).sqrt();
                if std.is_finite() && std > 1e-12 {
                    std
                } else {
                    1.0
                }
            }
        })
        .collect()
}

fn standardize_value(value: f64, mean: f64, scale: f64) -> f64 {
    let filled = if value.is_finite() { value } else { mean };
    let scale = if scale.is_finite() && scale > 1e-12 {
        scale
    } else {
        1.0
    };
    ((filled - mean) / scale).clamp(-10.0, 10.0)
}

fn scaled_standardize_value(
    value: f64,
    mean: f64,
    scale: f64,
    feature_name: Option<&str>,
    climate_scale: f64,
) -> f64 {
    let z = standardize_value(value, mean, scale);
    if feature_name
        .map(is_climate_feature_name)
        .unwrap_or(false)
    {
        z * climate_scale.clamp(0.0, 1.0)
    } else {
        z
    }
}

fn default_climate_influence_scale() -> f64 {
    1.0
}

fn default_climate_only_interaction_penalty() -> f64 {
    0.35
}

fn default_mixed_climate_interaction_penalty() -> f64 {
    0.75
}

fn climate_influence_scale() -> f64 {
    std::env::var("CLIMATE_INFLUENCE_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.65)
}

fn progress_log(progress: Option<&(dyn Fn(&str) + Sync)>, message: String) {
    if let Some(progress) = progress {
        progress(&message);
    }
}

fn progress_rows(
    progress: Option<&(dyn Fn(&str) + Sync)>,
    component: &str,
    iteration: usize,
    max_iterations: usize,
    pass: &str,
    completed_rows: usize,
    total_rows: usize,
    last_bucket: &mut usize,
) {
    if progress.is_none() || total_rows == 0 {
        return;
    }
    let bucket = if completed_rows >= total_rows {
        4
    } else {
        (completed_rows.saturating_mul(4)) / total_rows
    };
    if bucket == 0 && completed_rows < total_rows {
        return;
    }
    if *last_bucket != usize::MAX && bucket <= *last_bucket && completed_rows < total_rows {
        return;
    }
    *last_bucket = bucket;
    let prefix = if iteration > 0 && max_iterations > 0 {
        format!("{component} iteration {iteration}/{max_iterations}")
    } else {
        component.to_string()
    };
    progress_log(
        progress,
        format!(
            "[{prefix}] {pass}: rows {completed_rows}/{total_rows} ({:.1}%)",
            100.0 * completed_rows as f64 / total_rows as f64
        ),
    );
}

// ── DisturbanceClassEngine ─────────────────────────────────────

/// Engine for predicting disturbance class (fire, logging, geo_hazard, road_construction).
/// Trains a multinomial logistic regression on factor activations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisturbanceClassEngine {
    /// Per-class weight vectors [n_classes][n_features]
    pub weights: Vec<Vec<f64>>,
    /// Per-class intercepts
    pub intercepts: Vec<f64>,
    /// Sorted class values (e.g., [1, 3, 4])
    pub classes: Vec<i64>,
    /// Factor IDs used in this model
    pub factor_ids: Vec<String>,
    /// Class label mapping (1→fire, 2→logging, etc.)
    pub class_labels: HashMap<i64, String>,
    /// Backend identifier
    pub backend: String,
    /// Prior probabilities for each class (for fallback)
    pub priors: HashMap<i64, f64>,
    /// Numeric feature columns used by numeric multiclass classification.
    #[serde(default)]
    pub numeric_feature_columns: Vec<String>,
    /// Per-feature training mean inherited from the occurrence model.
    #[serde(default)]
    pub numeric_means: Vec<f64>,
    /// Per-feature training scale inherited from the occurrence model.
    #[serde(default)]
    pub numeric_scales: Vec<f64>,
    /// Baseline occurrence weights used to orient shared interaction activations.
    #[serde(default)]
    pub baseline_weights: Vec<f64>,
    /// Shared interaction terms used as numeric multiclass features.
    #[serde(default)]
    pub interactions: Vec<NumericInteractionTerm>,
    /// Activation strength for shared interaction features.
    #[serde(default = "default_activation_gamma")]
    pub activation_gamma: f64,
    /// Multiplicative scale applied to standardized rainfall/temperature features.
    #[serde(default = "default_climate_influence_scale")]
    pub climate_influence_scale: f64,
}

impl DisturbanceClassEngine {
    pub fn new() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(1, "fire".to_string());
        class_labels.insert(2, "logging".to_string());
        class_labels.insert(3, "geo_hazard".to_string());
        class_labels.insert(4, "road_construction".to_string());
        Self {
            weights: Vec::new(),
            intercepts: Vec::new(),
            classes: Vec::new(),
            factor_ids: Vec::new(),
            class_labels,
            backend: "multinomial_logistic".to_string(),
            priors: HashMap::new(),
            numeric_feature_columns: Vec::new(),
            numeric_means: Vec::new(),
            numeric_scales: Vec::new(),
            baseline_weights: Vec::new(),
            interactions: Vec::new(),
            activation_gamma: default_activation_gamma(),
            climate_influence_scale: default_climate_influence_scale(),
        }
    }

    pub fn fit_numeric_with_interactions(
        &mut self,
        feature_columns: &[String],
        feature_data: &[Vec<f64>],
        class_values: &[f64],
        class_mask: &[usize],
        occurrence_engine: &OccurrenceProbabilityEngine,
        progress: &(dyn Fn(&str) + Sync),
    ) {
        self.numeric_feature_columns = feature_columns.to_vec();
        self.numeric_means = occurrence_engine.numeric_means.clone();
        self.numeric_scales = occurrence_engine.numeric_scales.clone();
        self.baseline_weights = if occurrence_engine.baseline_weights.is_empty() {
            occurrence_engine.weights.clone()
        } else {
            occurrence_engine.baseline_weights.clone()
        };
        self.interactions = occurrence_engine.interactions.clone();
        self.activation_gamma = occurrence_engine.activation_gamma;
        self.climate_influence_scale = occurrence_engine.climate_influence_scale;
        self.factor_ids.clear();

        if class_mask.is_empty() || class_values.is_empty() {
            self.backend = "numeric_multiclass_fallback_no_data".to_string();
            return;
        }

        let mut unique_classes: Vec<i64> = class_values.iter().map(|&v| v as i64).collect();
        unique_classes.sort();
        unique_classes.dedup();
        self.classes = unique_classes.clone();

        let n_total = class_values.len() as f64;
        self.priors.clear();
        for &cls in &unique_classes {
            let count = class_values.iter().filter(|&&v| (v as i64) == cls).count() as f64;
            self.priors.insert(cls, count / n_total.max(1.0));
        }

        let n_cols = feature_columns
            .len()
            .min(self.numeric_means.len())
            .min(self.numeric_scales.len())
            .min(self.baseline_weights.len());
        if unique_classes.is_empty() || n_cols == 0 {
            self.backend = "numeric_multiclass_fallback_no_data".to_string();
            return;
        }

        if unique_classes.len() == 1 {
            self.weights = vec![vec![0.0; n_cols + self.interactions.len()]];
            self.intercepts = vec![0.0];
            self.backend = "numeric_multiclass_fallback_single_class".to_string();
            return;
        }

        let masked_features: Vec<Vec<f64>> = class_mask
            .iter()
            .filter_map(|&idx| feature_data.get(idx).cloned())
            .collect();
        if masked_features.len() != class_values.len() {
            self.backend = "numeric_multiclass_fallback_mask_mismatch".to_string();
            return;
        }

        let max_epochs = std::env::var("TYPE_LOGISTIC_EPOCHS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(20);
        let mut all_weights = Vec::with_capacity(unique_classes.len());
        let mut intercepts = Vec::with_capacity(unique_classes.len());

        for &class_value in &unique_classes {
            progress(&format!(
                "[type-logistic] train class {} vs rest: rows={}, main_dimensions={}, interaction_terms={}",
                class_value,
                masked_features.len(),
                n_cols,
                self.interactions.len()
            ));
            let target: Vec<f64> = class_values
                .iter()
                .map(|&value| {
                    if value as i64 == class_value {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let n_pos = target.iter().filter(|&&v| v > 0.5).count() as f64;
            let n_neg = target.len() as f64 - n_pos;
            if n_pos < 1.0 || n_neg < 1.0 {
                all_weights.push(vec![0.0; n_cols + self.interactions.len()]);
                intercepts.push(if n_pos > 0.0 { 10.0 } else { -10.0 });
                continue;
            }
            let class_weight = vec![
                target.len() as f64 / (2.0 * n_neg.max(1.0)),
                target.len() as f64 / (2.0 * n_pos.max(1.0)),
            ];
            let initial_weights = vec![0.0; n_cols];
            let prior = (n_pos / target.len() as f64).clamp(1e-6, 1.0 - 1e-6);
            let initial_intercept = (prior / (1.0 - prior)).ln();
            let (main_weights, intercept, interaction_weights) =
                fit_numeric_interaction_logistic_parallel(NumericInteractionBatchFit {
                    feature_data: &masked_features,
                    target: &target,
                    feature_names: &self.numeric_feature_columns[..n_cols],
                    means: &self.numeric_means[..n_cols],
                    scales: &self.numeric_scales[..n_cols],
                    class_weight: &class_weight,
                    initial_weights: &initial_weights,
                    initial_intercept,
                    baseline_weights: &self.baseline_weights[..n_cols],
                    interaction_terms: &self.interactions,
                    activation_gamma: self.activation_gamma,
                    max_epochs,
                    learning_rate: 0.4,
                    l2_penalty: 1.0,
                    progress: Some(progress),
                });
            let mut combined = main_weights;
            combined.extend(interaction_weights);
            all_weights.push(combined);
            intercepts.push(intercept);
        }

        self.weights = all_weights;
        self.intercepts = intercepts;
        self.backend = "numeric_multiclass_logistic_with_interactions".to_string();
    }

    pub fn is_numeric_backend(&self) -> bool {
        self.backend.starts_with("numeric_multiclass")
            && !self.numeric_feature_columns.is_empty()
            && !self.classes.is_empty()
    }

    pub fn predict_numeric_proba(&self, feature_data: &[Vec<f64>]) -> Vec<Vec<(i64, f64)>> {
        if feature_data.is_empty() || self.classes.is_empty() {
            return vec![Vec::new(); feature_data.len()];
        }
        if self.classes.len() == 1 {
            let cls = self.classes[0];
            return vec![vec![(cls, 1.0)]; feature_data.len()];
        }
        let n_cols = self
            .numeric_feature_columns
            .len()
            .min(self.numeric_means.len())
            .min(self.numeric_scales.len())
            .min(self.baseline_weights.len());
        feature_data
            .par_iter()
            .map(|row| {
                let mut standardized = vec![0.0; n_cols];
                for col_idx in 0..n_cols {
                    standardized[col_idx] = scaled_standardize_value(
                        row.get(col_idx).copied().unwrap_or(f64::NAN),
                        self.numeric_means[col_idx],
                        self.numeric_scales[col_idx],
                        self.numeric_feature_columns.get(col_idx).map(String::as_str),
                        self.climate_influence_scale,
                    );
                }
                let interaction_values: Vec<f64> = self
                    .interactions
                    .iter()
                    .map(|term| {
                        interaction_activation_product(
                            &standardized,
                            &self.baseline_weights[..n_cols],
                            &term.feature_indices,
                            self.activation_gamma,
                        )
                    })
                    .collect();
                let mut logits = Vec::with_capacity(self.classes.len());
                for class_idx in 0..self.classes.len() {
                    let mut logit = self.intercepts.get(class_idx).copied().unwrap_or(0.0);
                    if let Some(weights) = self.weights.get(class_idx) {
                        for col_idx in 0..n_cols {
                            logit += weights.get(col_idx).copied().unwrap_or(0.0)
                                * standardized[col_idx];
                        }
                        for (term_idx, &value) in interaction_values.iter().enumerate() {
                            logit += weights.get(n_cols + term_idx).copied().unwrap_or(0.0) * value;
                        }
                    }
                    logits.push(logit);
                }
                let probs = softmax_vec(&logits);
                self.classes.iter().copied().zip(probs).collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn predict_numeric_proba_with_dominant(
        &self,
        feature_data: &[Vec<f64>],
    ) -> (Vec<Vec<(i64, f64)>>, Vec<[i16; 4]>) {
        if feature_data.is_empty() || self.classes.is_empty() {
            return (
                vec![Vec::new(); feature_data.len()],
                vec![[0; 4]; feature_data.len()],
            );
        }
        if self.classes.len() == 1 {
            let cls = self.classes[0];
            let mut dominant = [0i16; 4];
            if (1..=4).contains(&cls) {
                dominant[cls as usize - 1] = 0;
            }
            return (
                vec![vec![(cls, 1.0)]; feature_data.len()],
                vec![dominant; feature_data.len()],
            );
        }
        let n_cols = self
            .numeric_feature_columns
            .len()
            .min(self.numeric_means.len())
            .min(self.numeric_scales.len())
            .min(self.baseline_weights.len());
        let dominant_interaction_limit = std::env::var("DOMINANT_INTERACTION_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(64);
        let pairs: Vec<(Vec<(i64, f64)>, [i16; 4])> = feature_data
            .par_iter()
            .map(|row| {
                let mut standardized = vec![0.0; n_cols];
                for col_idx in 0..n_cols {
                    standardized[col_idx] = scaled_standardize_value(
                        row.get(col_idx).copied().unwrap_or(f64::NAN),
                        self.numeric_means[col_idx],
                        self.numeric_scales[col_idx],
                        self.numeric_feature_columns.get(col_idx).map(String::as_str),
                        self.climate_influence_scale,
                    );
                }
                let interaction_values: Vec<f64> = self
                    .interactions
                    .iter()
                    .map(|term| {
                        interaction_activation_product(
                            &standardized,
                            &self.baseline_weights[..n_cols],
                            &term.feature_indices,
                            self.activation_gamma,
                        )
                    })
                    .collect();

                let mut logits = Vec::with_capacity(self.classes.len());
                let mut dominant_by_type = [0i16; 4];
                for class_idx in 0..self.classes.len() {
                    let mut logit = self.intercepts.get(class_idx).copied().unwrap_or(0.0);
                    let mut best_value = 0.0;
                    let mut best_index = 0i16;
                    let mut best_priority = 0usize;
                    if let Some(weights) = self.weights.get(class_idx) {
                        for col_idx in 0..n_cols {
                            let contribution = weights.get(col_idx).copied().unwrap_or(0.0)
                                * standardized[col_idx];
                            logit += contribution;
                            let is_climate = self
                                .numeric_feature_columns
                                .get(col_idx)
                                .map(|name| is_climate_feature_name(name))
                                .unwrap_or(false);
                            if !is_climate
                                && contribution > 0.0
                                && (best_priority < 1
                                    || (best_priority == 1 && contribution > best_value))
                            {
                                best_value = contribution;
                                best_index = (col_idx + 1) as i16;
                                best_priority = 1;
                            }
                        }
                        for (term_idx, (&value, term)) in interaction_values
                            .iter()
                            .zip(self.interactions.iter())
                            .enumerate()
                        {
                            let contribution =
                                weights.get(n_cols + term_idx).copied().unwrap_or(0.0) * value;
                            logit += contribution;
                            if term_idx < dominant_interaction_limit {
                                let priority = interaction_dominance_priority(term.order);
                                if contribution > 0.0
                                    && priority > 0
                                    && (priority > best_priority
                                        || (priority == best_priority && contribution > best_value))
                                {
                                    best_value = contribution;
                                    best_index = (n_cols + term_idx + 1) as i16;
                                    best_priority = priority;
                                }
                            }
                        }
                    }
                    logits.push(logit);
                    let class_value = self.classes[class_idx];
                    if (1..=4).contains(&class_value) {
                        dominant_by_type[class_value as usize - 1] = best_index;
                    }
                }
                let probs = softmax_vec(&logits);
                (
                    self.classes.iter().copied().zip(probs).collect::<Vec<_>>(),
                    dominant_by_type,
                )
            })
            .collect();
        pairs.into_iter().unzip()
    }

    /// Fit the model.
    ///
    /// `label_df`: discretized label data (all samples)
    /// `training_samples`: full training data with "disturbance_class" column
    /// `class_values`: pre-extracted disturbance_class values (only valid classes)
    /// `class_mask`: indices in label_df where disturbance_class is valid
    /// `factors`: selected inducing factors
    pub fn fit(
        &mut self,
        label_df: &[HashMap<String, String>],
        class_values: &[f64],
        class_mask: &[usize],
        factors: &[InducingFactor],
        random_state: u64,
    ) {
        self.factor_ids = factors.iter().map(|f| f.factor_id.clone()).collect();

        if class_mask.is_empty() || factors.is_empty() {
            self.backend = "fallback_no_data".to_string();
            return;
        }

        // Extract masked label data
        let masked_label_df: Vec<HashMap<String, String>> = class_mask
            .iter()
            .map(|&idx| label_df[idx].clone())
            .collect();

        let x_matrix = build_factor_activation(&masked_label_df, factors);

        // Get unique classes
        let mut unique_classes: Vec<i64> = class_values.iter().map(|&v| v as i64).collect();
        unique_classes.sort();
        unique_classes.dedup();

        // Compute priors
        let n_total = class_values.len() as f64;
        for &cls in &unique_classes {
            let count = class_values.iter().filter(|&&v| (v as i64) == cls).count() as f64;
            self.priors.insert(cls, count / n_total.max(1.0));
        }

        // Single-class fallback
        if unique_classes.len() <= 1 {
            let cls = unique_classes.first().copied().unwrap_or(0);
            self.classes = unique_classes;
            self.priors = HashMap::new();
            self.priors.insert(cls, 1.0);
            self.weights = vec![vec![0.0; factors.len()]];
            self.intercepts = vec![0.0];
            self.backend = "multinomial_logistic_fallback_single_class".to_string();
            return;
        }

        // Compute balanced class weights
        let n_samples = class_values.len();
        let class_weight: Vec<f64> = unique_classes
            .iter()
            .map(|&cls| {
                let count = class_values.iter().filter(|&&v| (v as i64) == cls).count();
                n_samples as f64 / (unique_classes.len() as f64 * count.max(1) as f64)
            })
            .collect();

        let _seed = random_state;

        let (weights, intercepts) = fit_logistic_multiclass(
            &x_matrix,
            class_values,
            &unique_classes,
            Some(&class_weight),
            100,
            1.0,
        );

        self.weights = weights;
        self.intercepts = intercepts;
        self.classes = unique_classes;
        self.backend = "multinomial_logistic".to_string();
    }

    /// Predict class probabilities for each sample.
    /// Returns Vec of (class_value, probability) pairs per sample.
    pub fn predict_proba(
        &self,
        label_df: &[HashMap<String, String>],
        factors: &[InducingFactor],
    ) -> Vec<Vec<(i64, f64)>> {
        let n = label_df.len();
        if n == 0 || self.classes.is_empty() {
            return vec![Vec::new(); n];
        }

        // Single-class fallback
        if self.classes.len() <= 1 {
            let cls = self.classes.first().copied().unwrap_or(0);
            return vec![vec![(cls, 1.0)]; n];
        }

        let x_matrix = build_factor_activation(label_df, factors);
        let probas = predict_softmax_proba(&x_matrix, &self.weights, &self.intercepts);

        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let mut row_probs: Vec<(i64, f64)> = Vec::with_capacity(self.classes.len());
            for (ci, &cls) in self.classes.iter().enumerate() {
                row_probs.push((cls, probas[[i, ci]]));
            }
            results.push(row_probs);
        }
        results
    }

    /// Predict class probabilities for a single sample.
    pub fn predict_single(
        &self,
        row: &HashMap<String, String>,
        factors: &[InducingFactor],
    ) -> Vec<(i64, f64)> {
        let activation = factor_activation_for_row(row, factors);
        let n_classes = self.classes.len();

        if n_classes == 0 {
            return Vec::new();
        }

        if n_classes <= 1 {
            let cls = self.classes.first().copied().unwrap_or(0);
            return vec![(cls, 1.0)];
        }

        // Compute logits
        let mut logits = vec![0.0; n_classes];
        #[allow(clippy::needless_range_loop)]
        for ci in 0..n_classes {
            let mut lin = self.intercepts.get(ci).copied().unwrap_or(0.0);
            for (j, &val) in activation.iter().enumerate() {
                if ci < self.weights.len() && j < self.weights[ci].len() {
                    lin += self.weights[ci][j] * val;
                }
            }
            logits[ci] = lin;
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let mut result = Vec::with_capacity(n_classes);
        for (ci, &cls) in self.classes.iter().enumerate() {
            let prob = (logits[ci] - max_logit).exp() / exp_sum.max(1e-10);
            result.push((cls, prob));
        }
        result
    }
}

impl Default for DisturbanceClassEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_label_row(vals: &[(&str, &str)]) -> HashMap<String, String> {
        let mut row = HashMap::new();
        for (k, v) in vals {
            row.insert(k.to_string(), v.to_string());
        }
        row
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 10.0, 0.0, 0.0])
                .unwrap();
        let probs = softmax(&logits);
        for r in 0..3 {
            let sum: f64 = probs.row(r).sum();
            assert!((sum - 1.0).abs() < 1e-10, "Row {} sums to {}", r, sum);
        }
    }

    #[test]
    fn test_factor_activation_empty() {
        let factors: Vec<InducingFactor> = vec![];
        let label_df: Vec<HashMap<String, String>> = vec![make_label_row(&[("a", "1")])];
        let x_matrix = build_factor_activation(&label_df, &factors);
        assert_eq!(x_matrix.dim(), (1, 0));
    }

    #[test]
    fn test_factor_activation_basic() {
        let factor = InducingFactor {
            factor_id: "F1".to_string(),
            name: "test".to_string(),
            semantics: "test".to_string(),
            items: vec!["a=high".to_string()],
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 1.0,
            score: 1.0,
            count_event: 50,
            count_background: 10,
            n_event: 100,
            n_background: 100,
            rr_ci_lower: 2.0,
            rr_ci_upper: 10.0,
            p_value: 0.01,
            fdr_significant: true,
            likelihood_ratio: 5.0,
            true_relative_risk: 5.0,
        };

        let label_df = vec![
            make_label_row(&[("a", "high")]),
            make_label_row(&[("a", "low")]),
            make_label_row(&[("a", "high")]),
        ];
        let x_matrix = build_factor_activation(&label_df, &[factor]);
        assert_eq!(x_matrix[[0, 0]], 1.0);
        assert_eq!(x_matrix[[1, 0]], 0.0);
        assert_eq!(x_matrix[[2, 0]], 1.0);
    }

    #[test]
    fn test_multifactor_activation() {
        let f1 = InducingFactor {
            factor_id: "F1".to_string(),
            name: "f1".to_string(),
            semantics: "f1".to_string(),
            items: vec!["a=high".to_string(), "b=near".to_string()],
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 1.0,
            score: 1.0,
            count_event: 50,
            count_background: 10,
            n_event: 100,
            n_background: 100,
            rr_ci_lower: 2.0,
            rr_ci_upper: 10.0,
            p_value: 0.01,
            fdr_significant: true,
            likelihood_ratio: 5.0,
            true_relative_risk: 5.0,
        };

        let label_df = vec![
            make_label_row(&[("a", "high"), ("b", "near")]),
            make_label_row(&[("a", "high"), ("b", "far")]),
            make_label_row(&[("a", "low"), ("b", "near")]),
        ];
        let x_matrix = build_factor_activation(&label_df, &[f1]);
        assert_eq!(x_matrix[[0, 0]], 1.0, "Both conditions met");
        assert_eq!(x_matrix[[1, 0]], 0.0, "b is far");
        assert_eq!(x_matrix[[2, 0]], 0.0, "a is low");
    }

    #[test]
    fn test_logistic_binary_perfect_separation() {
        // Simple separable data
        let x_matrix = Array2::from_shape_vec((4, 1), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let (w, _intercept) = fit_logistic_binary_internal(&x_matrix, &y, None, 50, 1.0);
        assert!(!w.is_empty());
        // Just verify no NaN
        assert!(w[0].is_finite());
    }

    #[test]
    fn test_occurrence_engine_predict() {
        let factor = InducingFactor {
            factor_id: "F1".to_string(),
            name: "test".to_string(),
            semantics: "test".to_string(),
            items: vec!["a=high".to_string()],
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 1.0,
            score: 1.0,
            count_event: 5,
            count_background: 1,
            n_event: 10,
            n_background: 10,
            rr_ci_lower: 2.0,
            rr_ci_upper: 10.0,
            p_value: 0.01,
            fdr_significant: true,
            likelihood_ratio: 5.0,
            true_relative_risk: 5.0,
        };

        let label_df = vec![
            make_label_row(&[("a", "high")]),
            make_label_row(&[("a", "high")]),
            make_label_row(&[("a", "low")]),
            make_label_row(&[("a", "low")]),
            make_label_row(&[("a", "low")]),
            make_label_row(&[("a", "low")]),
        ];
        let target = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let mut engine = OccurrenceProbabilityEngine::new();
        engine.fit(&label_df, &target, std::slice::from_ref(&factor), 42);

        assert_eq!(engine.backend, "logistic_regression");
        assert_eq!(engine.factor_ids, vec!["F1"]);

        let probs = engine.predict_proba(&label_df, &[factor]);
        assert_eq!(probs.len(), 6);
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p), "Probability {} out of range", p);
        }
        // High activation should give higher probability
        assert!(
            probs[0] > probs[2],
            "a=high should have higher prob than a=low"
        );
    }

    #[test]
    fn test_numeric_interaction_prediction_and_dominant() {
        let mut engine = OccurrenceProbabilityEngine::new();
        engine.backend = "numeric_logistic_regression_with_interactions".to_string();
        engine.numeric_feature_columns = vec!["rainfall_avg_prev_3m".into(), "dist_to_road".into()];
        engine.numeric_means = vec![0.0, 0.0];
        engine.numeric_scales = vec![1.0, 1.0];
        engine.weights = vec![0.1, 0.1];
        engine.baseline_weights = vec![1.0, -1.0];
        engine.intercept = -2.0;
        engine.activation_gamma = 1.5;
        engine.interactions = vec![NumericInteractionTerm {
            id: "rainfall_avg_prev_3m*dist_to_road".into(),
            order: 2,
            feature_indices: vec![0, 1],
            feature_names: vec!["rainfall_avg_prev_3m".into(), "dist_to_road".into()],
            weight: 5.0,
            support_pos: 0.2,
            support_neg: 0.05,
            lift: 4.0,
            log_lift: 4.0f64.ln(),
            stability: 1.0,
            score: 0.2 * 4.0f64.ln(),
            risk_thresholds: Vec::new(),
            activation_midpoints: Vec::new(),
            learned_thresholds: Vec::new(),
            explanation_text: String::new(),
        }];

        let rows = vec![vec![2.0, -2.0], vec![-2.0, 2.0]];
        let (probabilities, dominant) = engine.predict_numeric_rows_with_dominant(&rows);
        assert!(probabilities[0] > probabilities[1]);
        assert_eq!(dominant[0], 3);
    }

    #[test]
    fn test_occurrence_single_class_no_panic() {
        let factor = InducingFactor {
            factor_id: "F1".to_string(),
            name: "test".to_string(),
            semantics: "test".to_string(),
            items: vec!["a=high".to_string()],
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 1.0,
            score: 1.0,
            count_event: 10,
            count_background: 0,
            n_event: 10,
            n_background: 10,
            rr_ci_lower: 2.0,
            rr_ci_upper: 10.0,
            p_value: 0.01,
            fdr_significant: true,
            likelihood_ratio: 5.0,
            true_relative_risk: 5.0,
        };

        let label_df = vec![
            make_label_row(&[("a", "high")]),
            make_label_row(&[("a", "high")]),
        ];
        // All positive class
        let target = vec![1.0, 1.0];

        let mut engine = OccurrenceProbabilityEngine::new();
        engine.fit(&label_df, &target, std::slice::from_ref(&factor), 42);

        // Should use fallback, not panic
        assert!(engine.backend.contains("fallback"));

        let probs = engine.predict_proba(&label_df, &[factor]);
        assert_eq!(probs.len(), 2);
        for &p in &probs {
            (0.0..=1.0).contains(&p);
        }
    }

    #[test]
    fn test_class_engine_predict() {
        let factor = InducingFactor {
            factor_id: "F1".to_string(),
            name: "test".to_string(),
            semantics: "test".to_string(),
            items: vec!["a=high".to_string()],
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 1.0,
            score: 1.0,
            count_event: 5,
            count_background: 1,
            n_event: 10,
            n_background: 10,
            rr_ci_lower: 2.0,
            rr_ci_upper: 10.0,
            p_value: 0.01,
            fdr_significant: true,
            likelihood_ratio: 5.0,
            true_relative_risk: 5.0,
        };

        // 6 samples, all valid classes
        let label_df: Vec<HashMap<String, String>> = (0..6)
            .map(|i| {
                if i < 3 {
                    make_label_row(&[("a", "high")])
                } else {
                    make_label_row(&[("a", "low")])
                }
            })
            .collect();

        // Class values: 3 samples of class 1, 3 samples of class 3
        let class_values = vec![1.0, 1.0, 1.0, 3.0, 3.0, 3.0];
        let class_mask: Vec<usize> = (0..6).collect();

        let mut engine = DisturbanceClassEngine::new();
        engine.fit(
            &label_df,
            &class_values,
            &class_mask,
            std::slice::from_ref(&factor),
            42,
        );

        assert_eq!(engine.backend, "multinomial_logistic");
        assert!(!engine.classes.is_empty());

        let probs = engine.predict_proba(&label_df, &[factor]);
        assert_eq!(probs.len(), 6);
        for row in &probs {
            let sum: f64 = row.iter().map(|(_, p)| p).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Probs should sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_class_single_class_no_panic() {
        let factor = InducingFactor {
            factor_id: "F1".to_string(),
            name: "test".to_string(),
            semantics: "test".to_string(),
            items: vec!["a=high".to_string()],
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 1.0,
            score: 1.0,
            count_event: 10,
            count_background: 0,
            n_event: 10,
            n_background: 10,
            rr_ci_lower: 2.0,
            rr_ci_upper: 10.0,
            p_value: 0.01,
            fdr_significant: true,
            likelihood_ratio: 5.0,
            true_relative_risk: 5.0,
        };

        let label_df = vec![
            make_label_row(&[("a", "high")]),
            make_label_row(&[("a", "high")]),
        ];
        let class_values = vec![1.0, 1.0]; // All same class
        let class_mask: Vec<usize> = (0..2).collect();

        let mut engine = DisturbanceClassEngine::new();
        engine.fit(
            &label_df,
            &class_values,
            &class_mask,
            std::slice::from_ref(&factor),
            42,
        );

        // Should use fallback
        assert!(engine.backend.contains("fallback"));
        assert_eq!(engine.classes.len(), 1);

        let probs = engine.predict_proba(&label_df, &[factor]);
        assert_eq!(probs.len(), 2);
        for row in &probs {
            let sum: f64 = row.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_occurrence_low_sample() {
        // Very few training rows
        let factor = InducingFactor {
            factor_id: "F1".to_string(),
            name: "test".to_string(),
            semantics: "test".to_string(),
            items: vec!["a=high".to_string()],
            support_event: 0.5,
            support_background: 0.5,
            relative_risk: 1.0,
            stability: 1.0,
            score: 0.0,
            count_event: 1,
            count_background: 1,
            n_event: 2,
            n_background: 2,
            rr_ci_lower: 0.0,
            rr_ci_upper: 0.0,
            p_value: 1.0,
            fdr_significant: false,
            likelihood_ratio: 1.0,
            true_relative_risk: 1.0,
        };

        let label_df = vec![
            make_label_row(&[("a", "high")]),
            make_label_row(&[("a", "low")]),
        ];
        let target = vec![1.0, 0.0];

        let mut engine = OccurrenceProbabilityEngine::new();
        engine.fit(&label_df, &target, std::slice::from_ref(&factor), 42);
        // Should not panic even with very few rows
        let probs = engine.predict_proba(&label_df, &[factor]);
        assert_eq!(probs.len(), 2);
        for &p in &probs {
            (0.0..=1.0).contains(&p);
        }
    }
}

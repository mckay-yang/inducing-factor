use serde::{Deserialize, Serialize};

/// Configuration for the AdaptiveDiscretizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscretizerConfig {
    /// Discretization method: "supervised_tree", "mdlp", "quantile", "uniform"
    #[serde(default = "default_method")]
    pub method: String,
    /// Number of bins for continuous features (default 3)
    #[serde(default = "default_n_bins")]
    pub n_bins: usize,
    /// Minimum fraction of total samples per leaf (0.05 = 5%)
    #[serde(default = "default_min_samples_leaf")]
    pub min_samples_leaf: f64,
    /// Features with ≤ this many unique values are treated as categorical
    #[serde(default = "default_categorical_unique_threshold")]
    pub categorical_unique_threshold: usize,
}

fn default_method() -> String {
    "supervised_tree".to_string()
}
fn default_n_bins() -> usize {
    3
}
fn default_min_samples_leaf() -> f64 {
    0.05
}
fn default_categorical_unique_threshold() -> usize {
    12
}

impl Default for DiscretizerConfig {
    fn default() -> Self {
        Self {
            method: "supervised_tree".to_string(),
            n_bins: 3,
            min_samples_leaf: 0.05,
            categorical_unique_threshold: 12,
        }
    }
}

/// Metadata for a single feature after discretizer fitting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureMeta {
    /// "continuous" or "categorical"
    pub kind: String,
    /// Bin boundaries: [-inf, t1, t2, ..., inf] (continuous only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bins: Option<Vec<f64>>,
    /// Human-readable labels for each bin (continuous only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Sorted unique category strings (categorical only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categories: Option<Vec<String>>,
}

/// Configuration for the FactorMiner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorMiningConfig {
    /// Maximum number of items in a factor combination
    #[serde(default = "default_max_pattern_size")]
    pub max_pattern_size: usize,
    /// Minimum event support for a pattern
    #[serde(default = "default_min_support")]
    pub min_support: f64,
    /// Minimum relative risk (event / background)
    #[serde(default = "default_min_relative_risk")]
    pub min_relative_risk: f64,
    /// Minimum bootstrap stability
    #[serde(default = "default_min_stability")]
    pub min_stability: f64,
    /// Number of bootstrap iterations
    #[serde(default = "default_bootstrap_iterations")]
    pub bootstrap_iterations: usize,
    /// Maximum number of samples per bootstrap draw
    #[serde(default = "default_max_bootstrap_samples")]
    pub max_bootstrap_samples: usize,
    /// Random seed for reproducibility
    #[serde(default = "default_random_state")]
    pub random_state: u64,
    /// Scoring method: "log_rr" or "linear"
    #[serde(default = "default_score_method")]
    pub score_method: String,
}

fn default_max_pattern_size() -> usize {
    3
}
fn default_min_support() -> f64 {
    0.08
}
fn default_min_relative_risk() -> f64 {
    1.2
}
fn default_min_stability() -> f64 {
    0.4
}
fn default_bootstrap_iterations() -> usize {
    5
}
fn default_max_bootstrap_samples() -> usize {
    1000
}
fn default_random_state() -> u64 {
    42
}
fn default_score_method() -> String {
    "log_rr".to_string()
}

impl Default for FactorMiningConfig {
    fn default() -> Self {
        Self {
            max_pattern_size: 3,
            min_support: 0.08,
            min_relative_risk: 1.2,
            min_stability: 0.4,
            bootstrap_iterations: 5,
            max_bootstrap_samples: 1000,
            random_state: 42,
            score_method: "log_rr".to_string(),
        }
    }
}

/// An inducing factor — a pattern of feature-value combinations that co-occurs
/// with ecological disturbances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InducingFactor {
    pub factor_id: String,
    /// Machine-readable name (e.g., "rainfall_prev_2m_high")
    pub name: String,
    /// Human-readable semantics (e.g., "high rainfall_prev_2m")
    pub semantics: String,
    /// Sorted list of "feature=value" items
    pub items: Vec<String>,
    /// Support in event (disturbance) population
    pub support_event: f64,
    /// Support in background population
    pub support_background: f64,
    /// Relative risk: support_event / support_background (with smoothing)
    pub relative_risk: f64,
    /// Bootstrap stability (0-1)
    pub stability: f64,
    /// Composite score for ranking
    pub score: f64,
    /// Raw count in event population
    pub count_event: i64,
    /// Raw count in background population
    pub count_background: i64,
    /// Total event samples
    pub n_event: i64,
    /// Total background samples
    pub n_background: i64,
    /// Lower 95% CI bound for relative risk
    pub rr_ci_lower: f64,
    /// Upper 95% CI bound for relative risk
    pub rr_ci_upper: f64,
    /// Fisher exact test p-value
    pub p_value: f64,
    /// Whether FDR-significant after Benjamini-Hochberg correction
    pub fdr_significant: bool,
    /// Likelihood ratio (same as relative_risk for now)
    pub likelihood_ratio: f64,
    /// True relative risk: P(event|pattern) / P(event|¬pattern)
    pub true_relative_risk: f64,
}

/// A library of inducing factors, sorted by score descending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorLibrary {
    pub factors: Vec<InducingFactor>,
}

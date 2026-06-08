//! Typed artifact containers for trained model outputs, factor libraries,
//! discretization metadata, and pipeline cache manifests.
//!
//! Every top-level artifact type carries a version, schema_name, source_commit,
//! and generated_at so that serialized archives are self-describing and
//! machine-verifiable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Meta envelope ──────────────────────────────────────────────────────────

/// Mandatory metadata attached to every serialized artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactMeta {
    /// Schema version, e.g. "1.0.0".
    pub version: String,
    /// Human-readable schema identifier, e.g. "inducing_factor.factor_library.v1".
    pub schema_name: String,
    /// Git commit that produced this artifact (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_commit: Option<String>,
    /// ISO-8601 timestamp of generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_at: Option<String>,
}

impl ArtifactMeta {
    pub fn new(version: &str, schema_name: &str) -> Self {
        Self {
            version: version.to_string(),
            schema_name: schema_name.to_string(),
            source_commit: None,
            generated_at: None,
        }
    }
}

// ── Factor library ─────────────────────────────────────────────────────────

/// A single inducing factor with its statistics and metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactorEntry {
    pub factor_id: String,
    pub name: String,
    pub semantics: String,
    pub items: Vec<String>,
    pub support_event: f64,
    pub support_background: f64,
    pub relative_risk: f64,
    pub stability: f64,
    pub score: f64,
    pub count_event: u64,
    pub count_background: u64,
    pub n_event: u64,
    pub n_background: u64,
    pub rr_ci_lower: f64,
    pub rr_ci_upper: f64,
    pub p_value: f64,
    pub fdr_significant: bool,
}

/// Full inducing factor library — the raw payload matches the golden-fixture
/// JSON array format.
pub type FactorLibraryRaw = Vec<FactorEntry>;

/// Versioned artifact envelope for a factor library.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactorLibrary {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    pub factors: Vec<FactorEntry>,
}

impl FactorLibrary {
    pub fn from_raw(meta: ArtifactMeta, raw: FactorLibraryRaw) -> Self {
        Self { meta, factors: raw }
    }
}

// ── Model metadata ─────────────────────────────────────────────────────────

/// Metadata produced after training the attribution model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadataRaw {
    pub feature_columns: Vec<String>,
    pub network_feature_columns: Vec<String>,
    pub selected_factor_ids: Vec<String>,
    pub class_labels: HashMap<String, String>,
    pub target: String,
    pub streaming_run: bool,
    pub discretizer_sample_limit: u64,
    pub model_sample_limit: u64,
}

/// Versioned artifact envelope for model metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    #[serde(flatten)]
    pub inner: ModelMetadataRaw,
}

impl ModelMetadata {
    pub fn from_raw(meta: ArtifactMeta, raw: ModelMetadataRaw) -> Self {
        Self { meta, inner: raw }
    }
}

// ── Bayesian network edges ─────────────────────────────────────────────────

/// A single directed edge in the Bayesian network.
pub type BayesianEdge = (String, String);

/// Raw edge list (matches golden JSON format).
pub type BayesianNetworkEdgesRaw = Vec<BayesianEdge>;

/// Versioned artifact envelope for Bayesian network edges.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BayesianNetworkEdges {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    pub edges: Vec<BayesianEdge>,
}

impl BayesianNetworkEdges {
    pub fn from_raw(meta: ArtifactMeta, raw: BayesianNetworkEdgesRaw) -> Self {
        Self { meta, edges: raw }
    }
}

// ── Discretization labels ──────────────────────────────────────────────────

mod serde_f64_vec {
    use serde::{de, ser, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bins: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        use ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(bins.len()))?;
        for &v in bins {
            seq.serialize_element(&display_f64(v))?;
        }
        seq.end()
    }

    pub fn serialize_opt<S: Serializer>(
        bins: &Option<Vec<f64>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match bins {
            Some(v) => serialize(v, serializer),
            None => serializer.serialize_none(),
        }
    }

    fn display_f64(v: f64) -> serde_json::Value {
        if v.is_infinite() {
            if v.is_sign_positive() {
                serde_json::Value::String("Infinity".into())
            } else {
                serde_json::Value::String("-Infinity".into())
            }
        } else if v.is_nan() {
            serde_json::Value::String("NaN".into())
        } else {
            serde_json::Number::from_f64(v)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
    }

    struct F64VecVisitor;
    impl<'de> de::Visitor<'de> for F64VecVisitor {
        type Value = Vec<f64>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("an array of floats or 'Infinity'/'NaN' strings")
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<f64>, A::Error> {
            let mut out = Vec::new();
            while let Some(elem) = seq.next_element::<serde_json::Value>()? {
                out.push(value_to_f64(elem));
            }
            Ok(out)
        }
    }

    fn value_to_f64(v: serde_json::Value) -> f64 {
        match v {
            serde_json::Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
            serde_json::Value::String(s) => match s.as_str() {
                "Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                "NaN" => f64::NAN,
                _ => f64::NAN,
            },
            serde_json::Value::Null => f64::NAN,
            _ => f64::NAN,
        }
    }

    #[allow(dead_code)]
    pub fn deserialize_opt<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<f64>>, D::Error> {
        Ok(Some(d.deserialize_seq(F64VecVisitor)?))
    }
}

/// Metadata for a single discretized feature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureDiscretization {
    pub kind: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serde_f64_vec::serialize_opt",
        deserialize_with = "serde_f64_vec::deserialize_opt"
    )]
    pub bins: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub categories: Option<Vec<String>>,
}

/// Raw discretization labels map (matches golden JSON format).
pub type DiscretizationLabelsRaw = HashMap<String, FeatureDiscretization>;

/// Versioned artifact envelope for discretization labels.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscretizationLabels {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    /// Map of feature name -> discretization info.
    pub features: HashMap<String, FeatureDiscretization>,
}

impl DiscretizationLabels {
    pub fn from_raw(meta: ArtifactMeta, raw: DiscretizationLabelsRaw) -> Self {
        Self {
            meta,
            features: raw,
        }
    }
}

// ── Cache manifest ─────────────────────────────────────────────────────────

/// A single chunk entry in the cache manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkEntry {
    /// Chunk identifier, typically "chunk_<row_start>_<row_end>".
    pub chunk_id: String,
    /// SHA-256 hex digest of the chunk file.
    pub file_hash: String,
    /// ISO-8601 timestamp of chunk generation.
    pub generated_at: String,
}

/// Manifest tracking all cached chunks on disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheManifest {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    pub chunks: Vec<ChunkEntry>,
}

impl CacheManifest {
    pub fn new(meta: ArtifactMeta) -> Self {
        Self {
            meta,
            chunks: Vec::new(),
        }
    }
}

// ── Attribution preview ────────────────────────────────────────────────────

/// A single row in the attribution preview — one sample with top-ranked factors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct AttributionPreviewRow {
    pub sample_id: u64,
    pub top_factor_id: String,
    pub top_factor_name: String,
    pub top_probability: f64,
    pub predicted_class: Option<String>,
}

/// Raw attribution preview (matches golden parquet schema).
pub type AttributionPreviewRaw = Vec<AttributionPreviewRow>;

/// Versioned artifact envelope for an attribution preview.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttributionPreview {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    pub rows: Vec<AttributionPreviewRow>,
}

impl AttributionPreview {
    pub fn from_raw(meta: ArtifactMeta, raw: AttributionPreviewRaw) -> Self {
        Self { meta, rows: raw }
    }
}

// ── Factor class summary ───────────────────────────────────────────────────

/// One row of the factor-class association summary (chi-square, frequencies).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct FactorClassSummaryRow {
    pub factor_id: String,
    pub class_label: String,
    pub chi2: f64,
    pub p_value: f64,
    pub freq_event: f64,
    pub freq_background: f64,
}

/// Raw factor-class summary (matches golden parquet schema).
pub type FactorClassSummaryRaw = Vec<FactorClassSummaryRow>;

/// Versioned artifact envelope for factor-class summaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactorClassSummary {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    pub rows: Vec<FactorClassSummaryRow>,
}

impl FactorClassSummary {
    pub fn from_raw(meta: ArtifactMeta, raw: FactorClassSummaryRaw) -> Self {
        Self { meta, rows: raw }
    }
}

// ── Model predictions ──────────────────────────────────────────────────────

/// A single prediction row with occurrence probability and class prediction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct ModelPredictionRow {
    pub point_id: i64,
    pub x: f64,
    pub y: f64,
    pub year: i64,
    pub disturbance_probability: f64,
    pub class_probability_1: f64,
    pub class_probability_3: f64,
    pub class_probability_4: f64,
    pub dominant_disturbance_class: i64,
    pub prob_fire_abs: f64,
    pub prob_geo_hazard_abs: f64,
    pub prob_road_abs: f64,
}

/// Raw model predictions (matches golden parquet schema).
pub type ModelPredictionsRaw = Vec<ModelPredictionRow>;

/// Versioned artifact envelope for model predictions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPredictions {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    pub predictions: Vec<ModelPredictionRow>,
}

impl ModelPredictions {
    pub fn from_raw(meta: ArtifactMeta, raw: ModelPredictionsRaw) -> Self {
        Self {
            meta,
            predictions: raw,
        }
    }
}

// ── Selected factors ───────────────────────────────────────────────────────

/// Abbreviated factor info returned by the selection stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectedFactor {
    pub factor_id: String,
    pub name: String,
    pub semantics: String,
    pub items: Vec<String>,
    pub relative_risk: f64,
    pub stability: f64,
    pub score: f64,
}

/// Raw selected factors payload (matches golden JSON format).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectedFactorsRaw {
    pub factors: Vec<SelectedFactor>,
    pub count: u64,
}

/// Versioned artifact envelope for selected factors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectedFactors {
    #[serde(flatten)]
    pub meta: ArtifactMeta,
    pub factors: Vec<SelectedFactor>,
    pub count: u64,
}

impl SelectedFactors {
    pub fn from_raw(meta: ArtifactMeta, raw: SelectedFactorsRaw) -> Self {
        Self {
            meta,
            factors: raw.factors,
            count: raw.count,
        }
    }
}

// ── Manifest (pipeline generation manifest) ────────────────────────────────

/// Tolerances used for golden comparison, deserialized from MANIFEST.json.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestTolerances {
    pub discretization_labels: String,
    pub statistical_metrics: Option<StatisticalTolerances>,
    pub factor_rankings: Option<FactorRankingTolerances>,
    pub occurrence_probabilities: Option<OccurrenceTolerances>,
    pub class_predictions: Option<ClassPredictionTolerances>,
    pub raster_metadata: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatisticalTolerances {
    pub absolute: f64,
    pub relative: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactorRankingTolerances {
    pub spearman_min: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OccurrenceTolerances {
    pub mae_max: f64,
    pub max_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClassPredictionTolerances {
    pub top1_match_min: f64,
}

/// File listing within the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestFiles {
    #[serde(flatten)]
    pub entries: HashMap<String, String>,
}

/// Pipeline parameters recorded in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineParameters {
    pub lookback_years: u64,
    pub background_sample_rate: f64,
    pub max_pattern_size: u64,
    pub min_support: f64,
    pub min_relative_risk: f64,
    pub bootstrap_iterations: u64,
    pub min_stability: f64,
    pub max_bootstrap_samples: u64,
    pub max_network_factors: u64,
    pub discretizer_method: String,
    pub discretize_bins: u64,
    pub random_state: u64,
}

/// The pipeline generation manifest (MANIFEST.json).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub source_commit: String,
    pub generation_command: String,
    pub generated_at: String,
    pub random_seed: u64,
    pub description: String,
    pub tolerances: ManifestTolerances,
    pub pipeline_parameters: PipelineParameters,
    pub files: HashMap<String, String>,
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn roundtrip_factor_library() {
        let meta = ArtifactMeta::new("1.0.0", "inducing_factor.factor_library.v1");
        let factors = vec![FactorEntry {
            factor_id: "IF_001".into(),
            name: "rainfall_prev_2m_high".into(),
            semantics: "high rainfall_prev_2m".into(),
            items: vec!["rainfall_prev_2m=high".into()],
            support_event: 0.5,
            support_background: 0.1,
            relative_risk: 5.0,
            stability: 0.95,
            score: 3.2,
            count_event: 84,
            count_background: 5,
            n_event: 168,
            n_background: 48,
            rr_ci_lower: 2.0,
            rr_ci_upper: 12.0,
            p_value: 0.001,
            fdr_significant: true,
        }];
        let lib = FactorLibrary { meta, factors };
        let json = serde_json::to_string(&lib).unwrap();
        let back: FactorLibrary = serde_json::from_str(&json).unwrap();
        assert_eq!(lib, back);
    }

    #[test]
    fn roundtrip_model_metadata() {
        let meta = ArtifactMeta::new("1.0.0", "inducing_factor.model_metadata.v1");
        let raw = ModelMetadataRaw {
            feature_columns: vec!["slope".into(), "aspect".into()],
            network_feature_columns: vec!["slope".into()],
            selected_factor_ids: vec!["IF_001".into()],
            class_labels: HashMap::from([("fire".into(), "fire".into())]),
            target: "disturbance_flag".into(),
            streaming_run: true,
            discretizer_sample_limit: 200_000,
            model_sample_limit: 200_000,
        };
        let md = ModelMetadata::from_raw(meta, raw);
        let json = serde_json::to_string(&md).unwrap();
        let back: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(md, back);
    }

    #[test]
    fn roundtrip_discretization_labels() {
        let meta = ArtifactMeta::new("1.0.0", "inducing_factor.discretization_labels.v1");
        let mut raw = HashMap::new();
        raw.insert(
            "slope".into(),
            FeatureDiscretization {
                kind: "continuous".into(),
                bins: Some(vec![f64::NEG_INFINITY, 7.83, 22.45, f64::INFINITY]),
                labels: Some(vec!["low".into(), "medium".into(), "high".into()]),
                categories: None,
            },
        );
        let dl = DiscretizationLabels::from_raw(meta, raw);
        let json = serde_json::to_string(&dl).unwrap();
        let back: DiscretizationLabels = serde_json::from_str(&json).unwrap();
        assert_eq!(dl, back);
    }

    #[test]
    fn roundtrip_bayesian_network_edges() {
        let meta = ArtifactMeta::new("1.0.0", "inducing_factor.bayesian_network_edges.v1");
        let raw: BayesianNetworkEdgesRaw = vec![
            ("rainfall_prev_2m".into(), "IF_001".into()),
            ("IF_001".into(), "disturbance_flag".into()),
        ];
        let bn = BayesianNetworkEdges::from_raw(meta, raw);
        let json = serde_json::to_string(&bn).unwrap();
        let back: BayesianNetworkEdges = serde_json::from_str(&json).unwrap();
        assert_eq!(bn, back);
    }

    #[test]
    fn roundtrip_cache_manifest() {
        let meta = ArtifactMeta::new("1.0.0", "inducing_factor.cache_manifest.v1");
        let mut manifest = CacheManifest::new(meta);
        manifest.chunks.push(ChunkEntry {
            chunk_id: "chunk_0_560".into(),
            file_hash: "abc123".into(),
            generated_at: "2026-01-01T00:00:00Z".into(),
        });
        let json = serde_json::to_string(&manifest).unwrap();
        let back: CacheManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest, back);
    }

    #[test]
    fn roundtrip_selected_factors() {
        let meta = ArtifactMeta::new("1.0.0", "inducing_factor.selected_factors.v1");
        let sf = SelectedFactors {
            meta,
            factors: vec![SelectedFactor {
                factor_id: "IF_001".into(),
                name: "rainfall_prev_2m_high".into(),
                semantics: "high rainfall_prev_2m".into(),
                items: vec!["rainfall_prev_2m=high".into()],
                relative_risk: 500.0,
                stability: 1.0,
                score: 5.0,
            }],
            count: 1,
        };
        let json = serde_json::to_string(&sf).unwrap();
        let back: SelectedFactors = serde_json::from_str(&json).unwrap();
        assert_eq!(sf, back);
    }

    #[test]
    fn roundtrip_manifest() {
        let manifest = Manifest {
            source_commit: "abc123".into(),
            generation_command: "run".into(),
            generated_at: "2026-01-01T00:00:00Z".into(),
            random_seed: 42,
            description: "test".into(),
            tolerances: ManifestTolerances {
                discretization_labels: "exact".into(),
                statistical_metrics: Some(StatisticalTolerances {
                    absolute: 1e-6,
                    relative: 1e-4,
                }),
                factor_rankings: Some(FactorRankingTolerances { spearman_min: 0.95 }),
                occurrence_probabilities: Some(OccurrenceTolerances {
                    mae_max: 0.03,
                    max_error: 0.1,
                }),
                class_predictions: Some(ClassPredictionTolerances {
                    top1_match_min: 0.95,
                }),
                raster_metadata: "exact".into(),
            },
            pipeline_parameters: PipelineParameters {
                lookback_years: 2,
                background_sample_rate: 0.3,
                max_pattern_size: 3,
                min_support: 0.08,
                min_relative_risk: 1.2,
                bootstrap_iterations: 5,
                min_stability: 0.4,
                max_bootstrap_samples: 1000,
                max_network_factors: 6,
                discretizer_method: "supervised_tree".into(),
                discretize_bins: 3,
                random_state: 42,
            },
            files: HashMap::new(),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest, back);
    }

    #[test]
    fn artifact_has_required_metadata() {
        let meta = ArtifactMeta::new("1.0.0", "inducing_factor.factor_library.v1");
        assert_eq!(meta.version, "1.0.0");
        assert_eq!(meta.schema_name, "inducing_factor.factor_library.v1");
        // The Serialize output must contain version and schema_name at the
        // top level because #[serde(flatten)] puts them inline with factors.
        let lib = FactorLibrary {
            meta,
            factors: vec![],
        };
        let json = serde_json::to_string(&lib).unwrap();
        assert!(json.contains(r#""version":"1.0.0""#));
        assert!(json.contains(r#""schema_name":"inducing_factor.factor_library.v1""#));
    }
}

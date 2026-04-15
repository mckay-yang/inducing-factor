import glob
import json
import logging
import os
import pickle
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from inducing_factor.config.models import AppConfig
from inducing_factor.core.bayesian_model import BayesianAttributionEngine
from inducing_factor.core.class_model import DisturbanceClassEngine
from inducing_factor.core.discretization import AdaptiveDiscretizer
from inducing_factor.core.factor_mining import FactorMiner
from inducing_factor.core.models import InducingFactor
from inducing_factor.core.occurrence_model import OccurrenceProbabilityEngine
from inducing_factor.io.artifacts import ArtifactWriter
from inducing_factor.pipeline.feature_builder import FeatureBuilder
from inducing_factor.pipeline.sampling import StratifiedBackgroundSampler

logger = logging.getLogger("InducingFactorSystem")


def _decode_disturbance_month(values: pd.Series, years: pd.Series) -> pd.Series:
    months = pd.Series(np.nan, index=values.index, dtype=float)
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_years = pd.to_numeric(years, errors="coerce")
    valid = numeric_values.notna() & numeric_years.notna() & (numeric_values > 0)
    if not valid.any():
        return months

    matched = np.floor(numeric_values[valid]).astype(int) == numeric_years[valid].astype(int)
    matched_index = numeric_values[valid].index[matched]
    for idx in matched_index:
        raw_value = float(numeric_values.loc[idx])
        year = int(numeric_years.loc[idx])
        fraction = min(0.999999, max(0.0, raw_value - year))
        days_in_year = 366 if date(year, 12, 31).toordinal() - date(year, 1, 1).toordinal() + 1 == 366 else 365
        day_of_year = min(days_in_year, max(1, int(round(fraction * days_in_year))))
        months.loc[idx] = (date(year, 1, 1) + timedelta(days=day_of_year - 1)).month
    return months


class EcoDisturbanceAttributionSystem:
    def __init__(
        self,
        cache_dir: str = "/Volumes/T9/Datasets/.cache/chunks",
        out_dir: str = "/Volumes/T9/Datasets/.cache/patent_system",
        lookback_years: int = 3,
        background_sample_rate: float = 0.05,
        discretizer_method: str = "supervised_tree",
        discretize_bins: int = 3,
        max_pattern_size: int = 3,
        min_support: float = 0.05,
        min_relative_risk: float = 1.5,
        bootstrap_iterations: int = 10,
        min_stability: float = 0.6,
        max_bootstrap_samples: int = 3000,
        max_network_factors: int = 6,
        dominant_threshold: float = 0.55,
        synergy_threshold: float = 0.75,
        random_state: int = 42,
        config: Optional[AppConfig] = None,
    ):
        if config is not None:
            # Override parameters from config
            cache_dir = config.paths.cache_dir
            out_dir = config.paths.out_dir
            lookback_years = config.attribution.lookback_years
            background_sample_rate = config.attribution.background_sample_rate
            discretizer_method = config.attribution.discretizer_method
            discretize_bins = config.attribution.discretize_bins
            max_pattern_size = config.attribution.max_pattern_size
            min_support = config.attribution.min_support
            min_relative_risk = config.attribution.min_relative_risk
            bootstrap_iterations = config.attribution.bootstrap_iterations
            min_stability = config.attribution.min_stability
            max_bootstrap_samples = config.attribution.max_bootstrap_samples
            max_network_factors = config.attribution.max_network_factors
            dominant_threshold = config.attribution.dominant_threshold
            synergy_threshold = config.attribution.synergy_threshold
            random_state = config.attribution.random_state
        self.cache_dir = cache_dir
        self.out_dir = out_dir
        self.max_network_factors = max_network_factors
        self.target = "disturbance_flag"
        # Elevation is intentionally excluded from factor mining to avoid over-emphasizing
        # broad regional elevation background over more actionable local drivers.
        self.static_features = ["slope", "aspect", "dist_to_road", "dist_to_river"]
        self.dynamic_features = ["temperature", "resident"]
        self.categorical_temporal_features = ["landuse"]
        self.direct_features = ["rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m", "rainfall_prev_total_3m"]
        self.metadata_columns = [
            "point_id",
            "raster_row",
            "raster_col",
            "x",
            "y",
            "year",
            "disturbance_time_value",
            "disturbance_type",
            "disturbance_class",
            "disturbance_month",
            "reference_month",
            self.target,
        ]
        self.random_state = random_state

        self.discretizer = AdaptiveDiscretizer(method=discretizer_method, n_bins=discretize_bins)
        self.factor_miner = FactorMiner(
            max_pattern_size=max_pattern_size,
            min_support=min_support,
            min_relative_risk=min_relative_risk,
            min_stability=min_stability,
            bootstrap_iterations=bootstrap_iterations,
            max_bootstrap_samples=max_bootstrap_samples,
            random_state=random_state,
        )
        self.feature_builder = FeatureBuilder(
            target=self.target,
            lookback_years=lookback_years,
            static_features=self.static_features,
            dynamic_features=self.dynamic_features,
            categorical_temporal_features=self.categorical_temporal_features,
            direct_features=self.direct_features,
        )
        self.background_sampler = StratifiedBackgroundSampler(self.static_features, background_sample_rate, random_state)
        self.attribution_engine = BayesianAttributionEngine(self.target, random_state, dominant_threshold, synergy_threshold)
        self.artifacts = ArtifactWriter(out_dir)
        self.class_engine = DisturbanceClassEngine(random_state=random_state)
        self.occurrence_engine = OccurrenceProbabilityEngine(random_state=random_state)

        self.feature_columns: List[str] = []
        self.training_samples: Optional[pd.DataFrame] = None
        self.factor_library: List[InducingFactor] = []
        self.selected_factors: List[InducingFactor] = []
        self.model = None
        self.inference_engine = None
        self.model_backend = self.attribution_engine.model_backend
        self.network_feature_columns: List[str] = []
        self.class_summary: Optional[pd.DataFrame] = None

    def _iter_chunk_files(self) -> Iterable[str]:
        files = sorted(glob.glob(os.path.join(self.cache_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No chunk files found in {self.cache_dir}")
        return files

    def _load_chunk_dataframe(self, npz_path: str) -> pd.DataFrame:
        import numpy as np

        data = np.load(npz_path)
        return pd.DataFrame({col: data[col] for col in data.files})

    @property
    def feature_window_map(self) -> Dict[str, int]:
        return self.feature_builder.feature_window_map

    def load_artifacts(self):
        metadata_path = os.path.join(self.out_dir, "model_metadata.json")
        factor_path = os.path.join(self.out_dir, "inducing_factor_library.json")
        discretizer_path = os.path.join(self.out_dir, "discretizer.joblib")
        bayes_path = os.path.join(self.out_dir, "bayesian_network.pkl")
        occurrence_path = os.path.join(self.out_dir, "occurrence_probability_model.joblib")
        class_path = os.path.join(self.out_dir, "disturbance_class_model.joblib")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.feature_columns = metadata.get("feature_columns", self.feature_columns)
            self.network_feature_columns = metadata.get("network_feature_columns", self.network_feature_columns)
        if os.path.exists(factor_path):
            with open(factor_path, "r", encoding="utf-8") as handle:
                factor_payload = json.load(handle)
            self.factor_library = [InducingFactor(**item) for item in factor_payload]
            self.selected_factors = self.factor_library[: self.max_network_factors]
        if os.path.exists(discretizer_path):
            self.discretizer = joblib_load(discretizer_path)
        if os.path.exists(bayes_path):
            with open(bayes_path, "rb") as handle:
                self.model = pickle.load(handle)
            self.attribution_engine.model = self.model
            self.attribution_engine.model_backend = getattr(self.model, "backend", self.attribution_engine.model_backend)
            if self.attribution_engine.model_backend == "pgmpy":
                from pgmpy.inference import VariableElimination

                self.attribution_engine.inference_engine = VariableElimination(self.model)
                self.inference_engine = self.attribution_engine.inference_engine
        if os.path.exists(occurrence_path):
            self.occurrence_engine.load(occurrence_path)
        if os.path.exists(class_path):
            self.class_engine.load(class_path)
        return self

    def _assign_reference_months(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        df = df_chunk.copy()
        disturbance_source = "disturbance_time_value" if "disturbance_time_value" in df.columns else "disturbance_type"
        if "disturbance_month" in df.columns:
            disturbance_month = pd.to_numeric(df["disturbance_month"], errors="coerce")
        else:
            disturbance_month = _decode_disturbance_month(df[disturbance_source], df["year"])
        disturbance_flag = (pd.to_numeric(df[disturbance_source], errors="coerce") > 0).astype(int)
        df["disturbance_flag"] = disturbance_flag
        df["disturbance_month"] = disturbance_month
        df["reference_month"] = disturbance_month
        bg_mask = disturbance_flag == 0
        if not bg_mask.any():
            return df

        observed_months = disturbance_month[disturbance_flag == 1].dropna().astype(int)
        rng = np.random.default_rng(self.random_state)
        if observed_months.empty:
            sampled_months = rng.integers(1, 13, size=int(bg_mask.sum()))
        else:
            month_values = observed_months.to_numpy()
            sampled_months = rng.choice(month_values, size=int(bg_mask.sum()), replace=True)
        df.loc[bg_mask, "reference_month"] = sampled_months
        return df

    def _build_chunk_training_samples(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        if "disturbance_time_value" not in df_chunk.columns and "disturbance_type" not in df_chunk.columns:
            raise KeyError("Input chunk is missing required disturbance time column")
        df_prepared = self._assign_reference_months(df_chunk)
        df, required_cols = self.feature_builder.build_chunk_features(df_prepared)
        if not required_cols:
            raise ValueError("No usable engineered features were created from chunk data")
        required_cols = [col for col in required_cols if col in df.columns and df[col].notna().any()]
        if not required_cols:
            return pd.DataFrame(columns=self.metadata_columns)
        history_mask = df[required_cols].notna().all(axis=1)
        metadata_columns = [col for col in self.metadata_columns if col in df.columns]
        df_ready = df.loc[history_mask, metadata_columns + required_cols].copy()
        if df_ready.empty:
            return df_ready
        df_dist = df_ready[df_ready[self.target] == 1]
        df_bg = df_ready[df_ready[self.target] == 0]
        if df_bg.empty:
            return df_dist
        df_bg_sampled = self.background_sampler.sample(df_bg, df_dist)
        combined = pd.concat([df_dist, df_bg_sampled], ignore_index=True)
        combined = combined.replace([float("inf"), float("-inf")], pd.NA)
        return combined.dropna(subset=required_cols)

    def build_training_samples(self) -> pd.DataFrame:
        logger.info("Building pre-disturbance training samples from chunk cache...")
        sampled_chunks = []
        for npz_path in self._iter_chunk_files():
            try:
                df_chunk = self._load_chunk_dataframe(npz_path)
                df_chunk_samples = self._build_chunk_training_samples(df_chunk)
                if not df_chunk_samples.empty:
                    sampled_chunks.append(df_chunk_samples)
            except Exception as exc:
                logger.warning("Failed to process chunk %s: %s", npz_path, exc)
        if not sampled_chunks:
            raise ValueError("No valid samples could be built from chunk files")
        df_samples = pd.concat(sampled_chunks, ignore_index=True)
        self.feature_columns = [
            c for c in df_samples.columns if c not in self.metadata_columns and df_samples[c].notna().any()
        ]
        df_samples = df_samples.dropna(subset=self.feature_columns)
        if df_samples.empty:
            raise ValueError("Training samples were built, but all rows were removed by final feature completeness checks")
        self.training_samples = df_samples
        self.artifacts.save_training_samples(df_samples)
        return df_samples

    def mine_inducing_factors(self, training_samples: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if training_samples is None:
            training_samples = self.training_samples if self.training_samples is not None else self.build_training_samples()
        self.discretizer.fit(training_samples[self.feature_columns], training_samples[self.target])
        label_df = self.discretizer.transform(training_samples[self.feature_columns], as_labels=True)
        event_labels = label_df[training_samples[self.target] == 1]
        background_labels = label_df[training_samples[self.target] == 0]
        event_transactions = self.factor_miner.transactions_from_labels(event_labels)
        background_transactions = self.factor_miner.transactions_from_labels(background_labels)
        pattern_candidates = self.factor_miner.mine_pattern_candidates(event_transactions, background_transactions)
        pattern_stability = self.factor_miner.estimate_pattern_stability(event_transactions, background_transactions)
        reduced_patterns = self.factor_miner.reduce_redundant_patterns(pattern_candidates, pattern_stability)
        self.factor_library = self.factor_miner.build_factor_library(reduced_patterns)
        self.selected_factors = self.factor_library[: self.max_network_factors]
        if not self.factor_library:
            raise ValueError("No stable inducing factors were discovered. Consider relaxing thresholds.")
        df_factors = self.factor_miner.to_frame(self.factor_library)
        self.artifacts.save_factor_library(self.factor_library, df_factors, self.discretizer)
        return df_factors

    def fit_bayesian_network(self, training_samples: Optional[pd.DataFrame] = None):
        if training_samples is None:
            training_samples = self.training_samples if self.training_samples is not None else self.build_training_samples()
        if not self.selected_factors:
            self.mine_inducing_factors(training_samples)
        label_df = self.discretizer.transform(training_samples[self.feature_columns], as_labels=True)
        self.model = self.attribution_engine.fit(training_samples, label_df, self.selected_factors, self.feature_columns, self.out_dir)
        self.inference_engine = self.attribution_engine.inference_engine
        self.model_backend = self.attribution_engine.model_backend
        self.network_feature_columns = self.attribution_engine.network_feature_columns
        return self.model

    def fit_occurrence_probability_model(self, training_samples: Optional[pd.DataFrame] = None):
        if training_samples is None:
            training_samples = self.training_samples if self.training_samples is not None else self.build_training_samples()
        if not self.selected_factors:
            self.mine_inducing_factors(training_samples)
        label_df = self.discretizer.transform(training_samples[self.feature_columns], as_labels=True)
        self.occurrence_engine.fit(label_df, training_samples[self.target], self.selected_factors)
        self.artifacts.save_occurrence_model(self.occurrence_engine)
        return self.occurrence_engine.model

    def fit_disturbance_class_model(self, training_samples: Optional[pd.DataFrame] = None):
        if training_samples is None:
            training_samples = self.training_samples if self.training_samples is not None else self.build_training_samples()
        if not self.selected_factors:
            self.mine_inducing_factors(training_samples)
        label_df = self.discretizer.transform(training_samples[self.feature_columns], as_labels=True)
        self.class_engine.fit(label_df, training_samples, self.selected_factors)
        self.class_summary = self.class_engine.summarize_factor_class_associations(
            label_df,
            training_samples,
            self.selected_factors,
        )
        self.artifacts.save_class_model(self.class_engine, self.class_summary)
        return self.class_engine.model

    def attribute_sample(self, sample_row: pd.Series) -> dict:
        result = self.attribution_engine.attribute_sample(sample_row, self.discretizer, self.feature_columns, self.selected_factors)
        if self.class_engine.model is not None:
            sample_df = pd.DataFrame([sample_row[self.feature_columns].to_dict()])
            label_df = self.discretizer.transform(sample_df, as_labels=True)
            class_probs = self.class_engine.predict_proba(label_df, self.selected_factors)
            if not class_probs.empty:
                row = class_probs.iloc[0]
                result["disturbance_class_probabilities"] = row.to_dict()
                result["dominant_disturbance_class"] = int(row.astype(float).idxmax().rsplit("_", 1)[-1])
        return result

    def attribute_dataframe(self, samples: pd.DataFrame, limit: Optional[int] = 100) -> pd.DataFrame:
        if limit is not None:
            samples = samples.head(limit)
        df_attr = pd.DataFrame([self.attribute_sample(row) for _, row in samples.iterrows()])
        self.artifacts.save_attribution_preview(df_attr)
        return df_attr

    def predict_dataframe(self, samples: pd.DataFrame) -> pd.DataFrame:
        label_df = self.discretizer.transform(samples[self.feature_columns], as_labels=True)
        occurrence_probs = self.occurrence_engine.predict_proba(label_df, self.selected_factors)
        result = samples[[col for col in ["point_id", "raster_row", "raster_col", "x", "y", "year"] if col in samples.columns]].copy()
        if occurrence_probs.size:
            result["disturbance_probability"] = occurrence_probs[:, 1]
        else:
            result["disturbance_probability"] = 0.0
        class_probs = self.class_engine.predict_proba(label_df, self.selected_factors)
        if not class_probs.empty:
            result = pd.concat([result.reset_index(drop=True), class_probs.reset_index(drop=True)], axis=1)
            prob_cols = [col for col in result.columns if col.startswith("class_probability_")]
            if prob_cols:
                result["dominant_disturbance_class"] = result[prob_cols].idxmax(axis=1).str.rsplit("_", n=1).str[-1].astype(int)
        return result

    def run(self) -> dict:
        samples = self.build_training_samples()
        factors = self.mine_inducing_factors(samples)
        self.fit_occurrence_probability_model(samples)
        self.fit_bayesian_network(samples)
        self.fit_disturbance_class_model(samples)
        self.artifacts.save_model_metadata(
            {
                "feature_columns": self.feature_columns,
                "network_feature_columns": self.network_feature_columns,
                "selected_factor_ids": [factor.factor_id for factor in self.selected_factors],
                "class_labels": self.class_engine.class_labels,
                "target": self.target,
            }
        )
        attribution = self.attribute_dataframe(samples)
        return {
            "training_samples": samples,
            "factors": factors,
            "factor_class_summary": self.class_summary,
            "attribution_preview": attribution,
        }

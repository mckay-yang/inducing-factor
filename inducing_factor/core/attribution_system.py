import glob
import json
import logging
import os
import pickle
from collections import Counter
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
from inducing_factor.utils.progress import ProgressTracker

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
        cache_dir: str = "data/.cache/chunks",
        out_dir: str = "data/.cache/patent_system",
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
        discretizer_sample_limit: int = 200000,
        model_sample_limit: int = 200000,
        progress_log_every: int = 1,
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
            discretizer_sample_limit = getattr(config.attribution, "discretizer_sample_limit", discretizer_sample_limit)
            model_sample_limit = getattr(config.attribution, "model_sample_limit", model_sample_limit)
            progress_log_every = getattr(config.attribution, "progress_log_every", progress_log_every)
        self.cache_dir = cache_dir
        self.out_dir = out_dir
        self.max_network_factors = max_network_factors
        self.discretizer_sample_limit = discretizer_sample_limit
        self.model_sample_limit = model_sample_limit
        self.progress_log_every = progress_log_every
        self.target = "disturbance_flag"
        # Elevation is intentionally excluded from factor mining to avoid over-emphasizing
        # broad regional elevation background over more actionable local drivers.
        self.static_features = ["slope", "aspect", "dist_to_river"]
        self.dynamic_features = ["temperature", "resident", "dist_to_road", "dist_to_railway", "dist_to_waterway"]
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

    def _update_sample_frame(self, current: Optional[pd.DataFrame], new_rows: pd.DataFrame, limit: int) -> pd.DataFrame:
        if limit <= 0:
            return new_rows.iloc[0:0].copy()
        if new_rows.empty:
            return current.copy() if current is not None else new_rows.copy()
        if current is None or current.empty:
            if len(new_rows) <= limit:
                return new_rows.copy()
            return new_rows.sample(n=limit, random_state=self.random_state).reset_index(drop=True)
        combined = pd.concat([current, new_rows], ignore_index=True)
        if len(combined) <= limit:
            return combined
        return combined.sample(n=limit, random_state=self.random_state).reset_index(drop=True)

    def _load_chunk_samples(self, chunk_files: List[str]):
        for idx, npz_path in enumerate(chunk_files, start=1):
            try:
                # First sample indices using numpy (memory efficient)
                selected_indices = self._sample_chunk_indices(npz_path)
                if len(selected_indices) == 0:
                    # No rows selected, return empty DataFrame
                    yield idx, npz_path, pd.DataFrame(), None
                    continue
                
                # Load only the sampled rows
                df_chunk = self._load_chunk_dataframe(npz_path, row_indices=selected_indices)
                # Skip sampling since we already sampled in numpy
                df_chunk_samples = self._build_chunk_training_samples(df_chunk, skip_sampling=True)
                yield idx, npz_path, df_chunk_samples, None
            except Exception as exc:
                logger.warning("Failed to process chunk %s: %s", npz_path, exc)
                yield idx, npz_path, pd.DataFrame(), exc

    def _streaming_run(self) -> dict:
        chunk_files = list(self._iter_chunk_files())
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self.cache_dir}")

        progress = ProgressTracker(logger, "run", log_every=self.progress_log_every)
        stage_total = 3
        feature_columns: List[str] = []
        discretizer_sample: Optional[pd.DataFrame] = None
        event_sample_count = 0
        background_sample_count = 0

        progress.start_stage(1, stage_total, "scan & fit discretizer", total=len(chunk_files))
        for idx, npz_path, df_chunk_samples, exc in self._load_chunk_samples(chunk_files):
            if exc is None and not df_chunk_samples.empty:
                cols = [c for c in df_chunk_samples.columns if c not in self.metadata_columns and df_chunk_samples[c].notna().any()]
                for col in cols:
                    if col not in feature_columns:
                        feature_columns.append(col)
                discretizer_sample = self._update_sample_frame(discretizer_sample, df_chunk_samples, self.discretizer_sample_limit)
                event_sample_count += int(df_chunk_samples[self.target].sum())
                background_sample_count += int((df_chunk_samples[self.target] == 0).sum())
            progress.advance(1, {"chunk": idx, "events": event_sample_count, "background": background_sample_count})
        progress.finish_stage({"events": event_sample_count, "background": background_sample_count})

        if discretizer_sample is None or discretizer_sample.empty:
            raise ValueError("No valid samples could be built from chunk files")

        self.feature_columns = feature_columns
        # Ensure discretizer_sample has all feature columns (fill missing with NaN)
        for col in self.feature_columns:
            if col not in discretizer_sample.columns:
                discretizer_sample[col] = np.nan
        discretizer_sample = discretizer_sample.dropna(subset=self.feature_columns)
        self.discretizer.fit(discretizer_sample[self.feature_columns], discretizer_sample[self.target])

        progress.start_stage(2, stage_total, "pattern counts & model samples", total=len(chunk_files))
        event_counts: Counter = Counter()
        background_counts: Counter = Counter()
        n_event = 0
        n_bg = 0
        event_label_sample: Optional[pd.DataFrame] = None
        background_label_sample: Optional[pd.DataFrame] = None
        event_training_sample: Optional[pd.DataFrame] = None
        background_training_sample: Optional[pd.DataFrame] = None
        event_limit = max(1, self.model_sample_limit // 2)
        background_limit = max(1, self.model_sample_limit - event_limit)
        for idx, npz_path, df_chunk_samples, exc in self._load_chunk_samples(chunk_files):
            if exc is None and not df_chunk_samples.empty:
                # Ensure df_chunk_samples has all feature columns (fill missing with NaN)
                for col in self.feature_columns:
                    if col not in df_chunk_samples.columns:
                        df_chunk_samples[col] = np.nan
                label_df = self.discretizer.transform(df_chunk_samples[self.feature_columns], as_labels=True)
                chunk_event, chunk_bg = self.factor_miner.update_pattern_counts(label_df, df_chunk_samples[self.target], event_counts, background_counts)
                n_event += chunk_event
                n_bg += chunk_bg
                event_label_sample = self._update_sample_frame(
                    event_label_sample,
                    label_df[df_chunk_samples[self.target] == 1],
                    self.factor_miner.max_bootstrap_samples,
                )
                background_label_sample = self._update_sample_frame(
                    background_label_sample,
                    label_df[df_chunk_samples[self.target] == 0],
                    self.factor_miner.max_bootstrap_samples,
                )
                event_training_sample = self._update_sample_frame(
                    event_training_sample,
                    df_chunk_samples[df_chunk_samples[self.target] == 1],
                    event_limit,
                )
                background_training_sample = self._update_sample_frame(
                    background_training_sample,
                    df_chunk_samples[df_chunk_samples[self.target] == 0],
                    background_limit,
                )
            progress.advance(1, {
                "chunk": idx,
                "event_patterns": len(event_counts),
                "events": n_event,
                "model_event": 0 if event_training_sample is None else len(event_training_sample),
                "model_bg": 0 if background_training_sample is None else len(background_training_sample),
            })
        progress.finish_stage({"event_patterns": len(event_counts), "events": n_event, "background": n_bg})

        pattern_candidates = self.factor_miner.mine_pattern_candidates_from_counts(event_counts, background_counts, n_event, n_bg)
        event_transactions = self.factor_miner.transactions_from_labels(event_label_sample) if event_label_sample is not None else []
        background_transactions = self.factor_miner.transactions_from_labels(background_label_sample) if background_label_sample is not None else []
        pattern_stability = self.factor_miner.estimate_pattern_stability(event_transactions, background_transactions)
        reduced_patterns = self.factor_miner.reduce_redundant_patterns(pattern_candidates, pattern_stability)
        self.factor_library = self.factor_miner.build_factor_library(reduced_patterns)
        self.selected_factors = self.factor_library[: self.max_network_factors]
        if not self.factor_library:
            raise ValueError("No stable inducing factors were discovered. Consider relaxing thresholds.")
        df_factors = self.factor_miner.to_frame(self.factor_library)
        self.artifacts.save_factor_library(self.factor_library, df_factors, self.discretizer)

        sampled_frames = [frame for frame in [event_training_sample, background_training_sample] if frame is not None and not frame.empty]
        if not sampled_frames:
            raise ValueError("No model training samples were collected during streaming run")
        training_samples = pd.concat(sampled_frames, ignore_index=True)
        training_samples = training_samples.dropna(subset=self.feature_columns)
        self.training_samples = training_samples
        self.artifacts.save_training_samples(training_samples)

        progress.start_stage(3, stage_total, "fit models and export artifacts", total=4)
        label_df = self.discretizer.transform(training_samples[self.feature_columns], as_labels=True)
        progress.advance(1, {"rows": len(training_samples), "factors": len(self.selected_factors)})
        self.occurrence_engine.fit(label_df, training_samples[self.target], self.selected_factors)
        self.artifacts.save_occurrence_model(self.occurrence_engine)
        progress.advance(1, {"occurrence_model": self.occurrence_engine.model is not None})
        self.model = self.attribution_engine.fit(training_samples, label_df, self.selected_factors, self.feature_columns, self.out_dir)
        self.inference_engine = self.attribution_engine.inference_engine
        self.model_backend = self.attribution_engine.model_backend
        self.network_feature_columns = self.attribution_engine.network_feature_columns
        progress.advance(1, {"bayes_backend": self.model_backend})
        self.class_engine.fit(label_df, training_samples, self.selected_factors)
        self.class_summary = self.class_engine.summarize_factor_class_associations(label_df, training_samples, self.selected_factors)
        self.artifacts.save_class_model(self.class_engine, self.class_summary)
        self.artifacts.save_model_metadata(
            {
                "feature_columns": self.feature_columns,
                "network_feature_columns": self.network_feature_columns,
                "selected_factor_ids": [factor.factor_id for factor in self.selected_factors],
                "class_labels": self.class_engine.class_labels,
                "target": self.target,
                "streaming_run": True,
                "discretizer_sample_limit": self.discretizer_sample_limit,
                "model_sample_limit": self.model_sample_limit,
            }
        )
        attribution = self.attribute_dataframe(training_samples)
        progress.advance(1, {"preview_rows": len(attribution)})
        progress.finish_stage({"factors": len(self.factor_library), "preview_rows": len(attribution)})
        return {
            "training_samples": training_samples,
            "factors": df_factors,
            "factor_class_summary": self.class_summary,
            "attribution_preview": attribution,
        }

    def _iter_chunk_files(self) -> Iterable[str]:
        files = sorted(glob.glob(os.path.join(self.cache_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No chunk files found in {self.cache_dir}")
        return files

    def _load_chunk_dataframe(self, npz_path: str, row_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        import numpy as np

        data = np.load(npz_path)
        if row_indices is None:
            return pd.DataFrame({col: data[col] for col in data.files})
        
        # Load only selected rows
        result = {}
        for col in data.files:
            arr = data[col]
            if len(arr) == 0:
                result[col] = arr
            else:
                result[col] = arr[row_indices]
        return pd.DataFrame(result)

    def _sample_chunk_indices(self, npz_path: str) -> np.ndarray:
        """Load only disturbance_time_value and point_id, sample background point_ids, return indices of rows to keep."""
        import numpy as np
        import time
        
        data = np.load(npz_path)
        try:
            # Get disturbance column (could be disturbance_time_value or disturbance_type)
            if "disturbance_time_value" in data.files:
                dist_col = "disturbance_time_value"
            elif "disturbance_type" in data.files:
                dist_col = "disturbance_type"
            else:
                raise KeyError("Input chunk is missing required disturbance time column")
            
            disturbance = data[dist_col]
            point_id = data["point_id"]
            n_total = len(disturbance)
            logger.debug("_sample_chunk_indices: loaded %d rows from %s", n_total, os.path.basename(npz_path))
            
            # Event rows: disturbance > 0
            event_mask = disturbance > 0
            event_indices = np.where(event_mask)[0]
            logger.debug("  event rows: %d", len(event_indices))
            
            # Background rows
            background_mask = ~event_mask
            
            if not background_mask.any():
                # No background rows, just return all event rows
                return event_indices
            
            # Get unique background point_ids
            t0 = time.time()
            bg_point_ids = np.unique(point_id[background_mask])
            t_unique = time.time() - t0
            logger.debug("  unique background point_ids: %d (took %.1fs)", len(bg_point_ids), t_unique)
            if len(bg_point_ids) == 0:
                return event_indices
            
            # Sample background point_ids (simple random sampling, stratified is too heavy for 100M rows)
            sample_n = max(1, int(len(bg_point_ids) * self.background_sampler.background_sample_rate))
            sample_n = min(sample_n, len(bg_point_ids))
            
            rng = np.random.default_rng(self.random_state)
            sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)
            logger.debug("  sampled background point_ids: %d", sample_n)
            
            # Find rows where point_id is in sampled_bg_point_ids
            # Use boolean mask for memory efficiency
            t0 = time.time()
            bg_selected_mask = np.isin(point_id, sampled_bg_point_ids)
            bg_selected_indices = np.where(bg_selected_mask & background_mask)[0]
            t_isin = time.time() - t0
            logger.debug("  np.isin selected %d background rows (took %.1fs)", len(bg_selected_indices), t_isin)
            
            # Combine event and selected background indices
            selected_indices = np.concatenate([event_indices, bg_selected_indices])
            logger.debug("  total selected rows: %d (%.2f%%)", len(selected_indices), len(selected_indices)/n_total*100)
            return np.sort(selected_indices)
        finally:
            data.close()
    
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

    def _build_chunk_training_samples(self, df_chunk: pd.DataFrame, skip_sampling: bool = False) -> pd.DataFrame:
        if "disturbance_time_value" not in df_chunk.columns and "disturbance_type" not in df_chunk.columns:
            raise KeyError("Input chunk is missing required disturbance time column")
        
        # 1. Create disturbance_flag target column
        disturbance_source = "disturbance_time_value" if "disturbance_time_value" in df_chunk.columns else "disturbance_type"
        disturbance_flag = (pd.to_numeric(df_chunk[disturbance_source], errors="coerce") > 0).astype(int)
        
        # 2. Identify event and background rows
        event_mask = disturbance_flag == 1
        background_mask = disturbance_flag == 0
        
        # If no background rows, just process all event rows
        if not background_mask.any():
            df_subset = df_chunk[event_mask].copy()
            df_prepared = self._assign_reference_months(df_subset)
            df, required_cols = self.feature_builder.build_chunk_features(df_prepared)
            if not required_cols:
                return pd.DataFrame(columns=self.metadata_columns)
            required_cols = [col for col in required_cols if col in df.columns and df[col].notna().any()]
            if not required_cols:
                return pd.DataFrame(columns=self.metadata_columns)
            history_mask = df[required_cols].notna().all(axis=1)
            metadata_columns = [col for col in self.metadata_columns if col in df.columns]
            df_ready = df.loc[history_mask, metadata_columns + required_cols].copy()
            return df_ready.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=required_cols)
        
        if skip_sampling:
            # Input DataFrame is already sampled, use it directly
            df_subset = df_chunk.copy()
        else:
            # 3. Prepare small DataFrames for stratified sampling (only static features + point_id)
            static_features = [f for f in self.static_features if f in df_chunk.columns]
            # Include elevation as well for stratification (though excluded from factor mining)
            if "elevation" in df_chunk.columns and "elevation" not in static_features:
                static_features.append("elevation")
            
            if not static_features:
                # Fallback to random sampling by point_id
                background_point_ids = df_chunk.loc[background_mask, "point_id"].unique()
                sample_n = max(1, int(len(background_point_ids) * self.background_sampler.background_sample_rate))
                sample_n = min(sample_n, len(background_point_ids))
                rng = np.random.default_rng(self.random_state)
                sampled_point_ids = rng.choice(background_point_ids, size=sample_n, replace=False)
                selected_mask = event_mask | (background_mask & df_chunk["point_id"].isin(sampled_point_ids))
            else:
                # Create small DataFrames for stratification
                df_bg_small = df_chunk.loc[background_mask, ["point_id"] + static_features].copy()
                df_dist_small = df_chunk.loc[event_mask, ["point_id"] + static_features].copy()
                
                # Add disturbance_flag column for compatibility with background_sampler
                df_bg_small[self.target] = 0
                df_dist_small[self.target] = 1
                
                # Perform stratified sampling on the small DataFrames
                df_bg_sampled_small = self.background_sampler.sample(df_bg_small, df_dist_small)
                
                # Get sampled point_ids
                sampled_point_ids = df_bg_sampled_small["point_id"].unique()
                selected_mask = event_mask | (background_mask & df_chunk["point_id"].isin(sampled_point_ids))
            
            # 4. Select the subset of rows (events + sampled background points)
            df_subset = df_chunk[selected_mask].copy()
        
        # 5. Assign reference months and build features
        df_prepared = self._assign_reference_months(df_subset)
        df, required_cols = self.feature_builder.build_chunk_features(df_prepared)
        
        if not required_cols:
            return pd.DataFrame(columns=self.metadata_columns)
        required_cols = [col for col in required_cols if col in df.columns and df[col].notna().any()]
        if not required_cols:
            return pd.DataFrame(columns=self.metadata_columns)
        
        # 6. Filter rows with complete feature history
        history_mask = df[required_cols].notna().all(axis=1)
        metadata_columns = [col for col in self.metadata_columns if col in df.columns]
        df_ready = df.loc[history_mask, metadata_columns + required_cols].copy()
        
        if df_ready.empty:
            return df_ready
        
        # 7. Final cleanup
        df_ready = df_ready.replace([float("inf"), float("-inf")], pd.NA)
        return df_ready.dropna(subset=required_cols)

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
        return self._streaming_run()

import glob
import json
import logging
import os
import pickle
from collections import Counter
from datetime import date, timedelta
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from inducing_factor.config.models import AppConfig
from inducing_factor.core.bayesian_model import BayesianAttributionEngine
from inducing_factor.core.class_conditional_analysis import FactorClassAnalyzer
from inducing_factor.core.class_model import DisturbanceClassEngine
from inducing_factor.core.discretization import AdaptiveDiscretizer
from inducing_factor.core.factor_mining import FactorMiner
from inducing_factor.core.models import InducingFactor
from inducing_factor.core.occurrence_model import OccurrenceProbabilityEngine
from inducing_factor.io.artifacts import ArtifactWriter
from inducing_factor.pipeline.execution import ChunkTask, execute_chunk_tasks, resolve_num_workers
from inducing_factor.pipeline.feature_builder import FeatureBuilder
from inducing_factor.pipeline.sampling import StratifiedBackgroundSampler
from inducing_factor.utils.progress import ProgressTracker

logger = logging.getLogger("InducingFactorSystem")

_RUN_SAMPLE_WORKER_SYSTEM: "EcoDisturbanceAttributionSystem | None" = None
_RUN_SAMPLE_WORKER_CHUNK_FILES: tuple[str, ...] = ()
_RUN_PATTERN_WORKER_SYSTEM: "EcoDisturbanceAttributionSystem | None" = None
_RUN_PATTERN_WORKER_CHUNK_FILES: tuple[str, ...] = ()
_RUN_PATTERN_WORKER_FEATURE_COLUMNS: tuple[str, ...] = ()


def _init_run_sample_worker(system: "EcoDisturbanceAttributionSystem", chunk_files: list[str]):
    global _RUN_SAMPLE_WORKER_SYSTEM, _RUN_SAMPLE_WORKER_CHUNK_FILES
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _RUN_SAMPLE_WORKER_SYSTEM = system
    _RUN_SAMPLE_WORKER_CHUNK_FILES = tuple(chunk_files)


def _run_sample_worker(task: ChunkTask):
    if _RUN_SAMPLE_WORKER_SYSTEM is None:
        raise RuntimeError("Run sample worker was not initialized")
    npz_path = _RUN_SAMPLE_WORKER_CHUNK_FILES[task.chunk_idx - 1]
    return _RUN_SAMPLE_WORKER_SYSTEM._load_chunk_sample_entry(task.chunk_idx, npz_path)


def _init_run_pattern_worker(system: "EcoDisturbanceAttributionSystem", chunk_files: list[str], feature_columns: list[str]):
    global _RUN_PATTERN_WORKER_SYSTEM, _RUN_PATTERN_WORKER_CHUNK_FILES, _RUN_PATTERN_WORKER_FEATURE_COLUMNS
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _RUN_PATTERN_WORKER_SYSTEM = system
    _RUN_PATTERN_WORKER_CHUNK_FILES = tuple(chunk_files)
    _RUN_PATTERN_WORKER_FEATURE_COLUMNS = tuple(feature_columns)


def _run_pattern_worker(task: ChunkTask, _max_retries: int = 3):
    if _RUN_PATTERN_WORKER_SYSTEM is None:
        raise RuntimeError("Run pattern worker was not initialized")
    npz_path = _RUN_PATTERN_WORKER_CHUNK_FILES[task.chunk_idx - 1]
    for attempt in range(1, _max_retries + 1):
        try:
            return _RUN_PATTERN_WORKER_SYSTEM._map_pattern_count_entry(npz_path, list(_RUN_PATTERN_WORKER_FEATURE_COLUMNS))
        except EOFError:
            if attempt == _max_retries:
                raise
            import time
            time.sleep(2 * attempt)


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
        cache_dir: str = "data/cache/chunks",
        out_dir: str = "data/result/models",
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
        score_method: str = "log_rr",
        spatial_buffer_distance: float = 0.0,
        deduplicate_temporal: bool = True,
        random_state: int = 42,
        discretizer_sample_limit: int = 200000,
        model_sample_limit: int = 200000,
        progress_log_every: int = 1,
        train_end_year: Optional[int] = None,
        validation_years: Optional[list[int]] = None,
        validation_required: bool = True,
        validation_metric_thresholds: Optional[dict[str, float]] = None,
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
            score_method = getattr(config.attribution, "score_method", score_method)
            spatial_buffer_distance = getattr(config.attribution, "spatial_buffer_distance", spatial_buffer_distance)
            deduplicate_temporal = getattr(config.attribution, "deduplicate_temporal", deduplicate_temporal)
            self.bdeu_equivalent_sample_size = getattr(config.attribution, "bdeu_equivalent_sample_size", 10.0)
            self.bn_bootstrap_iterations = getattr(config.attribution, "bn_bootstrap_iterations", 10)
            self.bn_edge_frequency_threshold = getattr(config.attribution, "bn_edge_frequency_threshold", 0.5)
            self.redundancy_threshold = getattr(config.attribution, "redundancy_threshold", 0.1)
            discretizer_sample_limit = getattr(config.attribution, "discretizer_sample_limit", discretizer_sample_limit)
            model_sample_limit = getattr(config.attribution, "model_sample_limit", model_sample_limit)
            progress_log_every = getattr(config.attribution, "progress_log_every", progress_log_every)
            self.run_num_workers = getattr(config.attribution, "run_num_workers", 0)
            self.run_max_auto_workers = getattr(config.attribution, "run_max_auto_workers", 4)
            self.run_memory_target_fraction = getattr(config.attribution, "run_memory_target_fraction", 0.7)
            self.run_estimated_worker_gb = getattr(config.attribution, "run_estimated_worker_gb", 1.5)
            self.predict_num_workers = getattr(config.attribution, "predict_num_workers", 0)
            self.predict_max_auto_workers = getattr(config.attribution, "predict_max_auto_workers", 4)
            self.predict_memory_target_fraction = getattr(config.attribution, "predict_memory_target_fraction", 0.7)
            self.predict_estimated_worker_gb = getattr(config.attribution, "predict_estimated_worker_gb", 1.0)
            self.analyze_by_class = getattr(config.attribution, "analyze_by_class", False)
            self.min_count_per_class = getattr(config.attribution, "min_count_per_class", 10)
            self.class_prior_lift_threshold = getattr(config.attribution, "class_prior_lift_threshold", 2.0)
            self.top_n_per_class = getattr(config.attribution, "top_n_per_class", 20)
            self.train_end_year = getattr(config.attribution, "train_end_year", None)
            self.validation_years = getattr(config.attribution, "validation_years", None)
            self.validation_required = getattr(config.attribution, "validation_required", True)
            self.validation_metric_thresholds = getattr(config.attribution, "validation_metric_thresholds", {"auc_min": 0.60, "pr_auc_min": 0.05})
            self.config = config
        else:
            self.analyze_by_class = False
            self.min_count_per_class = 10
            self.class_prior_lift_threshold = 2.0
            self.top_n_per_class = 20
            self.train_end_year = train_end_year
            self.validation_years = validation_years
            self.validation_metric_thresholds = validation_metric_thresholds or {"auc_min": 0.60, "pr_auc_min": 0.05}
            self.validation_required = validation_required
            self.config = None
            self.run_num_workers = 0
            self.run_max_auto_workers = 4
            self.run_memory_target_fraction = 0.7
            self.run_estimated_worker_gb = 1.5
            self.predict_num_workers = 0
            self.predict_max_auto_workers = 4
            self.predict_memory_target_fraction = 0.7
            self.predict_estimated_worker_gb = 1.0
        self.cache_dir = cache_dir
        self.out_dir = out_dir
        self.max_network_factors = max_network_factors
        self.discretizer_sample_limit = discretizer_sample_limit
        self.model_sample_limit = model_sample_limit
        self.progress_log_every = progress_log_every
        self.spatial_buffer_distance = spatial_buffer_distance
        self.deduplicate_temporal = deduplicate_temporal
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
            score_method=score_method,
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
        self.attribution_engine = BayesianAttributionEngine(
            self.target, random_state, dominant_threshold, synergy_threshold,
            bdeu_equivalent_sample_size=getattr(self, "bdeu_equivalent_sample_size", 10.0),
            bn_bootstrap_iterations=getattr(self, "bn_bootstrap_iterations", 10),
            bn_edge_frequency_threshold=getattr(self, "bn_edge_frequency_threshold", 0.5),
        )
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

    def _resolve_run_worker_count(self) -> int:
        return resolve_num_workers(
            num_workers=self.run_num_workers,
            max_auto_workers=self.run_max_auto_workers,
            memory_target_fraction=self.run_memory_target_fraction,
            estimated_worker_gb=self.run_estimated_worker_gb,
        )

    def _load_chunk_sample_entry(self, idx: int, npz_path: str):
        logger.info("[run][stage 1] loading chunk %d: %s", idx, os.path.basename(npz_path))
        try:
            selected_indices = self._sample_chunk_indices(npz_path)
            if len(selected_indices) == 0:
                logger.info("[run][stage 1] chunk %d has no valid indices", idx)
                return idx, npz_path, pd.DataFrame(), None
            df_chunk = self._load_chunk_dataframe(npz_path, row_indices=selected_indices)
            df_chunk_samples = self._build_chunk_training_samples(df_chunk, skip_sampling=True)
            df_chunk_samples = self._filter_chunk_by_train_years(df_chunk_samples)
            logger.info("[run][stage 1] chunk %d loaded: %d rows", idx, len(df_chunk_samples))
            return idx, npz_path, df_chunk_samples, None
        except Exception as exc:
            logger.warning("Failed to process chunk %s: %s", npz_path, exc)
            return idx, npz_path, pd.DataFrame(), exc

    def _load_chunk_samples(self, chunk_files: List[str]):
        worker_count = self._resolve_run_worker_count()
        logger.info("[run][stage 1] loading chunk samples: %d chunks, %d workers", len(chunk_files), worker_count)
        if worker_count <= 1:
            logger.info("[run][stage 1] using sequential loading")
            for idx, npz_path in enumerate(chunk_files, start=1):
                yield self._load_chunk_sample_entry(idx, npz_path)
            return

        logger.info("[run][stage 1] starting parallel loading with %d workers", worker_count)
        tasks = [ChunkTask(chunk_idx=idx, row_off=0, col_off=0, width=1, height=1) for idx in range(1, len(chunk_files) + 1)]
        results = execute_chunk_tasks(
            tasks,
            worker_count=worker_count,
            submit_fn=_run_sample_worker,
            initializer=_init_run_sample_worker,
            initargs=(self, chunk_files),
        )
        processed = 0
        for result in results:
            processed += 1
            if processed % 100 == 0:
                logger.info("[run][stage 1] processed %d/%d chunks", processed, len(chunk_files))
            yield result
        logger.info("[run][stage 1] parallel loading completed: %d chunks processed", processed)

    def _map_pattern_count_entry(self, npz_path: str, feature_columns: list[str]) -> dict:
        chunk_name = os.path.basename(npz_path)
        logger.debug("[run][stage 2] mapping pattern counts for chunk: %s", chunk_name)
        try:
            data = np.load(npz_path, allow_pickle=True)
        except EOFError:
            logger.warning("[run][stage 2] skipping corrupt chunk: %s", chunk_name)
            return {
                "event_counts": Counter(),
                "background_counts": Counter(),
                "n_event": 0,
                "n_bg": 0,
                "n_event_by_class": {},
                "event_counts_by_class": {},
                "event_label_sample": pd.DataFrame(),
                "background_label_sample": pd.DataFrame(),
                "event_training_sample": pd.DataFrame(),
                "background_training_sample": pd.DataFrame(),
            }
        try:
            selected_indices = self._sample_chunk_indices_from_data(data)
            empty_result = {
                "event_counts": Counter(),
                "background_counts": Counter(),
                "n_event": 0,
                "n_bg": 0,
                "n_event_by_class": {},
                "event_counts_by_class": {},
                "event_label_sample": pd.DataFrame(),
                "background_label_sample": pd.DataFrame(),
                "event_training_sample": pd.DataFrame(),
                "background_training_sample": pd.DataFrame(),
            }
            if len(selected_indices) == 0:
                logger.debug("[run][stage 2] chunk %s has no valid indices", chunk_name)
                return empty_result

            df_chunk = self._load_chunk_dataframe_from_data(data, row_indices=selected_indices)
            df_chunk_samples = self._build_chunk_training_samples(df_chunk, skip_sampling=True)
            df_chunk_samples = self._filter_chunk_by_train_years(df_chunk_samples)
            if df_chunk_samples.empty:
                logger.debug("[run][stage 2] chunk %s produced empty samples", chunk_name)
                return empty_result

            for col in feature_columns:
                if col not in df_chunk_samples.columns:
                    df_chunk_samples[col] = np.nan

            label_df = self.discretizer.transform(df_chunk_samples[feature_columns], as_labels=True)
            event_counts = Counter()
            background_counts = Counter()
            event_counts_by_class: Dict[int, Counter] = {}
            n_event_by_class: Dict[int, int] = {}
            class_values = df_chunk_samples["disturbance_class"] if "disturbance_class" in df_chunk_samples.columns else None
            chunk_event, chunk_bg = self.factor_miner.update_pattern_counts_by_class(
                label_df,
                df_chunk_samples[self.target],
                event_counts,
                background_counts,
                event_counts_by_class,
                n_event_by_class,
                class_values=class_values,
            )
            if chunk_event > 0 or chunk_bg > 0:
                logger.info("[run][stage 2] chunk %s: %d events, %d background", chunk_name, chunk_event, chunk_bg)
            else:
                logger.debug("[run][stage 2] chunk %s: 0 events, 0 background", chunk_name)
            return {
                "event_counts": event_counts,
                "background_counts": background_counts,
                "n_event": chunk_event,
                "n_bg": chunk_bg,
                "n_event_by_class": n_event_by_class,
                "event_counts_by_class": event_counts_by_class,
                "event_label_sample": label_df[df_chunk_samples[self.target] == 1],
                "background_label_sample": label_df[df_chunk_samples[self.target] == 0],
                "event_training_sample": df_chunk_samples[df_chunk_samples[self.target] == 1],
                "background_training_sample": df_chunk_samples[df_chunk_samples[self.target] == 0],
            }
        finally:
            data.close()

    def _map_pattern_counts(self, chunk_files: list[str], num_workers: int = 1) -> Iterator[dict]:
        if num_workers <= 1:
            logger.info("[run][stage 2] mapping pattern counts: %d chunks (sequential)", len(chunk_files))
            for idx, npz_path in enumerate(chunk_files, start=1):
                yield self._map_pattern_count_entry(npz_path, self.feature_columns)
                if idx % 10 == 0:
                    logger.info("[run][stage 2] mapped %d/%d chunks", idx, len(chunk_files))
        else:
            from joblib import Parallel, delayed
            logger.info("[run][stage 2] mapping pattern counts: %d chunks, %d workers", len(chunk_files), num_workers)
            results = Parallel(n_jobs=num_workers, backend="loky", prefer="processes", verbose=0)(
                delayed(self._map_pattern_count_entry)(npz_path, self.feature_columns)
                for npz_path in chunk_files
            )
            for idx, result in enumerate(results, start=1):
                yield result
                if idx % 10 == 0:
                    logger.info("[run][stage 2] mapped %d/%d chunks", idx, len(chunk_files))
        logger.info("[run][stage 2] mapping completed: %d chunks", len(chunk_files))

    def _cache_path(self, stage: str) -> str:
        return os.path.join(self.cache_dir, f"_stage{stage}_cache.pkl")

    def _load_cache(self, stage: str) -> Optional[dict]:
        import pickle
        cache_path = self._cache_path(stage)
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.info("[run][stage %s] loaded cache", stage)
            return data
        except Exception as e:
            logger.warning("[run][stage %s] cache load failed: %s, will recompute", stage, e)
            return None

    def _save_cache(self, stage: str, data: dict):
        import pickle
        cache_path = self._cache_path(stage)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            size_mb = os.path.getsize(cache_path) / 1024**2
            logger.info("[run][stage %s] saved cache (%.1f MB)", stage, size_mb)
        except Exception as e:
            logger.warning("[run][stage %s] cache save failed: %s", stage, e)

    def _clear_later_caches(self, from_stage: int):
        for s in range(from_stage + 1, 4):
            path = self._cache_path(str(s))
            if os.path.exists(path):
                os.remove(path)
                logger.info("[run] cleared stage %d cache (dependencies changed)", s)

    def _filter_chunk_by_train_years(self, df_chunk_samples: pd.DataFrame) -> pd.DataFrame:
        if self.train_end_year is None:
            return df_chunk_samples
        if "year" not in df_chunk_samples.columns:
            return df_chunk_samples
        n_before = len(df_chunk_samples)
        df_filtered = df_chunk_samples[df_chunk_samples["year"] <= self.train_end_year].copy()
        n_dropped = n_before - len(df_filtered)
        if n_dropped > 0:
            logger.debug("Filtered %d rows beyond train_end_year=%d (remaining: %d)",
                         n_dropped, self.train_end_year, len(df_filtered))
        return df_filtered

    def _resolve_validation_years(self) -> list[int]:
        if self.validation_years is not None:
            return sorted(self.validation_years)
        if self.train_end_year is None:
            return []
        val_years = []
        for npz_path in self._iter_chunk_files():
            try:
                data = np.load(npz_path, allow_pickle=True)
                if "year" in data.files:
                    years = np.unique(data["year"][data["year"] > self.train_end_year])
                    val_years.extend(int(y) for y in years)
                data.close()
            except Exception:
                continue
        return sorted(set(val_years))

    def _run_temporal_validation(self) -> Optional[dict]:
        if self.train_end_year is None and self.validation_years is None:
            logger.info("[validation] no temporal split configured — skipping validation")
            return None

        val_years = self._resolve_validation_years()
        if not val_years:
            logger.warning("[validation] no validation years found after train_end_year=%s", self.train_end_year)
            return None

        logger.info("[validation] running temporal validation on years: %s", val_years)

        all_val_predictions = []
        all_val_labels = []
        all_val_classes = []

        for npz_path in self._iter_chunk_files():
            try:
                data = np.load(npz_path, allow_pickle=True)
                if "year" not in data.files:
                    data.close()
                    continue
                years = data["year"]
                val_mask = np.isin(years, val_years)
                if not val_mask.any():
                    data.close()
                    continue
                selected_indices = np.where(val_mask)[0]
                df_chunk = self._load_chunk_dataframe_from_data(data, row_indices=selected_indices)
                df_chunk_samples = self._build_chunk_training_samples(df_chunk, skip_sampling=True)
                df_val = df_chunk_samples[df_chunk_samples["year"].isin(val_years)].copy()
                data.close()
                if df_val.empty:
                    continue

                for col in self.feature_columns:
                    if col not in df_val.columns:
                        df_val[col] = np.nan
                df_clean = df_val.dropna(subset=[c for c in self.feature_columns if c in self.static_features + self.direct_features])
                if df_clean.empty:
                    continue

                preds = self.predict_dataframe(df_clean)
                if "disturbance_probability" in preds.columns:
                    all_val_predictions.append(preds["disturbance_probability"].values)
                if self.target in df_clean.columns:
                    all_val_labels.append(df_clean[self.target].values)
                if "disturbance_class" in df_clean.columns:
                    all_val_classes.append(df_clean["disturbance_class"].values)
            except Exception as exc:
                logger.warning("[validation] error processing chunk %s: %s", npz_path, exc)
                continue

        if not all_val_labels:
            logger.warning("[validation] no validation data collected")
            return None

        y_true = np.concatenate(all_val_labels)
        y_pred_prob = np.concatenate(all_val_predictions)
        y_pred_class = np.concatenate(all_val_classes) if all_val_classes else None

        from inducing_factor.core.validation import check_validation_gate, compute_validation_metrics

        metrics = compute_validation_metrics(
            y_true, y_pred_prob, y_pred_class,
            class_labels=self.class_engine.class_labels if hasattr(self.class_engine, "class_labels") else None,
        )

        passed, reason = check_validation_gate(metrics, self.validation_metric_thresholds)
        metrics["gate_passed"] = passed
        metrics["gate_reason"] = reason
        metrics["train_end_year"] = self.train_end_year
        metrics["validation_years"] = val_years

        report_path = os.path.join(self.out_dir, "validation_report.json")
        import json
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        logger.info("[validation] report saved: %s (passed=%s)", report_path, passed)

        if not passed and self.validation_required:
            logger.warning("[validation] GATE FAILED: %s", reason)
        else:
            logger.info("[validation] %s", reason)

        return metrics

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

        cached = self._load_cache("1")
        if cached is not None:
            feature_columns = cached["feature_columns"]
            discretizer_sample = cached["discretizer_sample"]
            event_sample_count = cached["event_sample_count"]
            background_sample_count = cached["background_sample_count"]
            logger.info("[run][stage 1] skipping scan & fit discretizer (cached)")
        else:
            logger.info("[run][stage 1] starting scan & fit discretizer with %d chunks", len(chunk_files))
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
            logger.info("[run][stage 1] completed: %d events, %d background samples", event_sample_count, background_sample_count)
            progress.finish_stage({"events": event_sample_count, "background": background_sample_count})
            self._save_cache("1", {
                "feature_columns": feature_columns,
                "discretizer_sample": discretizer_sample,
                "event_sample_count": event_sample_count,
                "background_sample_count": background_sample_count,
            })
            self._clear_later_caches(1)

        if discretizer_sample is None or discretizer_sample.empty:
            raise ValueError("No valid samples could be built from chunk files")

        self.feature_columns = feature_columns
        for col in self.feature_columns:
            if col not in discretizer_sample.columns:
                discretizer_sample[col] = np.nan
        valid_feature_mask = discretizer_sample[self.feature_columns].notna().mean() > 0.3
        dropped_features = [c for c in self.feature_columns if c in valid_feature_mask.index and not valid_feature_mask[c]]
        if dropped_features:
            logger.warning("[run] dropping high-NaN features: %s", dropped_features)
            self.feature_columns = [c for c in self.feature_columns if c not in dropped_features]
        discretizer_sample = discretizer_sample.dropna(subset=[c for c in self.feature_columns if c in self.static_features + self.direct_features])
        logger.info("[run] fitting discretizer on %d rows, %d features: %s", len(discretizer_sample), len(self.feature_columns), self.feature_columns)
        self.discretizer.fit(discretizer_sample[self.feature_columns], discretizer_sample[self.target])

        progress.start_stage(2, stage_total, "pattern counts & model samples", total=len(chunk_files))
        event_counts: Counter = Counter()
        background_counts: Counter = Counter()
        event_counts_by_class: Dict[int, Counter] = {}
        n_event_by_class: Dict[int, int] = {}
        n_event = 0
        n_bg = 0
        event_label_sample: Optional[pd.DataFrame] = None
        background_label_sample: Optional[pd.DataFrame] = None
        event_training_sample: Optional[pd.DataFrame] = None
        background_training_sample: Optional[pd.DataFrame] = None

        cached2 = self._load_cache("2")
        if cached2 is not None:
            event_counts = cached2["event_counts"]
            background_counts = cached2["background_counts"]
            event_counts_by_class = cached2["event_counts_by_class"]
            n_event_by_class = cached2["n_event_by_class"]
            n_event = cached2["n_event"]
            n_bg = cached2["n_bg"]
            event_label_sample = cached2["event_label_sample"]
            background_label_sample = cached2["background_label_sample"]
            event_training_sample = cached2["event_training_sample"]
            background_training_sample = cached2["background_training_sample"]
            logger.info("[run][stage 2] skipping pattern counts (cached): %d events, %d bg, %d patterns",
                        n_event, n_bg, len(event_counts))
        else:
            event_limit = max(1, self.model_sample_limit // 2)
            background_limit = max(1, self.model_sample_limit - event_limit)

            stage2_workers = self._resolve_run_worker_count()
            pattern_mapped = self._map_pattern_counts(chunk_files, num_workers=stage2_workers)
            for idx, chunk in enumerate(pattern_mapped, start=1):
                event_counts.update(chunk["event_counts"])
                background_counts.update(chunk["background_counts"])
                n_event += int(chunk["n_event"])
                n_bg += int(chunk["n_bg"])
                chunk_n_by_class = chunk.get("n_event_by_class", {})
                chunk_counts_by_class = chunk.get("event_counts_by_class", {})
                for cls_id, cls_n in chunk_n_by_class.items():
                    n_event_by_class[cls_id] = n_event_by_class.get(cls_id, 0) + int(cls_n)
                for cls_id, cls_counter in chunk_counts_by_class.items():
                    if cls_id not in event_counts_by_class:
                        event_counts_by_class[cls_id] = Counter()
                    event_counts_by_class[cls_id].update(cls_counter)
                event_label_sample = self._update_sample_frame(event_label_sample, chunk["event_label_sample"], self.factor_miner.max_bootstrap_samples)
                background_label_sample = self._update_sample_frame(background_label_sample, chunk["background_label_sample"], self.factor_miner.max_bootstrap_samples)
                event_training_sample = self._update_sample_frame(event_training_sample, chunk["event_training_sample"], event_limit)
                background_training_sample = self._update_sample_frame(background_training_sample, chunk["background_training_sample"], background_limit)
                progress.advance(
                    1,
                    {
                        "chunk": idx,
                        "event_patterns": len(event_counts),
                        "events": n_event,
                        "model_event": 0 if event_training_sample is None else len(event_training_sample),
                        "model_bg": 0 if background_training_sample is None else len(background_training_sample),
                    },
                )
            self._save_cache("2", {
                "event_counts": event_counts,
                "background_counts": background_counts,
                "event_counts_by_class": event_counts_by_class,
                "n_event_by_class": n_event_by_class,
                "n_event": n_event,
                "n_bg": n_bg,
                "event_label_sample": event_label_sample,
                "background_label_sample": background_label_sample,
                "event_training_sample": event_training_sample,
                "background_training_sample": background_training_sample,
            })
            self._clear_later_caches(2)

        progress.finish_stage({"event_patterns": len(event_counts), "events": n_event, "background": n_bg})

        pattern_candidates = self.factor_miner.mine_pattern_candidates_from_counts(event_counts, background_counts, n_event, n_bg)
        event_transactions = self.factor_miner.transactions_from_labels(event_label_sample) if event_label_sample is not None else []
        background_transactions = self.factor_miner.transactions_from_labels(background_label_sample) if background_label_sample is not None else []
        pattern_stability = self.factor_miner.estimate_pattern_stability(event_transactions, background_transactions)
        reduced_patterns = self.factor_miner.reduce_redundant_patterns(pattern_candidates, pattern_stability)
        self.factor_library = self.factor_miner.build_factor_library(reduced_patterns, n_event=n_event, n_background=n_bg)
        if getattr(self, "auto_select_factors", False) and len(self.factor_library) > 1:
            from sklearn.linear_model import LogisticRegression as _LR
            best_bic = float("inf")
            best_k = 1
            factor_label_df = self.discretizer.transform(event_label_sample.append(background_label_sample, ignore_index=True) if background_label_sample is not None else event_label_sample, as_labels=True)
            factor_target = pd.Series([1] * len(event_label_sample) + ([0] * len(background_label_sample) if background_label_sample is not None else []))
            for k in range(1, min(len(self.factor_library), 10) + 1):
                candidate_factors = self.factor_library[:k]
                X = np.column_stack([self.attribution_engine.factor_activation(factor_label_df, f).to_numpy() for f in candidate_factors])
                y = factor_target.to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                lr = _LR(solver="liblinear", max_iter=1000, random_state=self.random_state)
                lr.fit(X, y)
                ll = -lr.predict_log_proba(X)[np.arange(len(y)), y].sum()
                n_params = k + 1
                bic = -2 * ll + n_params * np.log(len(y))
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            self.selected_factors = self.factor_library[:best_k]
            logger.info("Auto-selected %d factors by BIC (best_bic=%.1f)", best_k, best_bic)
        else:
            self.selected_factors = self.factor_library[: self.max_network_factors]
        if not self.factor_library:
            raise ValueError("No stable inducing factors were discovered. Consider relaxing thresholds.")
        df_factors = self.factor_miner.to_frame(self.factor_library)
        self.artifacts.save_factor_library(self.factor_library, df_factors, self.discretizer)

        sampled_frames = [frame for frame in [event_training_sample, background_training_sample] if frame is not None and not frame.empty]
        if not sampled_frames:
            raise ValueError("No model training samples were collected during streaming run")
        training_samples = pd.concat(sampled_frames, ignore_index=True)
        training_samples = training_samples.dropna(subset=[c for c in self.feature_columns if c in self.static_features + self.direct_features])
        self.training_samples = training_samples
        self.artifacts.save_training_samples(training_samples)

        progress.start_stage(3, stage_total, "fit models and export artifacts", total=4)
        label_df = self.discretizer.transform(training_samples[self.feature_columns], as_labels=True)
        logger.info("[run][stage 3] label_df shape=%s, feature_columns=%s, discretizer.features_=%s", label_df.shape, self.feature_columns, self.discretizer.features_)
        for c in label_df.columns:
            logger.info("[run][stage 3] label_df[%s]: %d unique, sample=%s", c, label_df[c].nunique(), label_df[c].value_counts().head(2).to_dict())
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
                "class_model_status": self.class_engine.model_status,
                "target": self.target,
                "streaming_run": True,
                "discretizer_sample_limit": self.discretizer_sample_limit,
                "model_sample_limit": self.model_sample_limit,
            }
        )
        attribution = self.attribute_dataframe(training_samples)
        progress.advance(1, {"preview_rows": len(attribution)})
        progress.finish_stage({"factors": len(self.factor_library), "preview_rows": len(attribution)})

        if getattr(self, "analyze_by_class", False):
            logger.info("[class-conditional] computing P(class|factor) for all %d factors", len(self.factor_library))
            self._analyze_by_class(
                inducing_factors=self.factor_library,
                label_df=training_samples,
                feature_columns=self.feature_columns,
                out_dir=self.out_dir,
                event_counts_by_class=event_counts_by_class,
                n_event_by_class=n_event_by_class,
                background_counts=background_counts,
                n_event=n_event,
                n_bg=n_bg,
            )

        validation_report = self._run_temporal_validation()

        return {
            "training_samples": training_samples,
            "factors": df_factors,
            "factor_class_summary": self.class_summary,
            "attribution_preview": attribution,
            "validation_report": validation_report,
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

    def _load_chunk_dataframe_from_data(self, data, row_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        import numpy as np
        if row_indices is None:
            return pd.DataFrame({col: data[col] for col in data.files})
        result = {}
        for col in data.files:
            arr = data[col]
            if len(arr) == 0:
                result[col] = arr
            else:
                result[col] = arr[row_indices]
        return pd.DataFrame(result)

    def _sample_chunk_indices_from_data(self, data) -> np.ndarray:
        import numpy as np
        import time
        try:
            if "disturbance_time_value" in data.files:
                dist_col = "disturbance_time_value"
            elif "disturbance_type" in data.files:
                dist_col = "disturbance_type"
            else:
                raise KeyError("Input chunk is missing required disturbance time column")

            disturbance = data[dist_col]
            point_id = data["point_id"]
            n_total = len(disturbance)
            logger.debug("_sample_chunk_indices: loaded %d rows", n_total)

            event_mask = disturbance > 0
            event_indices = np.where(event_mask)[0]

            background_mask = ~event_mask
            if "background_sample_keep" in data.files:
                background_keep_mask = data["background_sample_keep"].astype(bool)
                background_mask = background_mask & background_keep_mask

            if not background_mask.any():
                return event_indices

            t0 = time.time()
            bg_point_ids = np.unique(point_id[background_mask])
            if len(bg_point_ids) == 0:
                return event_indices

            sample_n = max(1, int(len(bg_point_ids) * self.background_sampler.background_sample_rate))
            sample_n = min(sample_n, len(bg_point_ids))

            rng = np.random.default_rng(self.random_state)
            strat_feature = None
            if "x" in data.files and "y" in data.files:
                strat_feature = "elevation" if "elevation" in data.files else None
                if strat_feature is None and "slope" in data.files:
                    strat_feature = "slope"
            if strat_feature is not None:
                try:
                    strat_data = data[strat_feature][background_mask]
                    valid_mask = ~np.isnan(strat_data) & (strat_data != -9999)
                    if valid_mask.sum() > 0:
                        strat_vals = strat_data[valid_mask]
                        p33 = np.percentile(strat_vals, 33)
                        p66 = np.percentile(strat_vals, 66)
                        n_per = max(1, sample_n // 3)
                        sampled_parts = []
                        for pid_group in [
                            np.unique(point_id[background_mask & (strat_data <= p33)]),
                            np.unique(point_id[background_mask & (strat_data > p33) & (strat_data <= p66)]),
                            np.unique(point_id[background_mask & (strat_data > p66)]),
                        ]:
                            if len(pid_group) > 0:
                                n_take = min(n_per, len(pid_group))
                                sampled_parts.append(rng.choice(pid_group, size=n_take, replace=False))
                        if sampled_parts:
                            sampled_bg_point_ids = np.concatenate(sampled_parts)
                        else:
                            sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)
                    else:
                        sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)
                except Exception:
                    sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)
            else:
                sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)

            bg_selected_mask = np.isin(point_id, sampled_bg_point_ids)
            bg_selected_indices = np.where(bg_selected_mask & background_mask)[0]

            if self.deduplicate_temporal and len(bg_selected_indices) > 0:
                bg_pids_selected = point_id[bg_selected_indices]
                rng_dedup = np.random.default_rng(self.random_state + 1)
                unique_pids = np.unique(bg_pids_selected)
                keep_indices = []
                for pid in unique_pids:
                    pid_rows = bg_selected_indices[bg_pids_selected == pid]
                    keep_indices.append(rng_dedup.choice(pid_rows))
                bg_selected_indices = np.array(keep_indices, dtype=bg_selected_indices.dtype)

            selected_indices = np.concatenate([event_indices, bg_selected_indices])
            logger.debug("  total selected rows: %d (%.2f%%)", len(selected_indices), len(selected_indices)/n_total*100)
            return np.sort(selected_indices)
        except Exception:
            return np.array([], dtype=np.int64)

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
            if "background_sample_keep" in data.files:
                background_keep_mask = data["background_sample_keep"].astype(bool)
                background_mask = background_mask & background_keep_mask
            
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
            
            # Sample background point_ids with lightweight stratification by elevation/slope if available
            sample_n = max(1, int(len(bg_point_ids) * self.background_sampler.background_sample_rate))
            sample_n = min(sample_n, len(bg_point_ids))
            
            rng = np.random.default_rng(self.random_state)
            strat_feature = None
            strat_values = None
            if "x" in data.files and "y" in data.files:
                strat_feature = "elevation" if "elevation" in data.files else None
                if strat_feature is None and "slope" in data.files:
                    strat_feature = "slope"
            if strat_feature is not None:
                try:
                    strat_data = data[strat_feature][background_mask]
                    valid_mask = ~np.isnan(strat_data) & (strat_data != -9999)
                    if valid_mask.sum() > 0:
                        strat_vals = strat_data[valid_mask]
                        p33 = np.percentile(strat_vals, 33)
                        p66 = np.percentile(strat_vals, 66)
                        bg_strat_ids = bg_point_ids  # fallback
                        low_pids = np.unique(point_id[background_mask & (strat_data <= p33)])
                        mid_pids = np.unique(point_id[background_mask & (strat_data > p33) & (strat_data <= p66)])
                        high_pids = np.unique(point_id[background_mask & (strat_data > p66)])
                        n_per = max(1, sample_n // 3)
                        sampled_parts = []
                        for pid_group in [low_pids, mid_pids, high_pids]:
                            if len(pid_group) > 0:
                                n_take = min(n_per, len(pid_group))
                                sampled_parts.append(rng.choice(pid_group, size=n_take, replace=False))
                        if sampled_parts:
                            sampled_bg_point_ids = np.concatenate(sampled_parts)
                        else:
                            sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)
                        logger.debug("  stratified sampled background point_ids: %d by %s", len(sampled_bg_point_ids), strat_feature)
                    else:
                        sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)
                except Exception:
                    sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)
                    logger.debug("  stratification failed, using simple random sampling")
            else:
                sampled_bg_point_ids = rng.choice(bg_point_ids, size=sample_n, replace=False)
                logger.debug("  sampled background point_ids: %d (no stratification feature)", sample_n)
            
            # Find rows where point_id is in sampled_bg_point_ids
            # Use boolean mask for memory efficiency
            t0 = time.time()
            bg_selected_mask = np.isin(point_id, sampled_bg_point_ids)
            bg_selected_indices = np.where(bg_selected_mask & background_mask)[0]
            t_isin = time.time() - t0
            logger.debug("  np.isin selected %d background rows (took %.1fs)", len(bg_selected_indices), t_isin)
            
            # Deduplicate: keep only 1 year per background point_id to avoid temporal pseudoreplication
            if self.deduplicate_temporal and len(bg_selected_indices) > 0:
                bg_pids_selected = point_id[bg_selected_indices]
                rng_dedup = np.random.default_rng(self.random_state + 1)
                unique_pids = np.unique(bg_pids_selected)
                keep_indices = []
                for pid in unique_pids:
                    pid_rows = bg_selected_indices[bg_pids_selected == pid]
                    keep_indices.append(rng_dedup.choice(pid_rows))
                bg_selected_indices = np.array(keep_indices, dtype=bg_selected_indices.dtype)
                logger.debug("  deduplicated: %d -> %d background rows", len(bg_pids_selected), len(bg_selected_indices))
            
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

    def _build_prediction_features(
        self,
        df_chunk: pd.DataFrame,
        predict_month: int,
    ) -> tuple[pd.DataFrame, List[str]]:
        """Build features for prediction mode without requiring CCDC columns.

        Unlike _build_chunk_training_samples, this method does NOT require
        disturbance_time_value, disturbance_type, disturbance_month, or
        reference_month. Instead, it passes an explicit predict_month to the
        feature builder for leakage-free rainfall window computation.

        Returns (features_df, required_cols) matching the shape expected by
        downstream callers.
        """
        df = df_chunk.sort_values(["point_id", "year"]).copy()
        if self.target not in df.columns:
            df[self.target] = 0
        return self.feature_builder.build_chunk_features(df, predict_month=predict_month)

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
            core_cols = [c for c in required_cols if c in self.static_features or c in self.direct_features]
            history_mask = df[core_cols].notna().all(axis=1)
            metadata_columns = [col for col in self.metadata_columns if col in df.columns]
            df_ready = df.loc[history_mask, metadata_columns + required_cols].copy()
            return df_ready.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=core_cols)
        
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
        
        # 6. Filter rows with complete feature history (core features only)
        core_cols = [c for c in required_cols if c in self.static_features or c in self.direct_features]
        history_mask = df[core_cols].notna().all(axis=1)
        metadata_columns = [col for col in self.metadata_columns if col in df.columns]
        df_ready = df.loc[history_mask, metadata_columns + required_cols].copy()
        
        if df_ready.empty:
            return df_ready
        
        # 7. Final cleanup
        df_ready = df_ready.replace([float("inf"), float("-inf")], pd.NA)
        return df_ready.dropna(subset=core_cols)

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
        core_feature_cols = [c for c in self.feature_columns if c in self.static_features + self.direct_features]
        df_samples = df_samples.dropna(subset=core_feature_cols)
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
        self.factor_library = self.factor_miner.build_factor_library(reduced_patterns, n_event=len(event_transactions), n_background=len(background_transactions))
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
        if self.class_engine.is_available():
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

    def predict_ensemble(
        self, samples: pd.DataFrame, year: int, n_jobs: int = 1,
    ) -> pd.DataFrame:
        """12-month ensemble prediction.

        For each month 1-12, builds rainfall window features aligned to that
        month, runs the model, and collects per-point probabilities.  Then
        aggregates into mean/max/peak_month columns.

        Args:
            samples: Multi-year DataFrame (lookback window + target year) with
                point_id, year, and all required covariates.
            year: Target prediction year (filters rows to ``samples[samples["year"] == year]``).
            n_jobs: Not yet wired; reserved for future parallelization.

        Returns:
            DataFrame with point_id, x, y, raster_row, raster_col,
            disturbance_probability_mean, disturbance_probability_max,
            disturbance_probability_peak_month, and class probability columns
            (mean across months).
        """
        month_probs: List[pd.DataFrame] = []
        for month in range(1, 13):
            df, _required_cols = self._build_prediction_features(samples, predict_month=month)
            df_year = df[df["year"] == year].copy()
            if df_year.empty:
                continue
            for col in self.feature_columns:
                if col not in df_year.columns:
                    df_year[col] = np.nan

            drop_mask = df_year[self.feature_columns].isna().any(axis=1)
            if drop_mask.all():
                continue
            pred = self.predict_dataframe(df_year[~drop_mask].copy(), predict_month=month)
            if pred.empty:
                continue
            id_cols = [c for c in ["point_id", "x", "y", "raster_row", "raster_col"] if c in pred.columns]
            keep = id_cols + ["disturbance_probability"]
            prob = pred[keep].copy()
            prob["month"] = month
            month_probs.append(prob)

        if not month_probs:
            return pd.DataFrame()

        all_probs = pd.concat(month_probs, ignore_index=True)
        grouped = all_probs.groupby([c for c in id_cols if c in all_probs.columns])
        result = grouped["disturbance_probability"].agg(["mean", "max"]).rename(
            columns={"mean": "disturbance_probability_mean", "max": "disturbance_probability_max"},
        ).reset_index()
        peak_month = all_probs.loc[grouped["disturbance_probability"].idxmax().dropna().astype(int)]
        result["disturbance_probability_peak_month"] = peak_month.set_index(id_cols)["month"].reindex(
            pd.MultiIndex.from_frame(result[id_cols]) if len(id_cols) > 1 else result.index,
        ).values
        # For backward compatibility, set legacy column = mean
        result["disturbance_probability"] = result["disturbance_probability_mean"]
        return result

    def predict_dataframe(self, samples: pd.DataFrame, predict_month: Optional[int] = None) -> pd.DataFrame:
        label_df = self.discretizer.transform(samples[self.feature_columns], as_labels=True)
        occurrence_probs = self.occurrence_engine.predict_proba(label_df, self.selected_factors)
        result = samples[[col for col in ["point_id", "raster_row", "raster_col", "x", "y", "year"] if col in samples.columns]].copy()
        if occurrence_probs.size:
            result["disturbance_probability"] = occurrence_probs[:, 1]
        else:
            result["disturbance_probability"] = 0.0
        if not self.class_engine.is_available():
            logger.debug("predict_dataframe: class model unavailable, skipping dominant_disturbance_class and class probabilities.")
            return result
        class_probs = self.class_engine.predict_proba(label_df, self.selected_factors)
        if not class_probs.empty:
            result = pd.concat([result.reset_index(drop=True), class_probs.reset_index(drop=True)], axis=1)
            prob_cols = [col for col in result.columns if col.startswith("class_probability_")]
            if prob_cols:
                result["dominant_disturbance_class"] = result[prob_cols].idxmax(axis=1).str.rsplit("_", n=1).str[-1].astype(int)
                _abs_col_short = {1: "fire", 2: "logging", 3: "geo_hazard", 4: "road"}
                for col in prob_cols:
                    cls = int(col.rsplit("_", 1)[-1])
                    if cls in _abs_col_short:
                        result[f"prob_{_abs_col_short[cls]}_abs"] = result[col] * result["disturbance_probability"]
        return result

    def _analyze_by_class(
        self,
        inducing_factors: List[InducingFactor],
        label_df: pd.DataFrame,
        feature_columns: List[str],
        out_dir: str,
        event_counts_by_class: Optional[Dict[int, Counter]] = None,
        n_event_by_class: Optional[Dict[int, int]] = None,
        background_counts: Optional[Counter] = None,
        n_event: int = 0,
        n_bg: int = 0,
    ):
        analyzer = FactorClassAnalyzer(min_count_per_class=self.min_count_per_class)
        if event_counts_by_class and n_event_by_class:
            profiles = analyzer.compute_class_conditional_from_counts(
                factors=inducing_factors,
                event_counts_by_class=event_counts_by_class,
                background_counts=background_counts or Counter(),
                n_event_by_class=n_event_by_class,
                n_event=n_event,
                n_bg=n_bg,
            )
        else:
            profiles = analyzer.compute_class_conditional(
                factors=inducing_factors,
                label_df=label_df,
                feature_columns=feature_columns,
                discretizer=self.discretizer,
            )
        top_factors = analyzer.find_top_class_specific_factors(
            profiles,
            min_prior_lift=self.class_prior_lift_threshold,
            top_n=self.top_n_per_class,
            min_count=self.min_count_per_class,
        )
        class_priors = {}
        if event_counts_by_class and n_event_by_class:
            n_labeled_total = sum(n_event_by_class.values())
            for cls, cls_name in FactorClassAnalyzer.CLASS_LABELS.items():
                class_priors[cls_name] = n_event_by_class.get(cls, 0) / max(n_labeled_total, 1)
        else:
            class_column = "disturbance_class"
            target = self.target
            events_mask = label_df[target] == 1
            labeled_events = label_df[events_mask & (label_df[class_column] > 0)]
            n_labeled = len(labeled_events)
            for cls, cls_name in FactorClassAnalyzer.CLASS_LABELS.items():
                class_priors[cls_name] = len(labeled_events[labeled_events[class_column] == cls]) / max(n_labeled, 1)
        self._save_class_conditional_results(profiles, top_factors, class_priors, out_dir, self.discretizer)

    def _save_class_conditional_results(self, profiles, top_factors, class_priors, out_dir, discretizer=None):
        from dataclasses import asdict

        os.makedirs(out_dir, exist_ok=True)

        all_profiles = [asdict(p) for p in profiles]
        json_path = os.path.join(out_dir, "class_conditional_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, indent=2, ensure_ascii=False)
        logger.info("[class-conditional] saved %d profiles to %s", len(profiles), json_path)

        df = pd.DataFrame([asdict(p) for p in profiles])
        csv_path = os.path.join(out_dir, "class_conditional_factors.csv")
        df.to_csv(csv_path, index=False)
        logger.info("[class-conditional] saved CSV to %s", csv_path)

        self._write_class_interpretation(top_factors, class_priors, out_dir, discretizer)

    def _write_class_interpretation(self, top_factors, class_priors, out_dir, discretizer=None):
        def _readable_item(item: str) -> str:
            if "=" not in item:
                return item
            feature, label = item.split("=", 1)
            base = feature.replace("_prev_last", "").replace("_prev_mean", "").replace("_prev_max", "").replace("_prev_", "_").replace("dist_to_", "").replace("rainfall_prev_", "rainfall_").replace("landuse_prev_last", "landuse")
            if discretizer is not None and feature in discretizer.feature_meta_:
                meta = discretizer.feature_meta_[feature]
                if meta["kind"] == "continuous" and "bins" in meta and "labels" in meta:
                    edges = meta["bins"]
                    labels = meta["labels"]
                    try:
                        idx = labels.index(label)
                        lo = edges[idx] if idx < len(edges) else -float("inf")
                        hi = edges[idx + 1] if idx + 1 < len(edges) else float("inf")
                        if lo == -float("inf"):
                            return f"{base} < {hi:.1f}"
                        elif hi == float("inf"):
                            return f"{base} > {lo:.1f}"
                        else:
                            return f"{base} [{lo:.1f}, {hi:.1f})"
                    except (ValueError, IndexError):
                        pass
            return f"{base}={label}"

        def readable_name(factor) -> str:
            parts = [_readable_item(item) for item in factor.factor_items]
            return " + ".join(parts)

        class_display = {
            "fire": ("Fire", "prob_fire_given_factor", "prior_lift_fire", "count_in_fire"),
            "logging": ("Logging", "prob_logging_given_factor", "prior_lift_logging", "count_in_logging"),
            "geo_hazard": ("Geo-hazard", "prob_geo_hazard_given_factor", "prior_lift_geo", "count_in_geo_hazard"),
            "road_construction": ("Road construction", "prob_road_given_factor", "prior_lift_road", "count_in_road_construction"),
        }
        lines = ["# Inducing Factor Class-Conditional Analysis\n"]

        non_empty = [v for v in top_factors.values() if v]
        total_labeled = getattr(non_empty[0][0], 'total_labeled', 'N/A') if non_empty else 'N/A'
        lines.append(f"\n**Total labeled events**: {total_labeled}\n")

        for class_name, factors in top_factors.items():
            display_name, prob_field, lift_field, count_field = class_display[class_name]
            prior_prob = class_priors.get(class_name, 0.0)

            lines.append(f"\n## {display_name}\n")
            lines.append(f"Prior probability P({class_name}) = {prior_prob:.4f}\n")
            if not factors:
                lines.append("No factors with prior lift above threshold.\n")
                continue
            lines.append("| Factor combination | P(" + class_name + "\\|F) | Prior P(" + class_name + ") | Lift | Count | Total labeled |")
            lines.append("|-------------------|------------------|-------------------|------|-------|---------------|")
            for f in factors:
                prob = getattr(f, prob_field)
                lift = getattr(f, lift_field)
                count = getattr(f, count_field)
                name = readable_name(f)
                lines.append(
                    f"| {name[:80]} | {prob:.3f} | {prior_prob:.4f} | {lift:.1f}x | {count} | {f.total_labeled_with_factor} |"
                )

        md_path = os.path.join(out_dir, "class_factor_interpretation.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("[class-conditional] saved interpretation to %s", md_path)

    def run(self) -> dict:
        return self._streaming_run()

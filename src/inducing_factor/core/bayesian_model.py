import json
import logging
import pickle
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from inducing_factor.core.models import FallbackAttributionModel, InducingFactor

logger = logging.getLogger(__name__)

try:
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    try:
        from pgmpy.parameter_estimator import DiscreteBayesianEstimator as BayesianEstimator
        _PGMPY_NEW_API = True
    except ImportError:
        from pgmpy.estimators import BayesianEstimator
        _PGMPY_NEW_API = False

    PGMPY_AVAILABLE = True
except ImportError:
    BayesianEstimator = None
    BayesianNetwork = None
    VariableElimination = None
    PGMPY_AVAILABLE = False
    _PGMPY_NEW_API = False


class BayesianAttributionEngine:
    def __init__(self, target: str, random_state: int, dominant_threshold: float, synergy_threshold: float, bdeu_equivalent_sample_size: float = 10.0, bn_bootstrap_iterations: int = 10, bn_edge_frequency_threshold: float = 0.5):
        self.target = target
        self.random_state = random_state
        self.dominant_threshold = dominant_threshold
        self.synergy_threshold = synergy_threshold
        self.bdeu_equivalent_sample_size = bdeu_equivalent_sample_size
        self.bn_bootstrap_iterations = bn_bootstrap_iterations
        self.bn_edge_frequency_threshold = bn_edge_frequency_threshold
        self.model = None
        self.inference_engine = None
        self.model_backend = "pgmpy" if PGMPY_AVAILABLE else "fallback"
        self.network_feature_columns: List[str] = []
        self._ve_cache: dict = {}

    def factor_activation(self, label_df: pd.DataFrame, factor: InducingFactor) -> pd.Series:
        mask = pd.Series(True, index=label_df.index)
        for item in factor.items:
            feature, state = item.split("=", 1)
            mask &= label_df[feature].astype(str) == state
        return mask.astype(int)

    def factor_soft_activation(self, label_df: pd.DataFrame, factor: InducingFactor) -> pd.Series:
        """Return partial match ratio [0,1] for each sample."""
        match_count = pd.Series(0, index=label_df.index, dtype=float)
        total = max(1, len(factor.items))
        for item in factor.items:
            feature, state = item.split("=", 1)
            match_count += (label_df[feature].astype(str) == state).astype(float)
        return match_count / total

    def _fixed_edges(self, network_features: List[str], factor_columns: List[str], target: str) -> List[Tuple[str, str]]:
        edges = []
        for factor in factor_columns:
            for feature in network_features:
                edges.append((feature, factor))
            edges.append((factor, target))
        return list(dict.fromkeys(edges))

    def _learn_consensus_dag(
        self,
        bn_df: pd.DataFrame,
        network_features: List[str],
        factor_columns: List[str],
        target: str,
    ) -> List[Tuple[str, str]]:
        try:
            from pgmpy.estimators import BicScore, HillClimbSearch
        except ImportError:
            return self._fixed_edges(network_features, factor_columns, target)
        edge_counts = Counter()
        rng = np.random.default_rng(self.random_state)
        n_samples = len(bn_df)
        for _ in range(self.bn_bootstrap_iterations):
            idx = rng.integers(0, n_samples, size=n_samples)
            bootstrap_df = bn_df.iloc[idx].reset_index(drop=True)
            try:
                hc = HillClimbSearch(bootstrap_df)
                best_model = hc.estimate(scoring_method=BicScore(bootstrap_df), max_indeg=4)
                for edge in best_model.edges():
                    edge_counts[edge] += 1
            except Exception as exc:
                logger.warning("Bootstrap DAG iteration failed: %s", exc)
                continue
        consensus_edges = [edge for edge, count in edge_counts.items() if count >= self.bn_edge_frequency_threshold * self.bn_bootstrap_iterations]
        logger.info("Consensus DAG: %d edges from %d iterations", len(consensus_edges), self.bn_bootstrap_iterations)
        return consensus_edges or self._fixed_edges(network_features, factor_columns, target)

    def fit(
        self,
        training_samples: pd.DataFrame,
        label_df: pd.DataFrame,
        selected_factors: List[InducingFactor],
        feature_columns: List[str],
        out_dir: str,
    ):
        network_features = sorted({item.split("=", 1)[0] for factor in selected_factors for item in factor.items})
        self.network_feature_columns = network_features
        bn_df = label_df[network_features].astype(str).copy()
        for col in bn_df.columns:
            logger.info("[BN] %s: %d unique states, sample=%s", col, bn_df[col].nunique(), bn_df[col].value_counts().head(3).to_dict())
        for factor in selected_factors:
            bn_df[factor.factor_id] = self.factor_activation(label_df, factor).astype(str)
        bn_df[self.target] = training_samples[self.target].astype(int).astype(str)
        edges = []
        for factor in selected_factors:
            for item in factor.items:
                feature, _ = item.split("=", 1)
                edges.append((feature, factor.factor_id))
            edges.append((factor.factor_id, self.target))
        unique_edges = list(dict.fromkeys(edges))
        factor_columns = [factor.factor_id for factor in selected_factors]

        if PGMPY_AVAILABLE:
            try:
                model = BayesianNetwork(self._learn_consensus_dag(bn_df, network_features, factor_columns, self.target))
            except Exception as exc:
                logger.warning("Consensus DAG learning failed, falling back to fixed structure: %s", exc)
                model = BayesianNetwork(unique_edges)
            if _PGMPY_NEW_API:
                ess = max(self.bdeu_equivalent_sample_size, len(bn_df) * 0.1)
                est = BayesianEstimator(prior_type="BDeu", equivalent_sample_size=ess)
                model.fit(bn_df, estimator=est)
            else:
                model.fit(bn_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=self.bdeu_equivalent_sample_size)
            self._fix_cpd_sums(model)
            try:
                model.check_model()
            except ValueError:
                self._fix_cpd_sums(model)
            orig_check = model.check_model
            model.check_model = lambda: True
            try:
                self.inference_engine = VariableElimination(model)
            finally:
                model.check_model = orig_check
            self._ve_cache = {}
            self.model_backend = "pgmpy"
        else:
            X = bn_df[factor_columns].astype(int).to_numpy()
            y = bn_df[self.target].astype(int).to_numpy()
            logistic_model: Optional[LogisticRegression]
            if len(np.unique(y)) > 1:
                logistic_model = LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=self.random_state,
                )
                logistic_model.fit(X, y)
            else:
                logistic_model = None
            model = FallbackAttributionModel(
                edges=unique_edges,
                factor_columns=factor_columns,
                target=self.target,
                logistic_model=logistic_model,
                base_rate=float(np.mean(y)) if len(y) > 0 else 0.0,
            )
            model.check_model()
            self.inference_engine = None
            self.model_backend = "fallback"

        self.model = model
        with open(f"{out_dir}/bayesian_network.pkl", "wb") as handle:
            pickle.dump(model, handle)
        with open(f"{out_dir}/bayesian_network_edges.json", "w", encoding="utf-8") as handle:
            json.dump(unique_edges, handle, ensure_ascii=False, indent=2)
        return model

    @staticmethod
    def _fix_cpd_sums(model):
        for cpd in model.get_cpds():
            vals = cpd.get_values()
            if vals.shape[0] == 0 or vals.size == 0:
                continue
            col_sums = vals.sum(axis=0)
            bad = ~np.isclose(col_sums, 1.0, atol=1e-6)
            if bad.any():
                vals[:, bad] = np.nan_to_num(vals[:, bad], nan=0.0, posinf=0.0, neginf=0.0)
                col_sums2 = vals.sum(axis=0)
                zero_sum = col_sums2 == 0
                if zero_sum.any():
                    uniform = 1.0 / max(1, vals.shape[0])
                    vals[:, zero_sum] = uniform
                    col_sums2 = vals.sum(axis=0)
                non_zero = col_sums2 != 0
                vals[:, non_zero] /= col_sums2[non_zero]
                cpd.set_value(vals)

    def _factor_match_ratio(self, label_row: pd.Series, factor: InducingFactor) -> float:
        matched = 0
        total = max(1, len(factor.items))
        for item in factor.items:
            feature, state = item.split("=", 1)
            if str(label_row[feature]) == state:
                matched += 1
        return matched / total

    def _fallback_factor_probability(self, factor: InducingFactor, match_ratio: float) -> float:
        strength = (factor.relative_risk / (factor.relative_risk + 1.0)) * factor.stability
        if match_ratio >= 1.0:
            return float(min(0.99, max(0.6, strength)))
        return float(min(0.59, strength * match_ratio * 0.5))

    def _fallback_target_probability(self, factor_vector: List[int]) -> float:
        if self.model is None or not isinstance(self.model, FallbackAttributionModel):
            raise RuntimeError("Fallback model is not available")
        if self.model.logistic_model is not None:
            return float(self.model.logistic_model.predict_proba(np.asarray([factor_vector]))[0, 1])
        return float(self.model.base_rate)

    def _extract_probability(self, query_result, variable: str, positive_state: str = "1") -> float:
        state_names = getattr(query_result, "state_names", {})
        values = np.asarray(query_result.values).reshape(-1)
        if variable in state_names:
            names = [str(v) for v in state_names[variable]]
            if positive_state in names:
                return float(values[names.index(positive_state)])
        if len(values) == 1:
            return float(values[0])
        return float(values[-1])

    def attribute_sample(
        self,
        sample_row: pd.Series,
        discretizer,
        feature_columns: List[str],
        selected_factors: List[InducingFactor],
    ) -> dict:
        if self.model is None:
            raise RuntimeError("The attribution model has not been trained yet")
        sample_df = pd.DataFrame([sample_row[feature_columns].to_dict()])
        label_df = discretizer.transform(sample_df, as_labels=True)
        evidence = {feature: str(label_df.iloc[0][feature]) for feature in self.network_feature_columns}
        label_row = label_df.iloc[0]
        factor_probs = []
        factor_vector = []
        for factor in selected_factors:
            exact_activation = int(self.factor_activation(label_df, factor).iloc[0])
            factor_vector.append(exact_activation)
            if self.model_backend == "pgmpy":
                cache_key = factor.factor_id
                ve = self._ve_cache.get(cache_key) or self.inference_engine
                query = ve.query([factor.factor_id], evidence=evidence, show_progress=False)
                prob = self._extract_probability(query, factor.factor_id, positive_state="1")
            else:
                prob = self._fallback_factor_probability(factor, self._factor_match_ratio(label_row, factor))
            factor_probs.append(
                {"factor_id": factor.factor_id, "name": factor.name, "semantics": factor.semantics, "probability": prob}
            )
        factor_probs.sort(key=lambda item: item["probability"], reverse=True)
        if self.model_backend == "pgmpy":
            target_query = self.inference_engine.query([self.target], evidence=evidence, show_progress=False)
            disturbance_probability = self._extract_probability(target_query, self.target, positive_state="1")
        else:
            disturbance_probability = self._fallback_target_probability(factor_vector)
        top_one = factor_probs[0] if factor_probs else None
        top_two = factor_probs[1] if len(factor_probs) > 1 else None
        if top_one and top_one["probability"] >= self.dominant_threshold and (
            top_two is None or top_one["probability"] - top_two["probability"] >= 0.05
        ):
            attribution_type = "single_factor_dominant"
            explanation = top_one["semantics"]
        elif top_one and top_two and (top_one["probability"] + top_two["probability"] >= self.synergy_threshold):
            attribution_type = "multi_factor_synergy"
            explanation = f"{top_one['semantics']} + {top_two['semantics']}"
        else:
            attribution_type = "background_or_unknown"
            explanation = "No dominant stable inducing factor was activated strongly enough."
        return {
            "point_id": sample_row.get("point_id"),
            "year": sample_row.get("year"),
            "disturbance_probability": disturbance_probability,
            "attribution_type": attribution_type,
            "explanation": explanation,
            "top_factors": factor_probs[:3],
        }

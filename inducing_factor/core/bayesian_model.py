import json
import pickle
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from inducing_factor.core.models import FallbackAttributionModel, InducingFactor

try:
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork

    PGMPY_AVAILABLE = True
except ImportError:
    BayesianEstimator = None
    BayesianNetwork = None
    VariableElimination = None
    PGMPY_AVAILABLE = False


class BayesianAttributionEngine:
    def __init__(self, target: str, random_state: int, dominant_threshold: float, synergy_threshold: float):
        self.target = target
        self.random_state = random_state
        self.dominant_threshold = dominant_threshold
        self.synergy_threshold = synergy_threshold
        self.model = None
        self.inference_engine = None
        self.model_backend = "pgmpy" if PGMPY_AVAILABLE else "fallback"
        self.network_feature_columns: List[str] = []

    def factor_activation(self, label_df: pd.DataFrame, factor: InducingFactor) -> pd.Series:
        mask = pd.Series(True, index=label_df.index)
        for item in factor.items:
            feature, state = item.split("=", 1)
            mask &= label_df[feature].astype(str) == state
        return mask.astype(int)

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
        bootstrap_iterations: int = 10,
        edge_frequency_threshold: float = 0.5,
    ) -> List[Tuple[str, str]]:
        try:
            from pgmpy.estimators import BicScore, HillClimbSearch
        except ImportError:
            return self._fixed_edges(network_features, factor_columns, target)
        edge_counts = Counter()
        rng = np.random.default_rng(self.random_state)
        n_samples = len(bn_df)
        for _ in range(bootstrap_iterations):
            idx = rng.integers(0, n_samples, size=n_samples)
            bootstrap_df = bn_df.iloc[idx].reset_index(drop=True)
            try:
                hc = HillClimbSearch(bootstrap_df)
                best_model = hc.estimate(scoring_method=BicScore(bootstrap_df), max_indeg=4)
                for edge in best_model.edges():
                    edge_counts[edge] += 1
            except Exception:
                continue
        consensus_edges = [edge for edge, count in edge_counts.items() if count >= edge_frequency_threshold * bootstrap_iterations]
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
            except Exception:
                model = BayesianNetwork(unique_edges)
            model.fit(bn_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
            model.check_model()
            self.inference_engine = VariableElimination(model)
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
                query = self.inference_engine.query([factor.factor_id], evidence=evidence, show_progress=False)
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

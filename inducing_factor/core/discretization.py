from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class AdaptiveDiscretizer:
    """Discretize continuous features while preserving categorical features."""

    def __init__(
        self,
        method: str = "supervised_tree",
        n_bins: int = 3,
        min_samples_leaf: float = 0.05,
        categorical_unique_threshold: int = 12,
    ):
        self.method = method
        self.n_bins = n_bins
        self.min_samples_leaf = min_samples_leaf
        self.categorical_unique_threshold = categorical_unique_threshold
        self.feature_meta_: Dict[str, dict] = {}
        self.features_: List[str] = []

    def fit(self, df: pd.DataFrame, y: pd.Series):
        self.features_ = list(df.columns)
        self.feature_meta_ = {}
        for feature in self.features_:
            series = df[feature]
            non_null = series.dropna()
            if non_null.empty:
                self.feature_meta_[feature] = {"kind": "categorical", "categories": ["missing"]}
                continue

            is_numeric = pd.api.types.is_numeric_dtype(series)
            unique_count = non_null.nunique(dropna=True)
            if (not is_numeric) or unique_count <= self.categorical_unique_threshold:
                self.feature_meta_[feature] = {
                    "kind": "categorical",
                    "categories": sorted(non_null.astype(str).unique().tolist()),
                }
                continue

            self.feature_meta_[feature] = self._fit_continuous_feature(feature, series.astype(float), y)
        return self

    def _fit_continuous_feature(self, feature: str, series: pd.Series, y: pd.Series) -> dict:
        thresholds: List[float] = []
        mask = series.notna() & y.notna()
        if mask.sum() > 2 and y[mask].nunique() > 1 and self.method in ("supervised_tree", "mdlp"):
            min_leaf = self.min_samples_leaf
            if 0 < min_leaf < 1:
                min_leaf = max(2, int(mask.sum() * min_leaf))
            else:
                min_leaf = max(2, int(min_leaf))

            tree = DecisionTreeClassifier(
                criterion="entropy",
                max_leaf_nodes=self.n_bins,
                min_samples_leaf=min_leaf,
                random_state=42,
            )
            try:
                tree.fit(series[mask].to_numpy().reshape(-1, 1), y[mask].astype(int).to_numpy())
                thresholds = sorted({float(t) for t in tree.tree_.threshold if t != -2})
            except Exception:
                thresholds = []

        if not thresholds:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            thresholds = sorted({float(series.quantile(q)) for q in quantiles if not pd.isna(series.quantile(q))})

        bins = [-np.inf] + thresholds + [np.inf]
        n_states = max(1, len(bins) - 1)
        return {"kind": "continuous", "bins": bins, "labels": self._default_labels(feature, n_states)}

    def _default_labels(self, feature: str, n_states: int) -> List[str]:
        if n_states <= 1:
            return ["state_0"]
        if "dist_" in feature:
            palettes = {2: ["near", "far"], 3: ["near", "mid", "far"], 4: ["very_near", "near", "far", "very_far"]}
            return palettes.get(n_states, [f"state_{i}" for i in range(n_states)])
        palettes = {2: ["low", "high"], 3: ["low", "medium", "high"], 4: ["very_low", "low", "high", "very_high"]}
        return palettes.get(n_states, [f"state_{i}" for i in range(n_states)])

    def transform(self, df: pd.DataFrame, as_labels: bool = False) -> pd.DataFrame:
        transformed = pd.DataFrame(index=df.index)
        for feature in self.features_:
            meta = self.feature_meta_[feature]
            series = df[feature]
            if meta["kind"] == "continuous":
                codes = pd.cut(series.astype(float), bins=meta["bins"], labels=False, include_lowest=True)
                if as_labels:
                    labels = pd.Series(index=df.index, dtype="object")
                    valid_mask = codes.notna()
                    labels.loc[valid_mask] = [meta["labels"][int(v)] for v in codes.loc[valid_mask]]
                    labels.loc[~valid_mask] = "missing"
                    transformed[feature] = labels.astype(str)
                else:
                    transformed[feature] = codes.fillna(-1).astype(int)
            else:
                categories = meta["categories"]
                mapping = {value: idx for idx, value in enumerate(categories)}
                raw = series.astype(str)
                if as_labels:
                    transformed[feature] = raw.where(series.notna(), "missing")
                else:
                    transformed[feature] = raw.map(mapping).fillna(-1).astype(int)
        return transformed

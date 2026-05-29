from typing import Dict, List
from math import log2

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

    def _fit_mdlp(self, feature: str, series: pd.Series, y: pd.Series) -> dict:
        """Minimum Description Length Principle discretization (Fayyad & Irani 1993)."""
        mask = series.notna() & y.notna()
        if mask.sum() < 4:
            return self._fallback_quantile(feature, series)

        def entropy(labels):
            counts = np.bincount(labels.astype(int))
            probs = counts[counts > 0] / len(labels)
            return -float(np.sum(probs * np.log2(probs + 1e-12)))

        def mdlp_accept(values, labels, threshold):
            left_mask = values <= threshold
            right_mask = ~left_mask
            if left_mask.sum() < 2 or right_mask.sum() < 2:
                return False
            n = len(labels)
            e_parent = entropy(labels)
            e_left = entropy(labels[left_mask])
            e_right = entropy(labels[right_mask])
            n_left, n_right = left_mask.sum(), right_mask.sum()
            info_gain = e_parent - (n_left / n * e_left + n_right / n * e_right)
            k = len(np.unique(labels))
            k_left = len(np.unique(labels[left_mask]))
            k_right = len(np.unique(labels[right_mask]))
            delta = log2(3 ** k - 2) - k * e_parent + k_left * e_left + k_right * e_right
            return info_gain > (log2(n - 1) / n + delta / n)

        def find_cuts(values, labels, depth=0, max_depth=6):
            if depth >= max_depth or len(values) < 4 or len(np.unique(labels)) < 2:
                return []
            sorted_idx = np.argsort(values)
            sv = values[sorted_idx]
            sl = labels[sorted_idx]
            midpoints = (sv[:-1] + sv[1:]) / 2
            unique_mids = np.unique(midpoints)
            best_gain, best_mid = -1.0, None
            sample_mids = unique_mids[::max(1, len(unique_mids) // 50)]
            for mid in sample_mids:
                if mdlp_accept(sv, sl, mid):
                    gain = entropy(sl) - (
                        (sv <= mid).sum() / len(sl) * entropy(sl[sv <= mid])
                        + (sv > mid).sum() / len(sl) * entropy(sl[sv > mid])
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_mid = mid
            if best_mid is None:
                return []
            left_mask = values <= best_mid
            right_mask = ~left_mask
            left_cuts = find_cuts(values[left_mask], labels[left_mask], depth + 1, max_depth)
            right_cuts = find_cuts(values[right_mask], labels[right_mask], depth + 1, max_depth)
            return left_cuts + [best_mid] + right_cuts

        try:
            thresholds = sorted(set(find_cuts(series[mask].to_numpy(), y[mask].astype(int).to_numpy())))
        except Exception:
            return self._fallback_quantile(feature, series)

        if not thresholds:
            return self._fallback_quantile(feature, series)

        bins = [-np.inf] + thresholds + [np.inf]
        n_states = len(bins) - 1
        return {"kind": "continuous", "bins": bins, "labels": self._default_labels(feature, n_states)}

    def _fallback_quantile(self, feature: str, series: pd.Series) -> dict:
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        thresholds = sorted({float(series.quantile(q)) for q in quantiles if not pd.isna(series.quantile(q))})
        bins = [-np.inf] + thresholds + [np.inf]
        n_states = max(1, len(bins) - 1)
        return {"kind": "continuous", "bins": bins, "labels": self._default_labels(feature, n_states)}

    def _fit_continuous_feature(self, feature: str, series: pd.Series, y: pd.Series) -> dict:
        thresholds: List[float] = []
        mask = series.notna() & y.notna()
        if mask.sum() > 2 and y[mask].nunique() > 1:
            if self.method == "mdlp":
                return self._fit_mdlp(feature, series, y)

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
            return self._fallback_quantile(feature, series)

        bins = [-np.inf] + thresholds + [np.inf]
        n_states = max(1, len(bins) - 1)
        return {"kind": "continuous", "bins": bins, "labels": self._default_labels(feature, n_states)}

    def _default_labels(self, feature: str, n_states: int) -> List[str]:
        if n_states <= 1:
            return ["state_0"]
        if "dist_" in feature:
            palettes = {
                2: ["near", "far"],
                3: ["near", "mid", "far"],
                4: ["very_near", "near", "far", "very_far"],
                5: ["very_near", "near", "mid", "far", "very_far"],
            }
            return palettes.get(n_states, [f"state_{i}" for i in range(n_states)])
        palettes = {
            2: ["low", "high"],
            3: ["low", "medium", "high"],
            4: ["very_low", "low", "high", "very_high"],
            5: ["very_low", "low", "medium", "high", "very_high"],
        }
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
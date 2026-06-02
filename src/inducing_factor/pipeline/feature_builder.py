from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureBuilder:
    def __init__(
        self,
        target: str,
        lookback_years: int,
        static_features: List[str],
        dynamic_features: List[str],
        categorical_temporal_features: List[str],
        direct_features: List[str],
    ):
        self.target = target
        self.lookback_years = lookback_years
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.categorical_temporal_features = categorical_temporal_features
        self.direct_features = direct_features
        self.feature_window_map: Dict[str, int] = {}

    def _previous_year_month(self, df: pd.DataFrame, month: int) -> pd.Series:
        return df.groupby("point_id")[f"rainfall_month_{month:02d}"].shift(1)

    def build_rainfall_window_features(
        self, df: pd.DataFrame, predict_month: Optional[int] = None,
    ) -> pd.DataFrame:
        """Build rainfall window features (prev_1m/2m/3m, total_3m).

        In training mode (predict_month is None): uses per-row ``reference_month``
        column (derived from disturbance event labels). Each row gets a different
        reference month depending on when its disturbance occurred.

        In prediction mode (predict_month is not None): uses the supplied month
        as a uniform reference for ALL rows. This is the leakage-free path that
        does NOT require any CCDC-derived columns.
        """
        if predict_month is not None:
            reference_month = pd.Series(predict_month, index=df.index, dtype=float)
        else:
            if "reference_month" not in df.columns:
                return df
            reference_month = pd.to_numeric(df["reference_month"], errors="coerce")

        # Fill missing current-year rainfall data from previous year (shift(1))
        for month in range(1, 13):
            col = f"rainfall_month_{month:02d}"
            if col not in df.columns:
                continue
            current_vals = pd.to_numeric(df[col], errors="coerce")
            nan_mask = current_vals.isna()
            if nan_mask.any():
                prev_vals = self._previous_year_month(df, month)
                df.loc[nan_mask, col] = prev_vals.loc[nan_mask]

        month_values = {month: pd.to_numeric(df[f"rainfall_month_{month:02d}"], errors="coerce") for month in range(1, 13) if f"rainfall_month_{month:02d}" in df.columns}
        prev_year_values = {month: self._previous_year_month(df, month) for month in month_values}
        for offset in range(1, 4):
            values = pd.Series(np.nan, index=df.index, dtype=float)
            for month in range(1, 13):
                current_mask = reference_month == month
                if not current_mask.any():
                    continue
                source_month = month - offset
                if source_month >= 1:
                    values.loc[current_mask] = month_values[source_month].loc[current_mask]
                else:
                    values.loc[current_mask] = prev_year_values[source_month + 12].loc[current_mask]
            df[f"rainfall_prev_{offset}m"] = values
        df["rainfall_prev_total_3m"] = df[["rainfall_prev_1m", "rainfall_prev_2m", "rainfall_prev_3m"]].sum(axis=1, min_count=3)
        return df

    def compute_feature_lags(self, df: pd.DataFrame, max_lag: int = 5) -> Dict[str, int]:
        feature_lags = {}
        for feature in self.dynamic_features:
            if feature not in df.columns:
                continue
            best_lag = 1
            best_corr = -1.0
            for lag in range(1, max_lag + 1):
                shifted = df.groupby("point_id")[feature].shift(lag)
                valid = df[self.target].notna() & shifted.notna()
                if valid.sum() < 10:
                    continue
                corr = np.corrcoef(df.loc[valid, self.target], shifted.loc[valid])[0, 1]
                if np.isfinite(corr) and abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            if best_corr == -1.0:
                best_lag = self.lookback_years
            feature_lags[feature] = best_lag
        self.feature_window_map = feature_lags
        return feature_lags

    def rolling_history_feature(self, df: pd.DataFrame, feature: str, agg: str) -> pd.Series:
        grouped = df.groupby("point_id")[feature]
        window = self.feature_window_map.get(feature, self.lookback_years)
        if agg == "last":
            return grouped.shift(1)
        if agg == "mean":
            return grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
        if agg == "max":
            return grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=window).max())
        if agg == "min":
            return grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=window).min())
        if agg == "sum":
            return grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=window).sum())
        raise ValueError(f"Unsupported aggregation: {agg}")

    def build_chunk_features(
        self,
        df_chunk: pd.DataFrame,
        predict_month: Optional[int] = None,
    ) -> tuple[pd.DataFrame, List[str]]:
        df = df_chunk.sort_values(["point_id", "year"]).copy()

        if predict_month is not None:
            if "disturbance_flag" in df.columns:
                df[self.target] = pd.to_numeric(df["disturbance_flag"], errors="coerce").fillna(0).astype(int)
            elif "disturbance_time_value" in df.columns:
                df[self.target] = (pd.to_numeric(df["disturbance_time_value"], errors="coerce") > 0).astype(int)
            elif "disturbance_type" in df.columns:
                df[self.target] = (pd.to_numeric(df["disturbance_type"], errors="coerce") > 0).astype(int)
            else:
                df[self.target] = 0
        else:
            if "disturbance_flag" in df.columns:
                df[self.target] = pd.to_numeric(df["disturbance_flag"], errors="coerce").fillna(0).astype(int)
            elif "disturbance_time_value" in df.columns:
                df[self.target] = (pd.to_numeric(df["disturbance_time_value"], errors="coerce") > 0).astype(int)
            else:
                df[self.target] = (pd.to_numeric(df["disturbance_type"], errors="coerce") > 0).astype(int)

        df = self.build_rainfall_window_features(df, predict_month=predict_month)
        if not self.feature_window_map:
            self.compute_feature_lags(df)
        engineered_features: List[str] = []
        for feature in self.static_features:
            if feature in df.columns:
                engineered_features.append(feature)
        for feature in self.direct_features:
            if feature in df.columns:
                engineered_features.append(feature)
        for feature in self.dynamic_features:
            if feature not in df.columns:
                continue
            for agg in ["last", "mean", "max"]:
                new_col = f"{feature}_prev_{agg}"
                df[new_col] = self.rolling_history_feature(df, feature, agg)
                engineered_features.append(new_col)
        for feature in self.categorical_temporal_features:
            if feature in df.columns:
                new_col = f"{feature}_prev_last"
                df[new_col] = self.rolling_history_feature(df, feature, "last")
                engineered_features.append(new_col)
        return df, list(dict.fromkeys(engineered_features))

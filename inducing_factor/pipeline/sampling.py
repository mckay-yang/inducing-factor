import pandas as pd


class StratifiedBackgroundSampler:
    def __init__(self, static_features, background_sample_rate: float, random_state: int):
        self.static_features = static_features
        self.background_sample_rate = background_sample_rate
        self.random_state = random_state

    def sample(self, df_bg: pd.DataFrame, df_dist: pd.DataFrame) -> pd.DataFrame:
        if self.background_sample_rate >= 1:
            return df_bg
        sample_n = max(1, int(len(df_bg) * self.background_sample_rate))
        sample_n = min(sample_n, len(df_bg))
        if sample_n >= len(df_bg):
            return df_bg
        static_features = [f for f in self.static_features if f in df_bg.columns]
        if not static_features:
            return df_bg.sample(n=sample_n, random_state=self.random_state)
        try:
            df_bg_copy = df_bg.copy()
            df_dist_copy = df_dist.copy()
            for feature in static_features:
                if pd.api.types.is_numeric_dtype(df_bg_copy[feature]):
                    df_bg_copy[f"{feature}_stratum"] = pd.qcut(df_bg_copy[feature], q=3, labels=False, duplicates="drop")
                    df_dist_copy[f"{feature}_stratum"] = pd.qcut(df_dist_copy[feature], q=3, labels=False, duplicates="drop")
                else:
                    df_bg_copy[f"{feature}_stratum"] = df_bg_copy[feature].astype(str)
                    df_dist_copy[f"{feature}_stratum"] = df_dist_copy[feature].astype(str)
            stratum_cols = [f"{f}_stratum" for f in static_features]
            df_bg_copy["stratum"] = df_bg_copy[stratum_cols].astype(str).agg("_".join, axis=1)
            df_dist_copy["stratum"] = df_dist_copy[stratum_cols].astype(str).agg("_".join, axis=1)
            stratum_counts_dist = df_dist_copy["stratum"].value_counts(normalize=True)
            stratum_samples = {}
            remaining = sample_n
            for stratum, prop in stratum_counts_dist.items():
                if stratum not in df_bg_copy["stratum"].values:
                    continue
                stratum_bg = df_bg_copy[df_bg_copy["stratum"] == stratum]
                take_n = min(max(1, int(sample_n * prop)), len(stratum_bg))
                stratum_samples[stratum] = take_n
                remaining -= take_n
            if remaining > 0:
                available_strata = [s for s in df_bg_copy["stratum"].unique() if s not in stratum_samples]
                if available_strata:
                    per_stratum = max(1, remaining // len(available_strata))
                    for stratum in available_strata:
                        stratum_bg = df_bg_copy[df_bg_copy["stratum"] == stratum]
                        take_n = min(per_stratum, len(stratum_bg))
                        stratum_samples[stratum] = take_n
                        remaining -= take_n
                        if remaining <= 0:
                            break
            sampled = []
            for stratum, take_n in stratum_samples.items():
                if take_n > 0:
                    stratum_bg = df_bg_copy[df_bg_copy["stratum"] == stratum]
                    sampled.append(stratum_bg.sample(n=take_n, random_state=self.random_state))
            if not stratum_samples:
                return df_bg.sample(n=sample_n, random_state=self.random_state)
            if sampled:
                combined = pd.concat(sampled, ignore_index=True)
                return combined.drop(columns=[c for c in combined.columns if c.endswith("_stratum") or c == "stratum"])
            return df_bg.sample(n=sample_n, random_state=self.random_state)
        except Exception:
            return df_bg.sample(n=sample_n, random_state=self.random_state)

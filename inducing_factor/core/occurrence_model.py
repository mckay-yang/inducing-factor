from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from inducing_factor.core.models import InducingFactor


@dataclass
class FallbackOccurrenceModel:
    base_rate: float

    def predict_proba(self, X):
        probs = np.full((X.shape[0], 2), self.base_rate, dtype=float)
        probs[:, 0] = 1.0 - self.base_rate
        return probs


class OccurrenceProbabilityEngine:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.factor_columns: List[str] = []

    def _factor_activation_frame(self, label_df: pd.DataFrame, selected_factors: List[InducingFactor]) -> pd.DataFrame:
        factor_frame = pd.DataFrame(index=label_df.index)
        for factor in selected_factors:
            mask = pd.Series(True, index=label_df.index)
            for item in factor.items:
                feature, state = item.split("=", 1)
                mask &= label_df[feature].astype(str) == state
            factor_frame[factor.factor_id] = mask.astype(int)
        return factor_frame

    def fit(self, label_df: pd.DataFrame, target: pd.Series, selected_factors: List[InducingFactor]):
        factor_frame = self._factor_activation_frame(label_df, selected_factors)
        self.factor_columns = factor_frame.columns.tolist()
        y = pd.to_numeric(target, errors="coerce").fillna(0).astype(int).to_numpy()
        if len(np.unique(y)) <= 1:
            self.model = FallbackOccurrenceModel(base_rate=float(np.mean(y)) if len(y) else 0.0)
            return self.model
        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        model.fit(factor_frame.to_numpy(), y)
        self.model = model
        return model

    def predict_proba(self, label_df: pd.DataFrame, selected_factors: List[InducingFactor]) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(label_df), 2), dtype=float)
        factor_frame = self._factor_activation_frame(label_df, selected_factors)
        return self.model.predict_proba(factor_frame.to_numpy())

    def save(self, path: str):
        joblib.dump({"model": self.model, "factor_columns": self.factor_columns}, path)

    def load(self, path: str):
        payload = joblib.load(path)
        self.model = payload.get("model")
        self.factor_columns = payload.get("factor_columns", [])
        return self

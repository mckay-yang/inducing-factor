from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from inducing_factor.core.models import InducingFactor


@dataclass
class FallbackClassModel:
    classes_: List[int]
    priors: Dict[int, float]

    def predict_proba(self, X):
        n_rows = X.shape[0]
        probs = np.asarray([self.priors[c] for c in self.classes_], dtype=float)
        return np.tile(probs, (n_rows, 1))


class DisturbanceClassEngine:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.classes_: np.ndarray | None = None
        self.factor_columns: List[str] = []
        self.class_labels: Dict[int, str] = {
            1: "fire",
            2: "logging",
            3: "geo_hazard",
            4: "road_construction",
        }

    def _factor_activation_frame(self, label_df: pd.DataFrame, selected_factors: List[InducingFactor]) -> pd.DataFrame:
        factor_frame = pd.DataFrame(index=label_df.index)
        for factor in selected_factors:
            mask = pd.Series(True, index=label_df.index)
            for item in factor.items:
                feature, state = item.split("=", 1)
                mask &= label_df[feature].astype(str) == state
            factor_frame[factor.factor_id] = mask.astype(int)
        return factor_frame

    def fit(self, label_df: pd.DataFrame, training_samples: pd.DataFrame, selected_factors: List[InducingFactor]):
        class_mask = pd.to_numeric(training_samples.get("disturbance_class"), errors="coerce").isin([1, 2, 3, 4])
        if not class_mask.any():
            self.model = None
            self.classes_ = np.asarray([], dtype=int)
            self.factor_columns = []
            return None
        factor_frame = self._factor_activation_frame(label_df.loc[class_mask], selected_factors)
        self.factor_columns = factor_frame.columns.tolist()
        y = pd.to_numeric(training_samples.loc[class_mask, "disturbance_class"], errors="coerce").astype(int).to_numpy()
        self.classes_ = np.unique(y)
        if len(self.classes_) <= 1:
            prior = {int(self.classes_[0]): 1.0}
            self.model = FallbackClassModel(classes_=self.classes_.tolist(), priors=prior)
            return self.model
        model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        model.fit(factor_frame.to_numpy(), y)
        self.model = model
        return self.model

    def predict_proba(self, label_df: pd.DataFrame, selected_factors: List[InducingFactor]) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(index=label_df.index)
        factor_frame = self._factor_activation_frame(label_df, selected_factors)
        probs = self.model.predict_proba(factor_frame.to_numpy())
        classes = [int(c) for c in getattr(self.model, "classes_", self.classes_)]
        return pd.DataFrame(probs, index=label_df.index, columns=[f"class_probability_{c}" for c in classes])

    def summarize_factor_class_associations(
        self,
        label_df: pd.DataFrame,
        training_samples: pd.DataFrame,
        selected_factors: List[InducingFactor],
    ) -> pd.DataFrame:
        class_mask = pd.to_numeric(training_samples.get("disturbance_class"), errors="coerce").isin([1, 2, 3, 4])
        if not class_mask.any():
            return pd.DataFrame()
        factor_frame = self._factor_activation_frame(label_df.loc[class_mask], selected_factors)
        class_values = pd.to_numeric(training_samples.loc[class_mask, "disturbance_class"], errors="coerce").astype(int)
        records = []
        for factor in selected_factors:
            active = factor_frame[factor.factor_id] == 1
            if not active.any():
                continue
            dist = class_values.loc[active].value_counts(normalize=True).sort_index()
            record = {
                "factor_id": factor.factor_id,
                "factor_name": factor.name,
                "semantics": factor.semantics,
                "active_samples": int(active.sum()),
            }
            for cls, prob in dist.items():
                record[f"class_{int(cls)}_share"] = float(prob)
            if not dist.empty:
                record["dominant_class"] = int(dist.idxmax())
                record["dominant_class_label"] = self.class_labels.get(int(dist.idxmax()), str(int(dist.idxmax())))
            records.append(record)
        return pd.DataFrame(records).sort_values(["active_samples", "factor_id"], ascending=[False, True])

    def save(self, path: str):
        joblib.dump(
            {
                "model": self.model,
                "classes": [] if self.classes_ is None else [int(c) for c in self.classes_],
                "factor_columns": self.factor_columns,
                "class_labels": self.class_labels,
            },
            path,
        )

    def load(self, path: str):
        payload = joblib.load(path)
        self.model = payload.get("model")
        self.classes_ = np.asarray(payload.get("classes", []), dtype=int)
        self.factor_columns = payload.get("factor_columns", [])
        self.class_labels = payload.get("class_labels", self.class_labels)
        return self

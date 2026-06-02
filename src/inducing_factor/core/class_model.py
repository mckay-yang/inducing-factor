import logging
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from inducing_factor.core.models import InducingFactor

logger = logging.getLogger(__name__)


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
        self.model_status: str = "untrained"

    def is_available(self) -> bool:
        """Return True if the class model can produce meaningful predictions."""
        return self.model_status in ("trained", "fallback_single_class")

    def available_classes(self) -> List[int]:
        """Return the list of disturbance classes this model can predict."""
        if not self.is_available():
            return []
        return [] if self.classes_ is None else [int(c) for c in self.classes_]

    def _factor_activation_frame(self, label_df: pd.DataFrame, selected_factors: List[InducingFactor]) -> pd.DataFrame:
        factor_frame = pd.DataFrame(index=label_df.index)
        for factor in selected_factors:
            mask = pd.Series(True, index=label_df.index)
            for item in factor.items:
                feature, state = item.split("=", 1)
                mask &= label_df[feature].astype(str) == state
            factor_frame[factor.factor_id] = mask.astype(int)
        return factor_frame

    def fit(self, label_df: pd.DataFrame, training_samples: pd.DataFrame, selected_factors: List[InducingFactor], continuous_features: pd.DataFrame = None):
        class_mask = pd.to_numeric(training_samples.get("disturbance_class"), errors="coerce").isin([1, 2, 3, 4])
        if not class_mask.any():
            self.model_status = "unavailable"
            self.model = None
            self.classes_ = np.asarray([], dtype=int)
            self.factor_columns = []
            logger.warning(
                "DisturbanceClassEngine.fit: no labeled disturbance_class values (1-4) found in training data. "
                "Class model is unavailable. Dominant disturbance class predictions will NOT be produced."
            )
            return None
        factor_frame = self._factor_activation_frame(label_df.loc[class_mask], selected_factors)
        if continuous_features is not None:
            cont_masked = continuous_features.loc[class_mask]
            feature_frame = pd.concat([factor_frame, cont_masked], axis=1)
        else:
            feature_frame = factor_frame
        self.factor_columns = feature_frame.columns.tolist()
        y = pd.to_numeric(training_samples.loc[class_mask, "disturbance_class"], errors="coerce").astype(int).to_numpy()
        self.classes_ = np.unique(y)
        if len(self.classes_) <= 1:
            self.model_status = "fallback_single_class"
            prior = {int(self.classes_[0]): 1.0}
            self.model = FallbackClassModel(classes_=self.classes_.tolist(), priors=prior)
            logger.info(
                "DisturbanceClassEngine.fit: only one disturbance class present (%s). "
                "Using simple prior-based fallback. model_status='fallback_single_class'.",
                self.classes_[0],
            )
            return self.model
        model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        model.fit(feature_frame.to_numpy(), y)
        self.model = model
        self.model_status = "trained"
        return self.model

    def predict_proba(self, label_df: pd.DataFrame, selected_factors: List[InducingFactor]) -> pd.DataFrame:
        if self.model is None:
            if self.model_status == "unavailable":
                logger.warning("DisturbanceClassEngine.predict_proba: class model is unavailable, returning empty DataFrame.")
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
                "model_status": self.model_status,
            },
            path,
        )

    def load(self, path: str):
        payload = joblib.load(path)
        self.model = payload.get("model")
        self.classes_ = np.asarray(payload.get("classes", []), dtype=int)
        self.factor_columns = payload.get("factor_columns", [])
        self.class_labels = payload.get("class_labels", self.class_labels)
        self.model_status = payload.get("model_status", "untrained")
        return self

from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.linear_model import LogisticRegression


@dataclass
class InducingFactor:
    factor_id: str
    name: str
    semantics: str
    items: Tuple[str, ...]
    support_event: float
    support_background: float
    relative_risk: float
    stability: float
    score: float


@dataclass
class FallbackAttributionModel:
    edges: List[Tuple[str, str]]
    factor_columns: List[str]
    target: str
    logistic_model: Optional[LogisticRegression]
    base_rate: float
    backend: str = "fallback"

    def check_model(self):
        return True

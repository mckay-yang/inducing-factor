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
    count_event: int = 0
    count_background: int = 0
    n_event: int = 0
    n_background: int = 0
    rr_ci_lower: float = 0.0
    rr_ci_upper: float = 0.0
    p_value: float = 1.0
    fdr_significant: bool = False
    likelihood_ratio: float = 0.0
    true_relative_risk: float = 0.0


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

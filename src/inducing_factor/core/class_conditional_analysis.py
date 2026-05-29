from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from inducing_factor.core.models import InducingFactor
from inducing_factor.core.statistics import fisher_exact_test


@dataclass
class FactorClassProfile:
    factor_id: str
    factor_name: str
    factor_items: List[str]

    count_in_fire: int
    count_in_logging: int
    count_in_geo_hazard: int
    count_in_road_construction: int
    count_in_unknown: int
    count_in_background: int
    total_events_with_factor: int
    total_labeled_with_factor: int
    total_events: int
    total_labeled: int

    prob_fire_given_factor: float
    prob_logging_given_factor: float
    prob_geo_hazard_given_factor: float
    prob_road_given_factor: float

    support_in_fire: float
    support_in_logging: float
    support_in_geo_hazard: float
    support_in_road: float

    prior_lift_fire: float
    prior_lift_logging: float
    prior_lift_geo: float
    prior_lift_road: float

    fisher_p_fire: float
    fisher_p_logging: float
    fisher_p_geo: float
    fisher_p_road: float


class FactorClassAnalyzer:
    CLASS_LABELS = {1: "fire", 2: "logging", 3: "geo_hazard", 4: "road_construction"}

    def __init__(self, min_count_per_class: int = 10):
        self.min_count_per_class = min_count_per_class

    def compute_class_conditional(
        self,
        factors: List[InducingFactor],
        label_df: pd.DataFrame,
        feature_columns: List[str],
        discretizer,
        target: str = "disturbance_flag",
        class_column: str = "disturbance_class",
    ) -> List[FactorClassProfile]:
        events_mask = label_df[target] == 1
        background_mask = label_df[target] == 0

        labeled_events = label_df[events_mask & (label_df[class_column] > 0)].copy()
        unlabeled_events = label_df[events_mask & (label_df[class_column] == 0)].copy()
        background = label_df[background_mask].copy()

        n_labeled = len(labeled_events)
        n_total_events = int(events_mask.sum())

        class_events = {}
        class_priors = {}
        for cls in self.CLASS_LABELS:
            cls_df = labeled_events[labeled_events[class_column] == cls]
            class_events[cls] = cls_df
            class_priors[cls] = len(cls_df) / max(n_labeled, 1)

        all_discrete = discretizer.transform(label_df[feature_columns], as_labels=True)
        for col in all_discrete.columns:
            all_discrete[col] = all_discrete[col].astype(str)
        all_discrete[target] = label_df[target].values
        all_discrete[class_column] = label_df[class_column].values

        labeled_idx = labeled_events.index
        bg_idx = background.index
        unlabeled_idx = unlabeled_events.index

        labeled_discrete = all_discrete.loc[labeled_idx]
        bg_discrete = all_discrete.loc[bg_idx]
        unlabeled_discrete = all_discrete.loc[unlabeled_idx] if len(unlabeled_idx) > 0 else pd.DataFrame(columns=all_discrete.columns)

        class_discrete = {}
        for cls in self.CLASS_LABELS:
            cls_df = class_events[cls]
            cls_idx = cls_df.index
            class_discrete[cls] = all_discrete.loc[cls_idx]

        results = []
        for factor in factors:
            items = factor.items
            pattern = tuple(sorted(items))

            class_counts = {}
            for cls in self.CLASS_LABELS:
                mask = self._pattern_mask(class_discrete[cls], pattern, feature_columns)
                class_counts[cls] = int(mask.sum())

            unknown_mask = self._pattern_mask(unlabeled_discrete, pattern, feature_columns)
            count_unknown = int(unknown_mask.sum())

            bg_mask = self._pattern_mask(bg_discrete, pattern, feature_columns)
            count_bg = int(bg_mask.sum())

            total_labeled_with_factor = sum(class_counts.values())
            total_with_factor = total_labeled_with_factor + count_unknown

            probs = {}
            for cls in self.CLASS_LABELS:
                probs[cls] = class_counts[cls] / max(total_labeled_with_factor, 1)

            prior_lifts = {}
            for cls in self.CLASS_LABELS:
                prior_lifts[cls] = probs[cls] / max(class_priors[cls], 1e-6)

            fisher_ps = {}
            for cls in self.CLASS_LABELS:
                n_cls = len(class_events[cls])
                _, p = fisher_exact_test(
                    int(class_counts[cls]),
                    n_cls,
                    int(total_labeled_with_factor - class_counts[cls]),
                    int(n_labeled - n_cls),
                )
                fisher_ps[cls] = p

            profile = FactorClassProfile(
                factor_id=factor.factor_id,
                factor_name=factor.name,
                factor_items=list(items),
                count_in_fire=class_counts[1],
                count_in_logging=class_counts[2],
                count_in_geo_hazard=class_counts[3],
                count_in_road_construction=class_counts[4],
                count_in_unknown=count_unknown,
                count_in_background=count_bg,
                total_events_with_factor=total_with_factor,
                total_labeled_with_factor=total_labeled_with_factor,
                total_events=n_total_events,
                total_labeled=n_labeled,
                prob_fire_given_factor=probs[1],
                prob_logging_given_factor=probs[2],
                prob_geo_hazard_given_factor=probs[3],
                prob_road_given_factor=probs[4],
                support_in_fire=class_counts[1] / max(len(class_events[1]), 1),
                support_in_logging=class_counts[2] / max(len(class_events[2]), 1),
                support_in_geo_hazard=class_counts[3] / max(len(class_events[3]), 1),
                support_in_road=class_counts[4] / max(len(class_events[4]), 1),
                prior_lift_fire=prior_lifts[1],
                prior_lift_logging=prior_lifts[2],
                prior_lift_geo=prior_lifts[3],
                prior_lift_road=prior_lifts[4],
                fisher_p_fire=fisher_ps[1],
                fisher_p_logging=fisher_ps[2],
                fisher_p_geo=fisher_ps[3],
                fisher_p_road=fisher_ps[4],
            )
            results.append(profile)

        return results

    def compute_class_conditional_from_counts(
        self,
        factors: List[InducingFactor],
        event_counts_by_class: Dict[int, Counter],
        background_counts: Counter,
        n_event_by_class: Dict[int, int],
        n_event: int,
        n_bg: int,
    ) -> List[FactorClassProfile]:
        n_labeled = sum(n_event_by_class.values())
        class_priors = {}
        for cls in self.CLASS_LABELS:
            class_priors[cls] = n_event_by_class.get(cls, 0) / max(n_labeled, 1)

        results = []
        for factor in factors:
            pattern = tuple(sorted(factor.items))
            class_counts = {}
            for cls in self.CLASS_LABELS:
                cls_counter = event_counts_by_class.get(cls, Counter())
                class_counts[cls] = cls_counter.get(pattern, 0)

            count_bg = background_counts.get(pattern, 0)
            total_labeled_with_factor = sum(class_counts.values())
            count_event = factor.count_event if hasattr(factor, "count_event") else 0
            known_count = count_event - total_labeled_with_factor if count_event > total_labeled_with_factor else 0
            count_unknown = max(0, known_count)

            probs = {}
            for cls in self.CLASS_LABELS:
                probs[cls] = class_counts[cls] / max(total_labeled_with_factor, 1)

            prior_lifts = {}
            for cls in self.CLASS_LABELS:
                prior_lifts[cls] = probs[cls] / max(class_priors[cls], 1e-6)

            fisher_ps = {}
            for cls in self.CLASS_LABELS:
                n_cls = n_event_by_class.get(cls, 0)
                _, p = fisher_exact_test(
                    int(class_counts[cls]),
                    n_cls,
                    int(total_labeled_with_factor - class_counts[cls]),
                    int(n_labeled - n_cls),
                )
                fisher_ps[cls] = p

            profile = FactorClassProfile(
                factor_id=factor.factor_id,
                factor_name=factor.name,
                factor_items=list(factor.items),
                count_in_fire=class_counts[1],
                count_in_logging=class_counts[2],
                count_in_geo_hazard=class_counts[3],
                count_in_road_construction=class_counts[4],
                count_in_unknown=count_unknown,
                count_in_background=count_bg,
                total_events_with_factor=total_labeled_with_factor + count_unknown,
                total_labeled_with_factor=total_labeled_with_factor,
                total_events=n_event,
                total_labeled=n_labeled,
                prob_fire_given_factor=probs[1],
                prob_logging_given_factor=probs[2],
                prob_geo_hazard_given_factor=probs[3],
                prob_road_given_factor=probs[4],
                support_in_fire=class_counts[1] / max(n_event_by_class.get(1, 1), 1),
                support_in_logging=class_counts[2] / max(n_event_by_class.get(2, 1), 1),
                support_in_geo_hazard=class_counts[3] / max(n_event_by_class.get(3, 1), 1),
                support_in_road=class_counts[4] / max(n_event_by_class.get(4, 1), 1),
                prior_lift_fire=prior_lifts[1],
                prior_lift_logging=prior_lifts[2],
                prior_lift_geo=prior_lifts[3],
                prior_lift_road=prior_lifts[4],
                fisher_p_fire=fisher_ps[1],
                fisher_p_logging=fisher_ps[2],
                fisher_p_geo=fisher_ps[3],
                fisher_p_road=fisher_ps[4],
            )
            results.append(profile)

        return results

    @staticmethod
    def _pattern_mask(
        discrete_df: pd.DataFrame,
        pattern: Tuple[str, ...],
        feature_columns: List[str],
    ) -> np.ndarray:
        mask = np.ones(len(discrete_df), dtype=bool)
        for item in pattern:
            if "=" not in item:
                continue
            feature, value = item.split("=", 1)
            if feature in discrete_df.columns:
                col_values = discrete_df[feature].astype(str).values
                mask &= col_values == value
            elif feature in feature_columns:
                col_idx = feature_columns.index(feature)
                if col_idx < discrete_df.shape[1]:
                    col_values = discrete_df.iloc[:, col_idx].astype(str).values
                    mask &= col_values == value
        return mask

    def find_top_class_specific_factors(
        self,
        profiles: List[FactorClassProfile],
        min_prior_lift: float = 2.0,
        top_n: int = 20,
        min_count: int = 10,
    ) -> Dict[str, List[FactorClassProfile]]:
        class_fields = {
            "fire": "prior_lift_fire",
            "logging": "prior_lift_logging",
            "geo_hazard": "prior_lift_geo",
            "road_construction": "prior_lift_road",
        }

        top_factors: Dict[str, List[FactorClassProfile]] = {}
        for class_name, field in class_fields.items():
            sorted_profiles = sorted(profiles, key=lambda p: getattr(p, field), reverse=True)
            top = [
                p for p in sorted_profiles
                if getattr(p, field) >= min_prior_lift
                and p.total_labeled_with_factor >= min_count
            ]
            top_factors[class_name] = top[:top_n]

        return top_factors
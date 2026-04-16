import itertools
import math
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from inducing_factor.core.models import InducingFactor
from inducing_factor.core.semantics import pattern_name, pattern_semantics


class FactorMiner:
    def __init__(
        self,
        max_pattern_size: int,
        min_support: float,
        min_relative_risk: float,
        min_stability: float,
        bootstrap_iterations: int,
        max_bootstrap_samples: int,
        random_state: int,
    ):
        self.max_pattern_size = max_pattern_size
        self.min_support = min_support
        self.min_relative_risk = min_relative_risk
        self.min_stability = min_stability
        self.bootstrap_iterations = bootstrap_iterations
        self.max_bootstrap_samples = max_bootstrap_samples
        self.random_state = random_state

    def transactions_from_labels(self, label_df: pd.DataFrame) -> List[Tuple[str, ...]]:
        transactions = []
        for _, row in label_df.iterrows():
            transactions.append(tuple(sorted(f"{feature}={row[feature]}" for feature in label_df.columns)))
        return transactions

    def count_patterns(self, transactions: Sequence[Tuple[str, ...]]) -> Counter:
        counts: Counter = Counter()
        for items in transactions:
            max_len = min(self.max_pattern_size, len(items))
            for size in range(1, max_len + 1):
                for combo in itertools.combinations(items, size):
                    counts[combo] += 1
        return counts

    def update_pattern_counts(self, label_df: pd.DataFrame, target: pd.Series, event_counts: Counter, background_counts: Counter) -> tuple[int, int]:
        target_numeric = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
        event_total = 0
        background_total = 0
        for idx, row in label_df.iterrows():
            items = tuple(sorted(f"{feature}={row[feature]}" for feature in label_df.columns))
            counts = event_counts if int(target_numeric.loc[idx]) == 1 else background_counts
            if int(target_numeric.loc[idx]) == 1:
                event_total += 1
            else:
                background_total += 1
            max_len = min(self.max_pattern_size, len(items))
            for size in range(1, max_len + 1):
                for combo in itertools.combinations(items, size):
                    counts[combo] += 1
        return event_total, background_total

    def mine_pattern_candidates_from_counts(
        self,
        event_counts: Counter,
        background_counts: Counter,
        n_event: int,
        n_bg: int,
    ) -> Dict[Tuple[str, ...], dict]:
        if n_event <= 0:
            raise ValueError("No disturbance transactions were provided for pattern mining")
        min_event_count = max(1, math.ceil(self.min_support * n_event))
        epsilon = 1e-6
        candidates: Dict[Tuple[str, ...], dict] = {}
        for pattern, count_event in event_counts.items():
            if count_event < min_event_count:
                continue
            count_bg = background_counts.get(pattern, 0)
            support_event = count_event / n_event
            support_background = count_bg / max(n_bg, 1)
            relative_risk = (support_event + epsilon) / (support_background + epsilon)
            if relative_risk < self.min_relative_risk:
                continue
            candidates[pattern] = {
                "count_event": count_event,
                "count_background": count_bg,
                "support_event": support_event,
                "support_background": support_background,
                "relative_risk": relative_risk,
            }
        return candidates

    def mine_pattern_candidates(
        self,
        event_transactions: Sequence[Tuple[str, ...]],
        background_transactions: Sequence[Tuple[str, ...]],
    ) -> Dict[Tuple[str, ...], dict]:
        if not event_transactions:
            raise ValueError("No disturbance transactions were provided for pattern mining")
        event_counts = self.count_patterns(event_transactions)
        background_counts = self.count_patterns(background_transactions)
        n_event = len(event_transactions)
        n_bg = max(len(background_transactions), 1)
        min_event_count = max(1, math.ceil(self.min_support * n_event))
        epsilon = 1e-6
        candidates: Dict[Tuple[str, ...], dict] = {}
        for pattern, count_event in event_counts.items():
            if count_event < min_event_count:
                continue
            count_bg = background_counts.get(pattern, 0)
            support_event = count_event / n_event
            support_background = count_bg / n_bg
            relative_risk = (support_event + epsilon) / (support_background + epsilon)
            if relative_risk < self.min_relative_risk:
                continue
            candidates[pattern] = {
                "count_event": count_event,
                "count_background": count_bg,
                "support_event": support_event,
                "support_background": support_background,
                "relative_risk": relative_risk,
            }
        return candidates

    def estimate_pattern_stability(
        self,
        event_transactions: Sequence[Tuple[str, ...]],
        background_transactions: Sequence[Tuple[str, ...]],
    ) -> Dict[Tuple[str, ...], float]:
        import numpy as np

        rng = np.random.default_rng(self.random_state)
        stability_counter: Counter = Counter()
        if not event_transactions:
            return {}
        for _ in range(self.bootstrap_iterations):
            event_size = min(len(event_transactions), self.max_bootstrap_samples)
            bg_size = min(len(background_transactions), self.max_bootstrap_samples)
            event_idx = rng.integers(0, len(event_transactions), size=event_size)
            event_sample = [event_transactions[i] for i in event_idx]
            if background_transactions:
                bg_idx = rng.integers(0, len(background_transactions), size=bg_size)
                bg_sample = [background_transactions[i] for i in bg_idx]
            else:
                bg_sample = []
            mined = self.mine_pattern_candidates(event_sample, bg_sample)
            for pattern in mined.keys():
                stability_counter[pattern] += 1
        return {pattern: count / max(self.bootstrap_iterations, 1) for pattern, count in stability_counter.items()}

    def reduce_redundant_patterns(
        self,
        pattern_candidates: Dict[Tuple[str, ...], dict],
        pattern_stability: Dict[Tuple[str, ...], float],
        redundancy_threshold: float = 0.1,
    ) -> Dict[Tuple[str, ...], dict]:
        stable_patterns = {
            p: {**stats, "stability": pattern_stability.get(p, 0.0)}
            for p, stats in pattern_candidates.items()
            if pattern_stability.get(p, 0.0) >= self.min_stability
        }
        if not stable_patterns:
            return {}
        patterns_by_size = {}
        for pattern in stable_patterns.keys():
            patterns_by_size.setdefault(len(pattern), []).append(pattern)
        to_remove = set()
        for size in sorted(patterns_by_size.keys(), reverse=True):
            if size == 1:
                continue
            for pattern in patterns_by_size[size]:
                if pattern in to_remove:
                    continue
                rr_pattern = stable_patterns[pattern]["relative_risk"]
                items = list(pattern)
                subsets = [tuple(items[:i] + items[i + 1 :]) for i in range(len(items))]
                for subset in subsets:
                    if subset in stable_patterns and subset not in to_remove:
                        rr_subset = stable_patterns[subset]["relative_risk"]
                        if abs(rr_pattern - rr_subset) < redundancy_threshold * rr_subset:
                            to_remove.add(pattern)
                            break
        return {p: stats for p, stats in stable_patterns.items() if p not in to_remove}

    def build_factor_library(self, reduced_patterns: Dict[Tuple[str, ...], dict]) -> List[InducingFactor]:
        library: List[InducingFactor] = []
        for pattern, stats in reduced_patterns.items():
            stability = stats["stability"]
            score = stats["support_event"] * stats["relative_risk"] * stability
            library.append(
                InducingFactor(
                    factor_id="",
                    name=pattern_name(pattern),
                    semantics=pattern_semantics(pattern),
                    items=pattern,
                    support_event=stats["support_event"],
                    support_background=stats["support_background"],
                    relative_risk=stats["relative_risk"],
                    stability=stability,
                    score=score,
                )
            )
        library.sort(key=lambda item: item.score, reverse=True)
        for idx, factor in enumerate(library, start=1):
            factor.factor_id = f"IF_{idx:03d}"
        return library

    def to_frame(self, library: List[InducingFactor]) -> pd.DataFrame:
        return pd.DataFrame([{**asdict(factor), "items": " | ".join(factor.items)} for factor in library])

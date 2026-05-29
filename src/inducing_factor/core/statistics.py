import numpy as np
from scipy import stats
from typing import List, Tuple


def relative_risk_ci(
    count_event: int,
    n_event: int,
    count_bg: int,
    n_bg: int,
    alpha: float = 0.05,
    epsilon: float = 0.5,
) -> Tuple[float, float, float, float]:
    count_event_adj = count_event + epsilon
    count_bg_adj = max(count_bg, 0) + epsilon
    p_event = count_event_adj / (n_event + epsilon)
    p_bg = count_bg_adj / (n_bg + epsilon)
    rr = p_event / p_bg

    se_ln_rr = np.sqrt(1.0 / count_event_adj + 1.0 / count_bg_adj - 1.0 / (n_event + epsilon) - 1.0 / (n_bg + epsilon))
    se_ln_rr = max(se_ln_rr, 1e-10)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.exp(np.log(rr) - z * se_ln_rr)
    ci_hi = np.exp(np.log(rr) + z * se_ln_rr)

    oddsratio, p_value = fisher_exact_test(count_event, n_event, count_bg, n_bg)
    return float(rr), float(ci_lo), float(ci_hi), float(p_value)


def fisher_exact_test(
    count_event: int,
    n_event: int,
    count_bg: int,
    n_bg: int,
) -> Tuple[float, float]:
    a = count_event
    b = n_event - count_event
    c = count_bg
    d = n_bg - count_bg
    table = np.array([[a, b], [c, d]], dtype=np.int64)
    oddsratio, p_value = stats.fisher_exact(table, alternative="greater")
    return float(oddsratio), float(p_value)


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    threshold = 0.0
    for rank_i, (orig_idx, p) in enumerate(indexed, start=1):
        bh_threshold = alpha * rank_i / n
        if p <= bh_threshold:
            threshold = bh_threshold
    for orig_idx, p in indexed:
        if p <= threshold:
            rejected[orig_idx] = True
    return rejected


def true_relative_risk(
    count_pattern_event: int,
    count_pattern_bg: int,
    n_event: int,
    n_bg: int,
    epsilon: float = 0.5,
) -> float:
    count_pattern_bg_adj = max(count_pattern_bg, 0) + epsilon
    n_pattern = count_pattern_event + count_pattern_bg_adj
    n_not_pattern = n_event + n_bg - count_pattern_event - max(count_pattern_bg, 0)
    p_event_given_pattern = (count_pattern_event + epsilon) / (n_pattern + epsilon)
    p_event_given_not_pattern = (
        (max(n_event - count_pattern_event, 0) + epsilon)
        / (max(n_not_pattern, 1) + epsilon)
    )
    return float(p_event_given_pattern / p_event_given_not_pattern)

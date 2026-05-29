import numpy as np
from inducing_factor.core.statistics import (
    relative_risk_ci,
    fisher_exact_test,
    benjamini_hochberg,
    true_relative_risk,
)


def test_relative_risk_ci_basic():
    rr, ci_lo, ci_hi, p_value = relative_risk_ci(
        count_event=100, n_event=500, count_bg=30, n_bg=1000
    )
    assert rr > 1.0
    assert ci_lo < rr < ci_hi
    assert p_value < 0.05


def test_relative_risk_ci_zero_bg():
    rr, ci_lo, ci_hi, p_value = relative_risk_ci(
        count_event=10, n_event=500, count_bg=0, n_bg=1000
    )
    assert rr > 1.0
    assert ci_lo > 1.0


def test_fisher_exact_test_significant():
    _, p_value = fisher_exact_test(
        count_event=80, n_event=200, count_bg=20, n_bg=200
    )
    assert p_value < 0.01


def test_fisher_exact_test_not_significant():
    _, p_value = fisher_exact_test(
        count_event=50, n_event=200, count_bg=45, n_bg=200
    )
    assert p_value > 0.05


def test_benjamini_hochberg_basic():
    p_values = [0.001, 0.01, 0.03, 0.04, 0.05, 0.08, 0.12, 0.5, 0.8, 1.0]
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert rejected[0] is True
    assert rejected[-1] is False


def test_benjamini_hochberg_all_significant():
    p_values = [0.001, 0.002, 0.003, 0.004, 0.005]
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert all(rejected)


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg([], alpha=0.05) == []


def test_true_relative_risk():
    rr = true_relative_risk(
        count_pattern_event=80, count_pattern_bg=30, n_event=500, n_bg=1000
    )
    assert rr > 1.0


def test_true_relative_risk_equals_likelihood_ratio_different():
    lr_style = (80 / 500) / (30 / 1000)
    true_rr = true_relative_risk(80, 30, 500, 1000)
    assert abs(lr_style - true_rr) > 0.01
